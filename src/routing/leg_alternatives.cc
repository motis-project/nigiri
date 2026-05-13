#include "nigiri/routing/leg_alternatives.h"

#include <algorithm>
#include <iostream>
#include <iterator>

#include "fmt/core.h"
#include "fmt/ostream.h"

#include "nigiri/for_each_meta.h"
#include "nigiri/routing/direct.h"
#include "nigiri/routing/transfer_time_settings.h"

namespace nigiri::routing {

constexpr auto const kTracing = false;

template <typename... Args>
void trace_alt(char const* fmt_str, Args&&... args) {
  if constexpr (kTracing) {
    fmt::print(std::cout, fmt::runtime(fmt_str), std::forward<Args>(args)...);
  }
}

query make_alternative_query(timetable const&,
                             rt_timetable const*,
                             query const& q,
                             location_idx_t const from,
                             location_idx_t const to) {
  return query{
      .start_match_mode_ = location_match_mode::kExact,
      .dest_match_mode_ = location_match_mode::kExact,
      .use_start_footpaths_ = true,
      .start_ = {{from, 0_minutes, transport_mode_id_t{0}}},
      .destination_ = {{to, 0_minutes, transport_mode_id_t{0}}},
      .prf_idx_ = q.prf_idx_,
      .allowed_claszes_ = q.allowed_claszes_,
      .require_bike_transport_ = q.require_bike_transport_,
      .require_car_transport_ = q.require_car_transport_,
      .transfer_time_settings_ = q.transfer_time_settings_,
  };
}

std::vector<journey> get_leg_alternatives(timetable const& tt,
                                          rt_timetable const* rtt,
                                          query const& q,
                                          journey const& j,
                                          std::size_t const leg_idx,
                                          std::size_t const max_alternatives) {
  trace_alt(
      "[leg_alt] get_leg_alternatives leg_idx={}, max={}, n_legs={}, "
      "j.start_time={}, j.dest_time={}\n",
      leg_idx, max_alternatives, j.legs_.size(), j.start_time_, j.dest_time_);
  if (max_alternatives == 0U || leg_idx >= j.legs_.size() ||
      !std::holds_alternative<journey::run_enter_exit>(
          j.legs_[leg_idx].uses_)) {
    trace_alt(
        "[leg_alt]   -> skip (max=0 or leg_idx OOB or leg not transit)\n");
    return {};
  }
  auto const& tl = j.legs_[leg_idx];
  auto const& original_ree = std::get<journey::run_enter_exit>(tl.uses_);
  trace_alt(
      "[leg_alt]   tl.from={}, tl.to={}, tl.dep={}, tl.arr={}, "
      "tl.uses=run_enter_exit\n",
      tl.from_, tl.to_, tl.dep_time_, tl.arr_time_);

  auto const find_transit = [&](auto first, auto last) {
    return std::find_if(first, last, [](journey::leg const& l) {
      return std::holds_alternative<journey::run_enter_exit>(l.uses_);
    });
  };

  auto const prev_it =
      find_transit(std::reverse_iterator{begin(j.legs_) +
                                         static_cast<std::ptrdiff_t>(leg_idx)},
                   std::rend(j.legs_));
  auto const next_it = find_transit(
      begin(j.legs_) + static_cast<std::ptrdiff_t>(leg_idx) + 1, end(j.legs_));

  auto const has_prev = prev_it != std::rend(j.legs_);
  auto const has_next = next_it != end(j.legs_);
  trace_alt("[leg_alt]   has_prev={}, has_next={}\n", has_prev, has_next);

  // Forward path: cursor starts here. Backward path: only used for
  // termination (we rely on max_alternatives instead).
  auto const from_arr =
      has_prev ? prev_it->arr_time_ : j.legs_.front().dep_time_;
  auto const to_dep = has_next ? next_it->dep_time_ : j.legs_.back().arr_time_;

  // Inner boundary (transit leg before/after the alternative): hand
  // direct.cc the adjacent transit leg's exit/entry stop and let it
  // expand the timetable's static + td footpaths via
  // `use_start_footpaths_`. Open boundary (no surrounding transit
  // leg): inherit the original query's start/destination so the
  // alternative can reach the journey's actual origin/destination —
  // preserving the intermodal transport mode (BIKE, RENTAL, ...) on
  // that side. direct.cc handles kBwd internally so we no longer
  // pre-swap start/destination here.
  auto const inner_from = std::vector<offset>{
      {prev_it != std::rend(j.legs_) ? prev_it->to_ : location_idx_t{},
       0_minutes, transport_mode_id_t{0}}};
  auto const inner_to = std::vector<offset>{
      {next_it != end(j.legs_) ? next_it->from_ : location_idx_t{}, 0_minutes,
       transport_mode_id_t{0}}};

  auto direct_query = query{
      .start_match_mode_ =
          has_prev ? location_match_mode::kExact : q.start_match_mode_,
      .dest_match_mode_ =
          has_next ? location_match_mode::kExact : q.dest_match_mode_,
      .use_start_footpaths_ = has_prev ? true : q.use_start_footpaths_,
      .start_ = has_prev ? inner_from : q.start_,
      .destination_ = has_next ? inner_to : q.destination_,
      .td_start_ = has_prev ? td_offsets_t{} : q.td_start_,
      .td_dest_ = has_next ? td_offsets_t{} : q.td_dest_,
      .prf_idx_ = q.prf_idx_,
      .allowed_claszes_ = q.allowed_claszes_,
      .require_bike_transport_ = q.require_bike_transport_,
      .require_car_transport_ = q.require_car_transport_,
      .transfer_time_settings_ = q.transfer_time_settings_,
  };
  trace_alt("[leg_alt]   direct_query.start_ ({} entries):\n",
            direct_query.start_.size());
  for (auto const& o : direct_query.start_) {
    trace_alt("[leg_alt]     - target={}, duration={}, mode={}\n", o.target(),
              o.duration(), o.transport_mode_id_);
  }
  trace_alt("[leg_alt]   direct_query.destination_ ({} entries):\n",
            direct_query.destination_.size());
  for (auto const& o : direct_query.destination_) {
    trace_alt("[leg_alt]     - target={}, duration={}, mode={}\n", o.target(),
              o.duration(), o.transport_mode_id_);
  }

  auto const make_journey = [&](std::vector<journey::leg> legs) {
    auto alt = journey{};
    alt.start_time_ = legs.front().dep_time_;
    alt.dest_time_ = legs.back().arr_time_;
    alt.dest_ = legs.back().to_;
    alt.transfers_ = 0U;
    alt.legs_ = std::move(legs);
    return alt;
  };
  auto const is_original = [&](std::vector<journey::leg> const& legs) {
    auto const transit_it =
        std::find_if(begin(legs), end(legs), [](journey::leg const& l) {
          return std::holds_alternative<journey::run_enter_exit>(l.uses_);
        });
    if (transit_it == end(legs)) {
      return false;
    }
    auto const& alt_ree = std::get<journey::run_enter_exit>(transit_it->uses_);
    return alt_ree == original_ree;
  };

  // First transit leg with a successor (open start, inner end): walk
  // backwards from `to_dep` to surface the LATEST alternatives that
  // still arrive in time. All other shapes walk forwards from
  // `from_arr`.
  auto const iter_backward = !has_prev && has_next;

  auto alternatives = std::vector<journey>{};
  alternatives.reserve(max_alternatives);
  if (iter_backward) {
    // First transit leg with a successor: collect the LATEST alternatives
    // that still arrive in time for the next leg's departure by iterating
    // backward from to_dep.
    trace_alt("[leg_alt]   iterating BACKWARD from to_dep={}\n", to_dep);
    auto cursor = get_direct_journeys<direction::kBackward>(
        tt, rtt, direct_query, to_dep);
    auto yielded = 0U;
    while (cursor && alternatives.size() < max_alternatives) {
      auto legs = cursor();
      ++yielded;
      trace_alt("[leg_alt]     bwd yield #{} dep={} arr={}\n", yielded,
                legs.front().dep_time_, legs.back().arr_time_);
      if (is_original(legs)) {
        trace_alt("[leg_alt]     -> skip: is original transit\n");
        continue;
      }
      alternatives.push_back(make_journey(std::move(legs)));
    }
    trace_alt("[leg_alt]   bwd done: yielded={}, kept={}\n", yielded,
              alternatives.size());
  } else {
    trace_alt(
        "[leg_alt]   iterating FORWARD from from_arr={} (upper_bound={})\n",
        from_arr, has_next ? fmt::format("{}", to_dep) : "<unbounded>");
    auto cursor = get_direct_journeys<direction::kForward>(
        tt, rtt, direct_query, from_arr);
    auto yielded = 0U;
    while (cursor && alternatives.size() < max_alternatives) {
      auto legs = cursor();
      ++yielded;
      trace_alt("[leg_alt]     fwd yield #{} dep={} arr={}\n", yielded,
                legs.front().dep_time_, legs.back().arr_time_);
      // Only apply the arrival cap when there IS a next transit leg to
      // catch — otherwise (last leg / single transit) the user is free to
      // arrive later than the original.
      if (has_next && legs.back().arr_time_ > to_dep) {
        trace_alt("[leg_alt]     -> break: arr above to_dep\n");
        break;
      }
      if (is_original(legs)) {
        trace_alt("[leg_alt]     -> skip: is original transit\n");
        continue;
      }
      alternatives.push_back(make_journey(std::move(legs)));
    }
    trace_alt("[leg_alt]   fwd done: yielded={}, kept={}\n", yielded,
              alternatives.size());
  }
  trace_alt("[leg_alt] returning {} alternatives\n", alternatives.size());
  return alternatives;
}

}  // namespace nigiri::routing
