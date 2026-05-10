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

constexpr auto const kTracing = true;

template <typename... Args>
void trace_alt(char const* fmt_str, Args&&... args) {
  if constexpr (kTracing) {
    fmt::print(std::cout, fmt::runtime(fmt_str), std::forward<Args>(args)...);
  }
}

namespace {

// Offsets for a leg-alternative boundary that connects to another transit
// leg via timetable transfers — the boundary stop plus all footpath-
// reachable neighbours, each with the appropriate transfer time.
std::vector<offset> footpath_offsets(timetable const& tt,
                                     query const& q,
                                     location_idx_t const l,
                                     auto const& fps) {
  auto v = std::vector<offset>{};
  v.emplace_back(l,
                 adjusted_transfer_time(q.transfer_time_settings_,
                                        tt.locations_.transfer_time_[l]),
                 transport_mode_id_t{0});
  for (auto const& fp : fps) {
    v.emplace_back(
        fp.target(),
        adjusted_transfer_time(q.transfer_time_settings_, fp.duration()),
        transport_mode_id_t{0});
  }
  return v;
}

td_offsets_t td_footpath_offsets(
    rt_timetable const* rtt,
    query const& q,
    location_idx_t const l,
    array<bitvec_map<location_idx_t>, kNProfiles> const& has_td,
    array<vecvec<location_idx_t, td_footpath>, kNProfiles> const& td_fps) {
  auto m = td_offsets_t{};
  if (rtt == nullptr || q.prf_idx_ == 0U ||
      to_idx(l) >= has_td[q.prf_idx_].size() || !has_td[q.prf_idx_][l]) {
    return m;
  }
  for (auto const& tdfp : td_fps[q.prf_idx_][l]) {
    m[tdfp.target_].push_back(
        td_offset{tdfp.valid_from_, tdfp.duration_, transport_mode_id_t{0}});
  }
  return m;
}

// Offsets for a leg-alternative boundary that represents the journey's
// own start/destination: every equivalent location (per the match mode)
// is acceptable at no extra cost.
std::vector<offset> match_mode_offsets(timetable const& tt,
                                       location_match_mode const mode,
                                       location_idx_t const l) {
  auto v = std::vector<offset>{};
  for_each_meta(tt, mode, l, [&](location_idx_t const eq) {
    v.emplace_back(eq, duration_t{0}, transport_mode_id_t{0});
  });
  return v;
}

}  // namespace

query make_alternative_query(timetable const& tt,
                             rt_timetable const* rtt,
                             query const& q,
                             location_idx_t const from,
                             location_idx_t const to) {
  return query{
      .start_ = footpath_offsets(
          tt, q, from, tt.locations_.footpaths_out_[q.prf_idx_][from]),
      .destination_ = footpath_offsets(
          tt, q, to, tt.locations_.footpaths_in_[q.prf_idx_][to]),
      .td_start_ =
          rtt != nullptr
              ? td_footpath_offsets(rtt, q, from, rtt->has_td_footpaths_out_,
                                    rtt->td_footpaths_out_)
              : td_offsets_t{},
      .td_dest_ = rtt != nullptr ? td_footpath_offsets(
                                       rtt, q, to, rtt->has_td_footpaths_in_,
                                       rtt->td_footpaths_in_)
                                 : td_offsets_t{},
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

  // Without a surrounding transit leg, fall back to the original transit's
  // boarding/alighting stop as the alternative's boundary — the journey's
  // own front/back leg may originate at a special intermodal station
  // (kStart/kEnd) that has no footpaths attached, which would prevent the
  // direct search from finding any alternative.
  auto const from = has_prev ? prev_it->to_ : tl.from_;
  auto const to = has_next ? next_it->from_ : tl.to_;
  // Forward path: cursor starts here. Backward path: only used for
  // termination (we rely on max_alternatives instead).
  auto const from_arr = has_prev ? prev_it->arr_time_ : j.start_time_;
  auto const to_dep = has_next ? next_it->dep_time_ : tl.arr_time_;
  trace_alt(
      "[leg_alt]   from={}, to={}, from_arr={}, to_dep={}, "
      "start_match_mode={}, dest_match_mode={}\n",
      from, to, from_arr, to_dep, static_cast<int>(q.start_match_mode_),
      static_cast<int>(q.dest_match_mode_));

  // On the "open" side (no surrounding transit) the boundary represents
  // the journey's origin/destination — honour the original query's match
  // mode so e.g. kEquivalent accepts all equivalent stops at offset 0
  // instead of only the footpath-reachable ones.
  auto const from_offsets =
      has_prev
          ? footpath_offsets(tt, q, from,
                             tt.locations_.footpaths_out_[q.prf_idx_][from])
          : match_mode_offsets(tt, q.start_match_mode_, from);
  auto const to_offsets =
      has_next ? footpath_offsets(tt, q, to,
                                  tt.locations_.footpaths_in_[q.prf_idx_][to])
               : match_mode_offsets(tt, q.dest_match_mode_, to);
  auto const from_td =
      has_prev && rtt != nullptr
          ? td_footpath_offsets(rtt, q, from, rtt->has_td_footpaths_out_,
                                rtt->td_footpaths_out_)
          : td_offsets_t{};
  auto const to_td =
      has_next && rtt != nullptr
          ? td_footpath_offsets(rtt, q, to, rtt->has_td_footpaths_in_,
                                rtt->td_footpaths_in_)
          : td_offsets_t{};

  // direct.cc's `for_each_pair<kBwd>` iterates routes in reverse and treats
  // `q.start_` as the search start (= journey destination) and
  // `q.destination_` as the search end (= journey origin). For the kBwd
  // path we therefore swap the two offset sets so the underlying search
  // sees the journey direction correctly.
  auto const swap_for_bwd = !has_prev && has_next;
  auto const direct_query = query{
      .start_ = swap_for_bwd ? to_offsets : from_offsets,
      .destination_ = swap_for_bwd ? from_offsets : to_offsets,
      .td_start_ = swap_for_bwd ? to_td : from_td,
      .td_dest_ = swap_for_bwd ? from_td : to_td,
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
  auto const make_journey = [&](std::array<journey::leg, 3> legs) {
    // direct.cc emits the ingress/egress legs as `offset` from/to the
    // intermodal special stations kStart/kEnd. For a leg-alternative the
    // boundaries are the real surrounding locations and the walks are
    // ordinary transfer footpaths — rewrite them to match.
    auto const ingress_dur = duration_t{static_cast<duration_t::rep>(
        (legs[0].arr_time_ - legs[0].dep_time_).count())};
    auto const egress_dur = duration_t{static_cast<duration_t::rep>(
        (legs[2].arr_time_ - legs[2].dep_time_).count())};
    legs[0].from_ = from;
    legs[0].uses_ = footpath{legs[0].to_, ingress_dur};
    legs[2].to_ = to;
    legs[2].uses_ = footpath{to, egress_dur};

    auto alt = journey{};
    alt.start_time_ = legs[0].dep_time_;
    alt.dest_time_ = legs[2].arr_time_;
    alt.dest_ = legs[2].to_;
    alt.transfers_ = 0U;
    alt.legs_.assign(begin(legs), end(legs));
    return alt;
  };
  auto const is_original = [&](std::array<journey::leg, 3> const& legs) {
    auto const* alt_ree = std::get_if<journey::run_enter_exit>(&legs[1].uses_);
    return alt_ree != nullptr && *alt_ree == original_ree;
  };

  auto alternatives = std::vector<journey>{};
  alternatives.reserve(max_alternatives);
  if (!has_prev && has_next) {
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
                legs[0].dep_time_, legs[2].arr_time_);
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
                legs[0].dep_time_, legs[2].arr_time_);
      // Only apply the arrival cap when there IS a next transit leg to
      // catch — otherwise (last leg / single transit) the user is free to
      // arrive later than the original.
      if (has_next && legs[2].arr_time_ > to_dep) {
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
