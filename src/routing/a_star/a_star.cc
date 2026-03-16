#include "nigiri/routing/a_star/a_star.h"

#include <ranges>

#include "fmt/ranges.h"

#include "utl/enumerate.h"
#include "utl/raii.h"

#include "nigiri/common/dial.h"
#include "nigiri/for_each_meta.h"
#include "nigiri/routing/get_earliest_transport.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/tb/tb_data.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri {
namespace routing {

template <bool UseLowerBounds>
a_star<UseLowerBounds>::a_star(
    timetable const& tt,
    rt_timetable const*,
    a_star_state& state,
    bitvec const& is_dest,
    std::array<bitvec, kMaxVias> const&,
    std::vector<std::uint16_t> const& dist_to_dest,
    hash_map<location_idx_t, std::vector<td_offset>> const&,
    std::vector<std::uint16_t> const& lb,
    std::vector<via_stop> const&,
    day_idx_t base,
    clasz_mask_t,
    bool,
    bool,
    bool,
    transfer_time_settings tts)
    : tt_{tt},
      state_{state},
      is_dest_{is_dest},
      dist_to_dest_{dist_to_dest},
      lb_{lb},
      base_{base - QUERY_DAY_SHIFT} {
  stats_.use_lower_bounds_ = UseLowerBounds;
  state_.use_lower_bounds_ = UseLowerBounds;
  state_.transfer_factor_ = tts.factor_;

  auto const mark_dest_segments = [&](location_idx_t const l,
                                      duration_t const d) {
    for (auto const r : tt_.location_routes_[l]) {
      auto const stop_seq = tt_.route_location_seq_[r];
      for (auto i = stop_idx_t{1U}; i != stop_seq.size(); ++i) {
        auto const stp = stop{stop_seq[i]};
        if (stp.location_idx() != l || !stp.out_allowed()) {
          continue;
        }

        for (auto const t : tt_.route_transport_ranges_[r]) {
          auto const segment = state_.tbd_.transport_first_segment_[t] + i - 1;
          state_.end_reachable_.set(segment, true);

          auto const it = state_.dist_to_dest_.find(segment);
          if (it == end(state_.dist_to_dest_)) {
            state_.dist_to_dest_.emplace_hint(it, segment, d);
          } else {
            it->second = std::min(it->second, d);
          }
        }
      }
    }
  };

  if (dist_to_dest.empty()) /* Destination is stop. */ {
    is_dest_.for_each_set_bit([&](std::size_t const i) {
      auto const l = location_idx_t{i};
      as_debug("{} is dest!", location{tt_, l});
      mark_dest_segments(l, duration_t{0U});
      for (auto const fp :
           tt_.locations_.footpaths_in_[state_.tbd_.prf_idx_][l]) {
        mark_dest_segments(fp.target(), fp.duration());
      }
    });
  } else /* Destination is coordinate. */ {
    for (auto const [l_idx, dist] : utl::enumerate(dist_to_dest_)) {
      if (dist != kUnreachable) {
        mark_dest_segments(location_idx_t{l_idx}, duration_t{dist});
      }
    }
  }
};

template <bool UseLowerBounds>
void a_star<UseLowerBounds>::execute(unixtime_t const start_time,
                                     std::uint8_t const max_transfers,
                                     unixtime_t const worst_time_at_dest,
                                     profile_idx_t const,
                                     pareto_set<journey>& results) {
  auto const results_size_before = results.size();
  auto const start_delta = day_idx_mam(start_time);
  uint16_t const worst_cost =
      std::min(day_idx_mam(worst_time_at_dest) - start_delta,
               delta(maxASTravelTime))
          .count();
  state_.setup(start_delta, worst_cost, max_transfers);

  uint16_t current_best_arrival =
      worst_cost + std::ceil(max_transfers * state_.transfer_factor_);
  while (!state_.pq_.empty()) {
    auto const& current = state_.pq_.top();
    auto bucket = state_.cost_function(current);
    if (bucket >= current_best_arrival) {
      return;
    }
    state_.pq_.pop();
    auto const& segment = current.segment_;

    if (state_.settled_segments_.test(segment)) {
      continue;
    }
    state_.settled_segments_.set(segment, true);
    ++stats_.n_segments_reached_;

    as_debug("Visiting segment {} with transfers {} and bucket {}", segment,
             current.transfers_, bucket);

    if (state_.end_reachable_.test(segment)) {
      ++stats_.n_dest_segments_reached_;
      auto const it = state_.dist_to_dest_.find(segment);
      if (it != end(state_.dist_to_dest_)) {
        bucket += it->second.count();
      }
      if (bucket >=
          std::min(current_best_arrival,
                   static_cast<uint16_t>(
                       (worst_cost + std::ceil(current.transfers_ *
                                               state_.transfer_factor_))))) {
        continue;
      }
      current_best_arrival = bucket;
      auto dest_time = to_unixtime(state_.arrival_time_.at(segment));

      if (it != end(state_.dist_to_dest_)) {
        dest_time += it->second;
      }
      as_debug("Reached destination via segment {} at time: {}", segment,
               dest_time);

      if (results.size() > results_size_before) {
        results.erase(std::prev(results.end()));
      }
      results.add(
          {.legs_{},
           .start_time_ = start_time,
           .dest_time_ = dest_time,
           .dest_ = location_idx_t::invalid(),
           .transfers_ = static_cast<std::uint8_t>(current.transfers_)});
    }

    auto const handle_new_segment = [&](segment_idx_t s, stop_idx_t to,
                                        transport t, bool transfer = false) {
      auto const next_stop_arr = event_day_idx_mam(t, to, event_type::kArr);
      if constexpr (UseLowerBounds) {
        state_.lb_.emplace(
            s, lb_.at(to_idx(stop{
                   tt_.route_location_seq_[tt_.transport_route_[t.t_idx_]][to]}
                                 .location_idx())));
      }
      state_.update_segment(
          s, next_stop_arr, segment,
          transfer ? current.transfers_ + 1 : current.transfers_);
    };

    // Handle next segment of transport if exists
    auto const transport_idx_current = state_.tbd_.segment_transports_[segment];
    auto const next_segment = segment + 1;
    auto const current_stop_idx = static_cast<stop_idx_t>(
        to_idx(segment -
               state_.tbd_.transport_first_segment_[transport_idx_current]) +
        1);
    auto const current_transport_offset = static_cast<day_idx_t>(
        state_.arrival_time_.at(segment).days() -
        tt_.event_mam(transport_idx_current, current_stop_idx, event_type::kArr)
            .days());
    if (state_.tbd_.get_segment_range(transport_idx_current)
            .contains(next_segment)) [[likely]] {
      handle_new_segment(
          next_segment, current_stop_idx + 1,
          transport{transport_idx_current, current_transport_offset});
    }

    // Handle transfers
    if (current.transfers_ >= max_transfers) [[unlikely]] {
      as_debug("Max transfers reached at segment {}", segment);
      stats_.max_transfers_reached_ = true;
      continue;
    }

    for (auto const& transfer : state_.tbd_.segment_transfers_[segment]) {
      auto const new_segment = transfer.to_segment_;
      if (state_.settled_segments_.test(new_segment)) {
        continue;
      }

      if (!state_.tbd_.bitfields_[transfer.traffic_days_].test(
              to_idx(current_transport_offset + base_))) {
        continue;
      }

      auto const transport_idx_new =
          state_.tbd_.segment_transports_[new_segment];
      handle_new_segment(
          new_segment,
          static_cast<stop_idx_t>(to_idx(
              new_segment -
              state_.tbd_.transport_first_segment_[transport_idx_new] + 1)),
          transport{transport_idx_new,
                    current_transport_offset + transfer.get_day_offset()},
          true);
    }
  }
  stats_.no_journey_found_ = results.empty();
}

template <bool UseLowerBounds>
void a_star<UseLowerBounds>::add_start(location_idx_t l, unixtime_t t) {
  auto const [day, mam] = tt_.day_idx_mam(t);
  for (auto const r : tt_.location_routes_[l]) {
    auto const stop_seq = tt_.route_location_seq_[r];
    for (auto i = stop_idx_t{0U}; i < stop_seq.size() - 1; ++i) {
      auto const stp = stop{stop_seq[i]};
      if (!stp.in_allowed() || stp.location_idx() != l) {
        continue;
      }

      auto et = get_earliest_transport<direction::kForward>(
          tt_, tt_, 0U, r, i, day, mam, stp.location_idx(),
          [](day_idx_t, std::int16_t) { return false; });
      if (!et.is_valid()) {
        continue;
      }

      auto const transport_day_offset = et.day_ - base_;
      if (transport_day_offset.v_ < 0 ||
          transport_day_offset.v_ > kASMaxDayOffset) {
        continue;
      }
      auto const arr_time =
          event_day_idx_mam(transport{et.t_idx_, transport_day_offset},
                            static_cast<stop_idx_t>(i + 1), event_type::kArr);
      auto const start_segment =
          state_.tbd_.transport_first_segment_[et.t_idx_] + i;
      state_.arrival_time_.emplace(start_segment, arr_time);
      state_.start_segments_.set(start_segment);
      as_debug("Adding start segment {} for location {}", start_segment,
               location{tt_, l});
      if constexpr (UseLowerBounds) {
        state_.lb_.emplace(
            start_segment,
            lb_.at(to_idx(
                stop{tt_.route_location_seq_[r][i + 1]}.location_idx())));
      }
    }
  }
}

template <bool UseLowerBounds>
void a_star<UseLowerBounds>::reconstruct(query const& q, journey& j) const {
  UTL_FINALLY([&]() { std::reverse(begin(j.legs_), end(j.legs_)); });

  auto const has_offset = [&](std::vector<offset> const& offsets,
                              location_match_mode const match_mode,
                              location_idx_t const l) {
    return utl::any_of(offsets, [&](offset const& o) {
      return matches(tt_, match_mode, o.target(), l);
    });
  };

  auto const get_transport_info = [&](segment_idx_t const s,
                                      event_type const ev_type)
      -> std::tuple<transport, stop_idx_t, location_idx_t, unixtime_t> {
    auto const arr_time = state_.arrival_time_.at(s);
    auto const t = state_.tbd_.segment_transports_[s];
    auto i = static_cast<stop_idx_t>(
        to_idx(s - state_.tbd_.transport_first_segment_.at(t) + 1));
    auto const d =
        base_ + arr_time.days() - tt_.event_mam(t, i, event_type::kArr).days();
    auto const loc_seq = tt_.route_location_seq_[tt_.transport_route_[t]];
    if (ev_type == event_type::kDep) {
      --i;
    }
    return {{t, d},
            i,
            stop{loc_seq[i]}.location_idx(),
            ev_type == event_type::kArr ? to_unixtime(arr_time)
                                        : tt_.event_time({t, d}, i, ev_type)};
  };

  auto const get_fp = [&](location_idx_t const from, location_idx_t const to) {
    if (from == to) {
      return footpath{to, tt_.locations_.transfer_time_[from]};
    }
    auto const from_fps =
        tt_.locations_.footpaths_out_[state_.tbd_.prf_idx_][from];
    auto const it = utl::find_if(
        from_fps, [&](footpath const& fp) { return fp.target() == to; });
    utl::verify(it != end(from_fps),
                "as  reconstruct: footpath from {} to {} not found",
                location{tt_, from}, location{tt_, to});
    return *it;
  };

  // ==================
  // (1) Last leg
  // ------------------
  auto dest_segment = segment_idx_t::invalid();

  if (q.dest_match_mode_ == location_match_mode::kIntermodal) {
    for (auto arr_candidate_segment = state_.end_reachable_.next_set_bit(0);
         arr_candidate_segment != std::nullopt;
         arr_candidate_segment = state_.end_reachable_.next_set_bit(
             static_cast<uint32_t>(arr_candidate_segment.value()) + 1)) {

      if (!state_.settled_segments_.test(arr_candidate_segment.value())) {
        continue;
      }

      auto const offset = state_.dist_to_dest_.at(arr_candidate_segment);
      auto const [_, _1, arr_l, arr_time] =
          get_transport_info(arr_candidate_segment.value(), event_type::kArr);
      if (arr_time + offset != j.arrival_time()) {
        continue;
      }

      auto const offset_it =
          utl::find_if(q.destination_, [&](routing::offset const& o) {
            return o.target() == arr_l && o.duration() == offset;
          });
      utl::verify(offset_it != end(q.destination_), "offset not found");
      j.legs_.push_back({direction::kForward, arr_l,
                         get_special_station(special_station::kEnd), arr_time,
                         j.arrival_time(), *offset_it});
      dest_segment = segment_idx_t{arr_candidate_segment.value()};
      break;
    }
  } else /* Stop destination -> footpath or direct arrival */ {

    for (auto arr_candidate_segment = state_.end_reachable_.next_set_bit(0);
         arr_candidate_segment != std::nullopt;
         arr_candidate_segment = state_.end_reachable_.next_set_bit(
             static_cast<uint32_t>(arr_candidate_segment.value()) + 1)) {
      as_debug("dest candidate {}", arr_candidate_segment.value());

      if (!state_.settled_segments_.test(arr_candidate_segment.value())) {
        as_debug("no dest candidate {} => has not been settled",
                 arr_candidate_segment.value());
        continue;
      }

      auto const [_, dep_l, arr_l, arr_time] =
          get_transport_info(arr_candidate_segment.value(), event_type::kArr);

      auto const handle_fp = [&](footpath const& fp) {
        if (arr_time + fp.duration() != j.arrival_time() ||
            !has_offset(q.destination_, q.dest_match_mode_, fp.target())) {
          as_debug(
              "no dest candidate {} arr_l={}: arr_time={} + fp.duration={} = "
              "{} != j.arrival_time={}",
              arr_candidate_segment.value(), arr_l, arr_time, fp.duration(),
              location{tt_, arr_l}, arr_time, fp.duration(),
              arr_time + fp.duration(), j.arrival_time());
          return false;
        }
        as_debug("FOUND!");
        j.dest_ = fp.target();
        j.legs_.emplace_back(journey::leg{direction::kForward, arr_l,
                                          fp.target(), arr_time,
                                          j.arrival_time(), fp});
        return true;
      };

      if (handle_fp(footpath{arr_l, duration_t{0}})) {
        dest_segment = segment_idx_t{arr_candidate_segment.value()};
        break;
      }

      for (auto const fp :
           tt_.locations_.footpaths_out_[state_.tbd_.prf_idx_][arr_l]) {
        if (handle_fp(fp)) {
          dest_segment = segment_idx_t{arr_candidate_segment.value()};
          break;
        }
      }
      if (dest_segment != segment_idx_t::invalid()) {
        break;
      }
    }
  }

  assert(dest_segment != segment_idx_t::invalid() &&
         "no dest segment found in reconstruct");

  // ==================
  // (2) Transport legs
  // ------------------
  auto current = dest_segment;
  auto [transport, arr_stop_idx, arr_l, arr_time] =
      get_transport_info(dest_segment, event_type::kArr);
  while (true) {
    auto const pred = state_.pred_table_.at(current);
    if (pred != state_.startSegmentPredecessor &&
        transport.t_idx_ == state_.tbd_.segment_transports_[pred]) {
      current = pred;
      continue;
    }

    if (j.legs_.size() != 1) {
      auto const fp = get_fp(arr_l, j.legs_.back().from_);
      j.legs_.emplace_back(journey::leg{direction::kForward, arr_l,
                                        j.legs_.back().from_, arr_time,
                                        arr_time + fp.duration(), fp});
    }
    auto const [_, dep_stop_idx, dep_l, dep_time] =
        get_transport_info(current, event_type::kDep);
    j.legs_.emplace_back(journey::leg{
        direction::kForward, dep_l, arr_l, dep_time, arr_time,
        journey::run_enter_exit{
            rt::run{
                .t_ = transport,
                .stop_range_ = {static_cast<stop_idx_t>(0U),
                                static_cast<stop_idx_t>(
                                    tt_.route_location_seq_
                                        [tt_.transport_route_[transport.t_idx_]]
                                            .size())}},
            dep_stop_idx, arr_stop_idx}});
    if (pred == state_.startSegmentPredecessor) {
      break;
    }

    current = pred;
    std::tie(transport, arr_stop_idx, arr_l, arr_time) =
        get_transport_info(current, event_type::kArr);
  }

  // ==================
  // (3) First leg
  // ------------------
  assert(!j.legs_.empty());
  auto const start_time = j.start_time_;
  auto const first_dep_l = j.legs_.back().from_;
  auto const first_dep_time = j.legs_.back().dep_time_;
  if (q.start_match_mode_ == location_match_mode::kIntermodal) {
    auto const offset_it =
        utl::find_if(q.start_, [&](routing::offset const& o) {
          return o.target() == first_dep_l &&
                 start_time + o.duration() <= first_dep_time;
        });
    utl::verify(
        offset_it != end(q.start_),
        "no start offset found start_time={}, first_dep={}@{}, offsets={}",
        start_time, first_dep_time, location{tt_, first_dep_l},
        q.start_ | std::views::transform([&](offset const& x) {
          return std::pair{location{tt_, x.target_}, x.duration()};
        }));
    j.legs_.push_back(journey::leg{
        direction::kForward, get_special_station(special_station::kStart),
        first_dep_l, first_dep_time - offset_it->duration(), first_dep_time,
        *offset_it});
  } else {
    for (auto const fp :
         tt_.locations_.footpaths_in_[state_.tbd_.prf_idx_][first_dep_l]) {
      if (start_time + fp.duration() <= first_dep_time &&
          has_offset(q.start_, q.start_match_mode_, fp.target())) {
        j.legs_.push_back({direction::kForward, fp.target(), first_dep_l,
                           first_dep_time - fp.duration(), first_dep_time, fp});
        break;
      }
    }
  }
}

template struct a_star<false>;
template struct a_star<true>;

}  // namespace routing
}  // namespace nigiri