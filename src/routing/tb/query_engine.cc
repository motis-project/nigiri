#include "nigiri/routing/tb/query_engine.h"

#include <ranges>

#include "nigiri/for_each_meta.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/tb/query_engine.h"
#include "nigiri/routing/tb/settings.h"
#include "nigiri/routing/tb/transport_segment.h"
#include "nigiri/rt/frun.h"
#include "nigiri/special_stations.h"

#define TB_CACHE_PRESSURE_REDUCTION

namespace nigiri::routing::tb {

template <bool UseLowerBounds>
query_engine<UseLowerBounds>::query_engine(
    timetable const& tt,
    rt_timetable const* rtt,
    query_state& state,
    bitvec& is_dest,
    std::optional<std::array<bitvec, kMaxVias>> const is_via
    [[maybe_unused]],  // unsupported
    std::vector<std::uint16_t>& dist_to_dest,
    std::optional<hash_map<location_idx_t, std::vector<td_offset>>> const
        td_dist_to_dest [[maybe_unused]],  // unsupported
    std::vector<std::uint16_t>& lb,
    std::optional<std::vector<via_stop>> const via_stops
    [[maybe_unused]],  // unsupported
    day_idx_t const base,
    std::optional<clasz_mask_t> const allowed_claszes
    [[maybe_unused]],  // unsupported
    std::optional<bool> const require_bike_transport
    [[maybe_unused]],  // unsupported
    std::optional<bool> const is_wheelchair [[maybe_unused]],  // unsupported
    std::optional<transfer_time_settings> const tts
    [[maybe_unused]])  // unsupported
    : tt_{tt},
      rtt_{rtt},
      state_{state},
      is_dest_{is_dest},
      dist_to_dest_{dist_to_dest},
      lb_{lb},
      base_{base} {
  stats_.lower_bound_pruning_ = UseLowerBounds;

  // reset state for new query
  state_.reset(base);

  // init l_, i.e., routes that reach the destination at certain stop idx
  if (dist_to_dest.empty()) {
    // create l_entries for station-to-station query
    for (location_idx_t dest{0U}; dest != location_idx_t{is_dest_.size()};
         ++dest) {
      if (is_dest_[dest.v_]) {
        // fill l_
        auto create_l_entry = [this](footpath const& fp) {
          // iterate routes serving source of footpath
          for (auto const route_idx : tt_.location_routes_[fp.target()]) {
            // iterate stop sequence of route
            for (std::uint16_t stop_idx{1U};
                 stop_idx < tt_.route_location_seq_[route_idx].size();
                 ++stop_idx) {
              auto const route_stop =
                  stop{tt_.route_location_seq_[route_idx][stop_idx]};
              if (route_stop.location_idx() == fp.target() &&
                  route_stop.out_allowed()) {
                state_.route_dest_[route_idx.v_].emplace_back(
                    stop_idx, fp.duration().count());
              }
            }
          }
        };
        // virtual reflexive incoming footpath
        create_l_entry(footpath{dest, duration_t{0U}});
        // iterate incoming footpaths of target location
        for (auto const fp :
             tt_.locations_.footpaths_in_[profile_idx_t{0U}][dest]) {
          create_l_entry(fp);
        }
      }
    }
  } else {
    // create l_entries for coord-to-coord query
    for (location_idx_t dest{0U}; dest != location_idx_t{dist_to_dest.size()};
         ++dest) {
      if (dist_to_dest[dest.v_] != std::numeric_limits<std::uint16_t>::max()) {
        // fill l_
        auto create_l_entry = [this, &dest](footpath const& fp) {
          // iterate routes serving source of footpath
          for (auto const route_idx : tt_.location_routes_[fp.target()]) {
            // iterate stop sequence of route
            for (std::uint16_t stop_idx{1U};
                 stop_idx < tt_.route_location_seq_[route_idx].size();
                 ++stop_idx) {
              auto const route_stop =
                  stop{tt_.route_location_seq_[route_idx][stop_idx]};
              if (route_stop.location_idx() == fp.target() &&
                  route_stop.out_allowed()) {
                state_.route_dest_[route_idx.v_].emplace_back(
                    stop_idx, fp.duration().count() + dist_to_dest_[dest.v_]);
              }
            }
          }
        };
        // virtual reflexive incoming footpath
        create_l_entry(footpath{dest, duration_t{0U}});
        // iterate incoming footpaths of target location
        for (auto const fp :
             tt_.locations_.footpaths_in_[profile_idx_t{0U}][dest]) {
          create_l_entry(fp);
        }
      }
    }
  }
}

template <bool UseLowerBounds>
void query_engine<UseLowerBounds>::execute(unixtime_t const start_time,
                                           std::uint8_t const max_transfers,
                                           unixtime_t const worst_time_at_dest,
                                           profile_idx_t const,
                                           pareto_set<journey>& results) {
  // init Q_0
  for (auto const& qs : state_.query_starts_) {
    handle_start(qs);
  }

  // process all Q_n in ascending order, i.e., transport segments reached after
  // n transfers
  std::uint8_t n = 0U;
  for (; n != state_.q_n_.start_.size() && n < max_transfers; ++n) {
    // (1)  destination reached?
    for (auto q_cur = state_.q_n_.start_[n]; q_cur != state_.q_n_.end_[n];
         ++q_cur) {
      seg_dest(start_time, results, worst_time_at_dest, n, state_.q_n_[q_cur]);
    }
    // (2) pruning?
    for (auto q_cur = state_.q_n_.start_[n]; q_cur != state_.q_n_.end_[n];
         ++q_cur) {
      seg_prune(worst_time_at_dest, n, state_.q_n_[q_cur]);
    }
    // (3) process transfers & enqueue segments
    for (auto q_cur = state_.q_n_.start_[n]; q_cur != state_.q_n_.end_[n];
         ++q_cur) {
      seg_transfers(n, q_cur);
    }
  }

  stats_.n_segments_enqueued_ += state_.q_n_.size();
  stats_.empty_n_ = n;
  stats_.max_transfers_reached_ = n == max_transfers;
}

template <bool UseLowerBounds>
void query_engine<UseLowerBounds>::seg_dest(unixtime_t const start_time,
                                            pareto_set<journey>& results,
                                            unixtime_t worst_time_at_dest,
                                            std::uint8_t const n,
                                            transport_segment& seg) {

  // departure time at the start of the transport segment
  auto const tau_dep_t_b = tt_.event_mam(seg.get_transport_idx(),
                                         seg.stop_idx_start_, event_type::kDep)
                               .count();
  auto const tau_dep_t_b_d =
      tt_.event_mam(seg.get_transport_idx(), seg.stop_idx_start_,
                    event_type::kDep)
          .days();
  auto const tau_dep_t_b_tod =
      tt_.event_mam(seg.get_transport_idx(), seg.stop_idx_start_,
                    event_type::kDep)
          .mam();

  // the day index of the segment
  std::int32_t const d_seg = seg.get_transport_day(base_).v_;
  // departure time at start of current transport segment in minutes after
  // midnight on the day of the query
  auto const tau_d =
      (d_seg + tau_dep_t_b_d - base_.v_) * 1440 + tau_dep_t_b_tod;

  // the route index of the current segment
  auto const seg_route_idx = tt_.transport_route_[seg.get_transport_idx()];

  // check if target location is reached from current transport segment
  for (auto const& le : state_.route_dest_[seg_route_idx.v_]) {
    if (seg.stop_idx_start_ < le.stop_idx_ &&
        le.stop_idx_ <= seg.stop_idx_end_) {
      // the time it takes to travel on this transport segment
      auto const travel_time_seg =
          tt_.event_mam(seg.get_transport_idx(), le.stop_idx_, event_type::kArr)
              .count() -
          tau_dep_t_b;
      // the time at which the target location is reached by using the
      // current transport segment
      auto const t_cur = tt_.to_unixtime(
          base_, minutes_after_midnight_t{tau_d + travel_time_seg + le.time_});

      // add journey if it is non-dominated
      if (t_cur < state_.t_min_[n] && t_cur < worst_time_at_dest) {
        state_.t_min_[n] = t_cur;
        // add journey without reconstructing yet
        journey j{};
        j.start_time_ = start_time;
        j.dest_time_ = t_cur;
        j.dest_ = stop{tt_.route_location_seq_[seg_route_idx][le.stop_idx_]}
                      .location_idx();
        j.transfers_ = n;
        // add journey to pareto set (removes dominated entries)
        results.add(std::move(j));
        ++stats_.n_journeys_found_;
      }
    }
  }

  // the time it takes to travel to the next stop of the transport segment
  auto const travel_time_next =
      tt_.event_mam(seg.get_transport_idx(), seg.stop_idx_start_ + 1,
                    event_type::kArr)
          .count() -
      tau_dep_t_b;

  auto const unix_time_next = tt_.to_unixtime(
      base_, minutes_after_midnight_t{tau_d + travel_time_next});

  if constexpr (UseLowerBounds) {
    auto const location_next =
        stop{tt_.route_location_seq_
                 [tt_.transport_route_[seg.get_transport_idx()]]
                 [seg.stop_idx_start_ + 1]}
            .location_idx();
    auto const lb = lb_[location_next.v_];
    // arrival plus lowest possible travel time to destination
    seg.time_prune_ = lb == kUnreachable
                          ? std::numeric_limits<unixtime_t>::max()
                          : unix_time_next + duration_t{lb};
  } else {
    // the unix time at the next stop of the transport segment
    seg.time_prune_ = unix_time_next;
  }
}

template <bool UseLowerBounds>
void query_engine<UseLowerBounds>::seg_prune(
    unixtime_t const worst_time_at_dest,
    std::uint8_t const n,
    transport_segment& seg) {
  bool no_prune = seg.time_prune_ < worst_time_at_dest;
  seg.no_prune_ = no_prune && seg.time_prune_ < state_.t_min_[n];
}

template <bool UseLowerBounds>
void query_engine<UseLowerBounds>::seg_transfers(std::uint8_t const n,
                                                 queue_idx_t const q_cur) {

  auto const seg = state_.q_n_[q_cur];

  // transfer out of current transport segment?
  if (seg.no_prune_) {

    // the day index of the segment
    std::int32_t const d_seg = seg.get_transport_day(base_).v_;

    // iterate stops of the current transport segment
    for (stop_idx_t i = seg.get_stop_idx_start() + 1U;
         i <= seg.get_stop_idx_end(); ++i) {

      // get transfers for this transport/stop
      auto const& transfers =
          state_.ts_.data_.at(seg.get_transport_idx().v_, i);
      // iterate transfers from this stop
      for (auto const& transfer : transfers) {
        // bitset specifying the days on which the transfer is possible
        // from the current transport segment
        auto const& theta = tt_.bitfields_[transfer.get_bitfield_idx()];
        // enqueue if transfer is possible
        if (theta.test(static_cast<std::size_t>(d_seg))) {
          // arrival time at start location of transfer
          auto const tau_arr_t_i =
              tt_.event_mam(seg.get_transport_idx(), i, event_type::kArr)
                  .count();
          // departure time at end location of transfer
          auto const tau_dep_u_j =
              tt_.event_mam(transfer.get_transport_idx_to(),
                            transfer.stop_idx_to_, event_type::kDep)
                  .count();

          auto const d_tr = d_seg + tau_arr_t_i / 1440 - tau_dep_u_j / 1440 +
                            transfer.passes_midnight_;

          bool enq = state_.q_n_.enqueue(
              static_cast<std::uint16_t>(d_tr), transfer.get_transport_idx_to(),
              transfer.get_stop_idx_to(), n + 1U, q_cur);
          if (!enq) {
            ++stats_.n_enqueue_prevented_by_reached_;
          }
        }
      }
    }
  } else {
    ++stats_.n_segments_pruned_;
  }
}

template <bool UseLowerBounds>
void query_engine<UseLowerBounds>::handle_start(query_start const& start) {
  // start day and time
  auto const day_idx_mam = tt_.day_idx_mam(start.time_);
  // start day
  std::int32_t const d = day_idx_mam.first.v_;
  // start time
  std::int32_t const tau = day_idx_mam.second.count();

  // virtual reflexive footpath
  handle_start_footpath(d, tau, footpath{start.location_, duration_t{0U}});
  // iterate outgoing footpaths of source location
  for (auto const fp :
       tt_.locations_.footpaths_out_[profile_idx_t{0U}][start.location_]) {
    handle_start_footpath(d, tau, fp);
  }
}

template <bool UseLowerBounds>
void query_engine<UseLowerBounds>::handle_start_footpath(std::int32_t const d,
                                                         std::int32_t const tau,
                                                         footpath const fp) {
  // arrival time after walking the footpath
  auto const alpha = tau + fp.duration().count();
  auto const alpha_d = alpha / 1440;
  auto const alpha_tod = alpha % 1440;

  // iterate routes at target stop of footpath
  for (auto const route_idx : tt_.location_routes_[fp.target()]) {
    // iterate stop sequence of route, skip last stop
    for (std::uint16_t i = 0U;
         i < tt_.route_location_seq_[route_idx].size() - 1; ++i) {
      auto const route_stop = stop{tt_.route_location_seq_[route_idx][i]};
      if (route_stop.location_idx() == fp.target() && route_stop.in_allowed()) {
        // departure times of this route at this q
        auto const event_times =
            tt_.event_times_at_stop(route_idx, i, event_type::kDep);
        // iterator to departure time of connecting transport at this
        // q
        auto tau_dep_t_i =
            std::lower_bound(event_times.begin(), event_times.end(), alpha_tod,
                             [&](auto&& x, auto&& y) { return x.mam() < y; });
        // shift amount due to walking the footpath
        auto sigma = alpha_d;
        // no departure found on the day of alpha
        if (tau_dep_t_i == event_times.end()) {
          // start looking at the following day
          ++sigma;
          tau_dep_t_i = event_times.begin();
        }
        // iterate departures until maximum waiting time is reached
        while (sigma <= 1) {
          // shift amount due to travel time of transport
          std::int32_t const sigma_t = tau_dep_t_i->days();
          // day index of the transport segment
          auto const d_seg = d + sigma - sigma_t;
          // offset of connecting transport in route_transport_ranges
          auto const k = static_cast<std::size_t>(
              std::distance(event_times.begin(), tau_dep_t_i));
          // transport_idx_t of the connecting transport
          auto const t = tt_.route_transport_ranges_[route_idx][k];
          // bitfield of the connecting transport
          auto const& beta_t = tt_.bitfields_[tt_.transport_traffic_days_[t]];

          if (beta_t.test(static_cast<std::size_t>(d_seg))) {
            // enqueue segment if matching bit is found
            state_.q_n_.enqueue(static_cast<std::uint16_t>(d_seg), t, i, 0U,
                                TRANSFERRED_FROM_NULL);
            break;
          }
          // passing midnight?
          if (tau_dep_t_i + 1 == event_times.end()) {
            ++sigma;
            // start with the earliest transport on the next day
            tau_dep_t_i = event_times.begin();
          } else {
            ++tau_dep_t_i;
          }
        }
      }
    }
  }
}

template struct query_engine<true>;
template struct query_engine<false>;

}  // namespace nigiri::routing::tb