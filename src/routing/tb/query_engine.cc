#include "nigiri/routing/tb/query_engine.h"

#include <ranges>

#include "nigiri/for_each_meta.h"
#include "nigiri/routing/get_earliest_transport.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/tb/query_engine.h"
#include "nigiri/routing/tb/settings.h"
#include "nigiri/rt/frun.h"
#include "nigiri/special_stations.h"

#include "utl/enumerate.h"

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
      base_{base - QUERY_DAY_SHIFT} {
  stats_.lower_bound_pruning_ = UseLowerBounds;

  // reset state for new query
  state_.reset();

  auto const mark_segments = [&](location_idx_t const l, duration_t const d) {
    for (auto const r : tt_.location_routes_[l]) {
      for (auto i = stop_idx_t{1U}; i != tt_.route_location_seq_[r].size();
           ++i) {
        for (auto const t : tt_.route_transport_ranges_[r]) {
          auto const segment = state_.tbd_.transport_first_segment_[t] + i - 1;
          state_.end_reachable_.set(segment, true);
          state_.dist_to_dest_.emplace(segment, d);
        }
      }
    }
  };

  if (dist_to_dest.empty()) /* Destination is stop. */ {
    for (auto l = location_idx_t{0U}; l != location_idx_t{is_dest_.size()};
         ++l) {
      if (!is_dest_[to_idx(l)]) {
        continue;
      }
      mark_segments(l, duration_t{0U});
      for (auto const fp :
           tt_.locations_.footpaths_in_[state_.tbd_.prf_idx_][l]) {
        mark_segments(fp.target(), fp.duration());
      }
    }
  } else /* Destination is coordinate. */ {
    for (auto const [l_idx, dist] : utl::enumerate(dist_to_dest_)) {
      if (dist != kUnreachable) {
        mark_segments(location_idx_t{l_idx}, duration_t{dist});
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
  for (auto k = 0U; k != kMaxTransfers; ++k) {
    if (state_.t_min_[k] >= worst_time_at_dest) {
      state_.t_min_[k] = worst_time_at_dest;
      state_.parent_[k] = queue_entry::kNoParent;
    }
  }

  // process all Q_n in ascending order, i.e., transport segments reached after
  // n transfers
  auto k = std::uint8_t{0U};
  auto round =
      interval{queue_idx_t{0U}, static_cast<queue_idx_t>(state_.q_n_.size())};
  for (; k != kMaxTransfers && !round.empty(); ++k) {
    // (1)  destination reached?
    for (auto const i : round) {
      seg_dest(k, i);
    }

    // (2) pruning?
    for (auto const i : round) {
      seg_prune(k, state_.q_n_[i]);
    }

    // (3) process transfers & enqueue segments
    for (auto const i : round) {
      seg_transfers(k, i);
    }

    round.from_ = round.to_;
    round.to_ = static_cast<queue_idx_t>(state_.q_n_.size());
  }

  stats_.n_segments_enqueued_ += state_.q_n_.size();
  stats_.n_rounds_ = k - 1U;
  stats_.max_transfers_reached_ = k == max_transfers;
}

template <bool UseLowerBounds>
void query_engine<UseLowerBounds>::seg_dest(std::uint8_t const k,
                                            queue_idx_t const q) {
  auto const& qe = state_.q_n_[q];
  for (auto const segment : qe.segment_range_) {
    if (!state_.end_reachable_.test(segment)) {
      [[likely]] continue;
    }

    auto const t = state_.tbd_.segment_transports_[segment];
    auto const i = static_cast<stop_idx_t>(
        to_idx(segment - state_.tbd_.transport_first_segment_[t] + 1));
    auto const arr_time = tt_.event_time(
        {t, base_ + qe.transport_query_day_offset_}, i, event_type::kArr);
    auto const time_at_dest = arr_time + state_.dist_to_dest_.at(segment);
    if (state_.t_min_[k] > time_at_dest) {
      state_.t_min_[k] = time_at_dest;
      state_.parent_[k] = q;
    }
  }
}

template <bool UseLowerBounds>
void query_engine<UseLowerBounds>::seg_prune(std::uint8_t const k,
                                             queue_entry& qe) {
  auto const segment = qe.segment_range_[0];
  auto const t = state_.tbd_.segment_transports_[segment];
  auto const i = static_cast<stop_idx_t>(
      to_idx(segment - state_.tbd_.transport_first_segment_[t] + 1));
  auto arr_time = tt_.event_time({t, base_ + qe.transport_query_day_offset_}, i,
                                 event_type::kArr);
  if constexpr (UseLowerBounds) {
    auto const l = stop{tt_.route_location_seq_[tt_.transport_route_[t]][i]}
                       .location_idx();
    arr_time += duration_t{lb_[to_idx(l)]};
  }
  if (arr_time > state_.t_min_[k]) {
    qe.segment_range_.to_ = qe.segment_range_.from_;
    ++stats_.n_segments_pruned_;
  }
}

template <bool UseLowerBounds>
void query_engine<UseLowerBounds>::seg_transfers(std::uint8_t const n,
                                                 queue_idx_t const q) {
  auto const qe = state_.q_n_[q];
  for (auto const s : qe.segment_range_) {
    for (auto const transfer : state_.tbd_.segment_transfers_[s]) {
      auto const day = to_idx(base_ + qe.transport_query_day_offset_);
      if (state_.tbd_.bitfields_[transfer.traffic_days_].test(day)) {
        state_.q_n_.enqueue(transfer, q);
      }
    }
  }
}

template <bool UseLowerBounds>
void query_engine<UseLowerBounds>::add_start(location_idx_t const l,
                                             unixtime_t const t) {
  auto const [day, mam] = tt_.day_idx_mam(t);
  auto const fp_arr_mam = mam.count();

  for (auto const r : tt_.location_routes_[l]) {
    // iterate stop sequence of route, skip last stop
    auto const stop_seq = tt_.route_location_seq_[r];
    for (auto i = stop_idx_t{0U}; i < stop_seq.size() - 1; ++i) {
      auto const stp = stop{stop_seq[i]};
      if (!stp.in_allowed() || stp.location_idx() != l) {
        continue;
      }

      auto const et = get_earliest_transport<direction::kForward>(
          tt_, tt_, 0U, r, i, day, mam, stp.location_idx(),
          [](day_idx_t, minutes_after_midnight_t) { return false; });

      if (et.is_valid()) {
        state_.q_n_.initial_enqueue(
            state_.tbd_.transport_first_segment_[et.t_idx_] + i, et.t_idx_, i,
            static_cast<std::int8_t>(to_idx(et.day_ - base_)));
      }
    }
  }
}

template struct query_engine<true>;
template struct query_engine<false>;

}  // namespace nigiri::routing::tb