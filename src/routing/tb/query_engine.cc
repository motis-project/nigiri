#include "nigiri/routing/tb/query_engine.h"

#include <ranges>

#include "nigiri/for_each_meta.h"
#include "nigiri/routing/get_earliest_transport.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/raptor/reconstruct.h"
#include "nigiri/routing/tb/query_engine.h"
#include "nigiri/routing/tb/settings.h"
#include "nigiri/rt/frun.h"
#include "nigiri/special_stations.h"

#include "utl/enumerate.h"

namespace nigiri::routing::tb {

template <bool UseLowerBounds>
query_engine<UseLowerBounds>::query_engine(
    timetable const& tt,
    rt_timetable const*,
    query_state& state,
    bitvec& is_dest,
    std::array<bitvec, kMaxVias>&,
    std::vector<std::uint16_t>& dist_to_dest,
    hash_map<location_idx_t, std::vector<td_offset>> const&,
    std::vector<std::uint16_t>& lb,
    std::vector<via_stop> const&,
    day_idx_t const base,
    clasz_mask_t,
    bool,
    bool,
    bool,
    transfer_time_settings)
    : tt_{tt},
      state_{state},
      is_dest_{is_dest},
      dist_to_dest_{dist_to_dest},
      lb_{lb},
      base_{base - QUERY_DAY_SHIFT} {
  stats_.lower_bound_pruning_ = UseLowerBounds;
  state_.reset();

  auto const mark_dest_segments = [&](location_idx_t const l,
                                      duration_t const d) {
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
    for (auto l = location_idx_t{0U}; l != is_dest_.size(); ++l) {
      if (!is_dest_[to_idx(l)]) {
        continue;
      }
      mark_dest_segments(l, duration_t{0U});
      for (auto const fp :
           tt_.locations_.footpaths_in_[state_.tbd_.prf_idx_][l]) {
        mark_dest_segments(fp.target(), fp.duration());
      }
    }
  } else /* Destination is coordinate. */ {
    for (auto const [l_idx, dist] : utl::enumerate(dist_to_dest_)) {
      if (dist != kUnreachable) {
        mark_dest_segments(location_idx_t{l_idx}, duration_t{dist});
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

  auto k = std::uint8_t{0U};
  auto round =  // Queue element indices from previous round.
      interval{queue_idx_t{0U}, static_cast<queue_idx_t>(state_.q_n_.size())};
  for (; k != kMaxTransfers && !round.empty(); ++k) {
    for (auto const i : round) {
      seg_dest(k, i);
    }
    for (auto const i : round) {
      seg_prune(k, state_.q_n_[i]);
    }
    for (auto const i : round) {
      seg_transfers(i);
    }
    round.from_ = round.to_;
    round.to_ = static_cast<queue_idx_t>(state_.q_n_.size());
  }

  for (auto n = 1U; n != kMaxTransfers; ++n) {
    if (state_.parent_[n] != queue_entry::kNoParent) {
      results.add({.legs_ = {},
                   .start_time_ = start_time,
                   .dest_time_ = state_.t_min_[n],
                   .dest_ = location_idx_t::invalid(),
                   .transfers_ = static_cast<std::uint8_t>(n - 1U)});
    }
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
void query_engine<UseLowerBounds>::seg_transfers(queue_idx_t const q) {
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
          [](day_idx_t, std::int16_t) { return false; });

      if (et.is_valid()) {
        state_.q_n_.initial_enqueue(
            state_.tbd_.transport_first_segment_[et.t_idx_] + i, et.t_idx_, i,
            static_cast<std::int8_t>(to_idx(et.day_ - base_)));
      }
    }
  }
}

template <bool UseLowerBounds>
void query_engine<UseLowerBounds>::reconstruct(query const& q,
                                               journey& j) const {
  auto parent = state_.parent_[j.transfers_ + 1U];
  auto departure_segment = state_.q_n_[parent].segment_range_.from_;

  auto const has_offset = [&](std::vector<offset> const& offsets,
                              location_match_mode const match_mode,
                              location_idx_t const l) {
    return utl::any_of(offsets, [&](offset const& o) {
      return matches(tt_, match_mode, o.target(), l);
    });
  };

  auto const get_arrival_segment = [&](queue_idx_t const x,
                                       segment_idx_t const to) {
    for (auto const segment : state_.q_n_[x].segment_range_) {
      for (auto const transfer : state_.tbd_.segment_transfers_[segment]) {
        if (transfer.to_segment_ == to) {
          return segment;
        }
      }
    }
    throw utl::fail("predecessor not found");
  };

  auto const get_fp = [&](location_idx_t const from, location_idx_t const to) {
    auto const from_fps =
        tt_.locations_.footpaths_out_[state_.tbd_.prf_idx_][from];
    auto const it = utl::find_if(
        from_fps, [&](footpath const& fp) { return fp.target() == to; });
    utl::verify(it != end(from_fps),
                "tb reconstruct: footpath from {} to {} not found",
                location{tt_, from}, location{tt_, to});
    return *it;
  };

  auto const get_transport_info = [&](segment_idx_t const s,
                                      event_type const ev_type)
      -> std::tuple<transport, stop_idx_t, location_idx_t, unixtime_t> {
    auto const d = base_ + state_.q_n_[parent].transport_query_day_offset_;
    auto const t = state_.tbd_.segment_transports_[s];
    auto const i = static_cast<stop_idx_t>(
        to_idx(s - state_.tbd_.transport_first_segment_[t] +
               (ev_type == event_type::kArr ? 1 : 0)));
    auto const loc_seq = tt_.route_location_seq_[tt_.transport_route_[t]];
    return {{t, d},
            i,
            stop{loc_seq[i]}.location_idx(),
            tt_.event_time({t, d}, i, ev_type)};
  };

  auto const get_run_leg =
      [&](segment_idx_t const arrival_segment) -> journey::leg {
    auto const [transport, arr_stop_idx, arr_l, arr_time] =
        get_transport_info(arrival_segment, event_type::kArr);
    auto const [_, dep_stop_idx, dep_l, dep_time] =
        get_transport_info(departure_segment, event_type::kDep);
    return {direction::kForward,
            dep_l,
            arr_l,
            dep_time,
            arr_time,
            journey::run_enter_exit{
                rt::run{.t_ = transport, .stop_range_ = {0U, 0U}}, dep_stop_idx,
                arr_stop_idx}};
  };

  // ============
  // (1) Last leg
  // ------------
  if (q.dest_match_mode_ == location_match_mode::kIntermodal) {
    for (auto const arr_candidate_segment :
         state_.q_n_[parent].segment_range_) {
      if (!state_.end_reachable_.test(arr_candidate_segment)) {
        continue;
      }

      auto const offset = state_.dist_to_dest_.at(arr_candidate_segment);
      auto const [_, _1, arr_l, arr_time] =
          get_transport_info(arr_candidate_segment, event_type::kArr);
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
      j.legs_.push_back(get_run_leg(arr_candidate_segment));
      break;
    }
  } else /* Stop destination -> footpath or direct arrival */ {
    for (auto const arr_candidate_segment :
         state_.q_n_[parent].segment_range_) {
      if (!state_.end_reachable_.test(arr_candidate_segment)) {
        continue;
      }

      auto const [_, _1, arr_l, arr_time] =
          get_transport_info(arr_candidate_segment, event_type::kArr);

      auto const handle_fp = [&](footpath const& fp) {
        if (arr_time + fp.duration() != j.arrival_time() ||
            !has_offset(q.destination_, q.dest_match_mode_, fp.target())) {
          return false;
        }
        j.legs_.push_back({direction::kForward, arr_l,
                           get_special_station(special_station::kEnd), arr_time,
                           j.arrival_time(), fp});
        j.legs_.push_back(get_run_leg(arr_candidate_segment));
        return true;
      };

      if (handle_fp(footpath{arr_l, duration_t{0}})) {
        break;
      }

      for (auto const fp :
           tt_.locations_.footpaths_out_[state_.tbd_.prf_idx_][arr_l]) {
        if (handle_fp(fp)) {
          goto stop;  // break out of outer loop
        }
      }

      continue;

    stop:
      break;
    }
  }

  // ==================
  // (2) Transport legs
  // ------------------
  while (parent != queue_entry::kNoParent) {
    auto const arrival_segment = get_arrival_segment(parent, departure_segment);
    auto const [transport, arr_stop_idx, arr_l, arr_time] =
        get_transport_info(arrival_segment, event_type::kArr);
    auto const fp = get_fp(arr_l, j.legs_.back().from_);
    j.legs_.emplace_back(journey::leg{direction::kForward, arr_l,
                                      j.legs_.back().from_, arr_time,
                                      arr_time + fp.duration(), fp});

    departure_segment = state_.q_n_[parent].segment_range_.from_;
    auto const [_, dep_stop_idx, dep_l, dep_time] =
        get_transport_info(departure_segment, event_type::kDep);
    j.legs_.push_back({direction::kForward, dep_l, arr_l, dep_time, arr_time,
                       journey::run_enter_exit{
                           rt::run{.t_ = transport, .stop_range_ = {0U, 0U}},
                           dep_stop_idx, arr_stop_idx}});

    parent = state_.q_n_[parent].parent_;
  }

  // =============
  // (3) First leg
  // -------------
  auto const start_time = j.start_time_;
  auto const first_dep_l = j.legs_.front().from_;
  auto const first_dep_time = j.legs_.front().dep_time_;
  if (q.start_match_mode_ == location_match_mode::kIntermodal) {
    auto const offset_it =
        utl::find_if(q.start_, [&](routing::offset const& o) {
          return o.target() == first_dep_l &&
                 start_time + o.duration() <= first_dep_time;
        });
    utl::verify(offset_it != end(q.start_), "no start offset found");
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

  std::reverse(begin(j.legs_), begin(j.legs_));
}

template struct query_engine<true>;
template struct query_engine<false>;

}  // namespace nigiri::routing::tb