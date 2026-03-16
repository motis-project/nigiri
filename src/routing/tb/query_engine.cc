#include "nigiri/routing/tb/query_engine.h"

#include <ranges>

#include "fmt/ranges.h"

#include "utl/enumerate.h"
#include "utl/raii.h"

#include "nigiri/for_each_meta.h"
#include "nigiri/routing/get_earliest_transport.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/raptor/reconstruct.h"
#include "nigiri/routing/tb/query_engine.h"
#include "nigiri/routing/tb/segment_info.h"
#include "nigiri/routing/tb/settings.h"
#include "nigiri/rt/frun.h"
#include "nigiri/special_stations.h"

// #define tb_debug fmt::println
#define tb_debug(...)

namespace nigiri::routing::tb {

template <bool UseLowerBounds>
query_engine<UseLowerBounds>::query_engine(
    timetable const& tt,
    rt_timetable const*,
    query_state& state,
    bitvec const& is_dest,
    std::array<bitvec, kMaxVias> const&,
    std::vector<std::uint16_t> const& dist_to_dest,
    hash_map<location_idx_t, std::vector<td_offset>> const&,
    std::vector<std::uint16_t> const& lb,
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
  state_.q_n_.base_ = base_;
  stats_.lower_bound_pruning_ = UseLowerBounds;
  state_.reset();

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
      tb_debug("{} is dest!", location{tt_, l});
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
}

template <bool UseLowerBounds>
void query_engine<UseLowerBounds>::execute(unixtime_t const start_time,
                                           std::uint8_t const max_transfers,
                                           unixtime_t const worst_time_at_dest,
                                           profile_idx_t const,
                                           pareto_set<journey>& results) {
  tb_queue_dbg("--- EXECUTE START_TIME={}", start_time);

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
    tb_debug("ROUND start_time={}, k={}", start_time, k);
    for (auto const i : round) {
      seg_dest(k, i);
    }
    for (auto const i : round) {
      seg_prune(k, state_.q_n_[i]);
    }
    for (auto const i : round) {
      seg_transfers(i, k);
    }
    round.from_ = round.to_;
    round.to_ = static_cast<queue_idx_t>(state_.q_n_.size());
    tb_debug("next round: {}", round);
  }

  for (auto n = 0U; n != kMaxTransfers; ++n) {
    if (state_.parent_[n] != queue_entry::kNoParent) {
      results.add({.legs_ = {},
                   .start_time_ = start_time,
                   .dest_time_ = state_.t_min_[n],
                   .dest_ = location_idx_t::invalid(),
                   .transfers_ = static_cast<std::uint8_t>(n)});
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
      tb_debug("[k={}] reached segment {}  => dest not reachable", k,
               seg(segment, qe));
      [[likely]] continue;
    }

    tb_debug("[k={}] reached segment {}  => dest reachable!", k,
             seg(segment, qe));
    auto const t = state_.tbd_.segment_transports_[segment];
    auto const i = static_cast<stop_idx_t>(
        to_idx(segment - state_.tbd_.transport_first_segment_[t] + 1));
    auto const arr_time = tt_.event_time(
        {t, base_ + qe.transport_query_day_offset_}, i, event_type::kArr);
    auto const time_at_dest = arr_time + state_.dist_to_dest_.at(segment);

    for (auto j = k; j != state_.t_min_.size(); ++j) {
      if (state_.t_min_[j] > time_at_dest) {
        state_.t_min_[j] = time_at_dest;
        if (j == k) {
          state_.parent_[j] = q;
        }
      }
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
  if (arr_time > state_.t_min_[k + 1]) {
    tb_debug("PRUNING {}", seg(segment, qe));
    qe.segment_range_.to_ = qe.segment_range_.from_;
    ++stats_.n_segments_pruned_;
  }
}

template <bool UseLowerBounds>
void query_engine<UseLowerBounds>::seg_transfers(queue_idx_t const q,
                                                 std::uint8_t const k) {
  auto const qe = state_.q_n_[q];

  auto const from =
      state_.tbd_.segment_transfers_[qe.segment_range_.from_].begin();
  auto const to = state_.tbd_.segment_transfers_[qe.segment_range_.to_].begin();
  for (auto it = from; it != to; ++it) {
#ifndef _MSC_VER
    if (it + 4 < to) {
      __builtin_prefetch(&*(it + 4));
    }
#endif

    auto const& transfer = *it;

    tb_debug("[k={}] handling queue entry {}: #transfers={}", k, seg(s, qe),
             state_.tbd_.segment_transfers_[s].size());
    auto const day = to_idx(base_ + qe.transport_query_day_offset_);
    if (state_.tbd_.bitfields_[transfer.traffic_days_].test(day)) {
      tb_debug("  -> enqueue transfer to {}", seg(transfer.to_segment_, qe));
      state_.q_n_.enqueue(transfer, q, k + 1, stats_.max_pareto_set_size_);
    } else {
      tb_debug("  transfer {} - {} not active on {}", seg(s, qe),
               seg(transfer.to_segment_, day_idx_t{day}),
               tt_.to_unixtime(day_idx_t{day}));
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
      if (!et.is_valid()) {
        continue;
      }

      auto const query_day_offset = to_idx(et.day_) - to_idx(base_);
      if (query_day_offset < 0 || query_day_offset >= kTBMaxDayOffset) {
        continue;
      }

      auto const transport_first_segment =
          state_.tbd_.transport_first_segment_[et.t_idx_];
      state_.q_n_.initial_enqueue(
          state_.tbd_, transport_first_segment, transport_first_segment + i, r,
          et.t_idx_, static_cast<query_day_offset_t>(query_day_offset), et.day_,
          stats_.max_pareto_set_size_);
    }
  }
}

template <bool UseLowerBounds>
void query_engine<UseLowerBounds>::reconstruct(query const& q,
                                               journey& j) const {
  UTL_FINALLY([&]() { std::reverse(begin(j.legs_), end(j.legs_)); })

  tb_debug("reconstruct journey: transfers={}, dep={} arr={}", j.transfers_,
           j.departure_time(), j.arrival_time());

  auto parent = state_.parent_[j.transfers_];
  auto departure_segment = state_.q_n_[parent].segment_range_.from_;

  auto const has_offset = [&](std::vector<offset> const& offsets,
                              location_match_mode const match_mode,
                              location_idx_t const l) {
    return utl::any_of(offsets, [&](offset const& o) {
      return matches(tt_, match_mode, o.target(), l);
    });
  };

  auto const get_full_segment_range = [&](queue_entry const& qe) {
    auto const transport_segments = state_.tbd_.get_segment_range(
        state_.tbd_.segment_transports_[qe.segment_range_.from_]);
    return interval{qe.segment_range_.from_, transport_segments.to_};
  };

  auto const get_arrival_segment = [&](queue_idx_t const x,
                                       segment_idx_t const to) {
    for (auto const segment : get_full_segment_range(state_.q_n_[x])) {
      for (auto const transfer : state_.tbd_.segment_transfers_[segment]) {
        if (transfer.to_segment_ == to) {
          return segment;
        }
      }
    }
    throw utl::fail("predecessor not found");
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
    tb_debug("run leg:\n\tdeparture segment: {}\n\tarrival segment: {}",
             seg(departure_segment, state_.q_n_[parent]),
             seg(arrival_segment, state_.q_n_[parent]));

    auto const [transport, arr_stop_idx, arr_l, arr_time] =
        get_transport_info(arrival_segment, event_type::kArr);
    auto const [_, dep_stop_idx, dep_l, dep_time] =
        get_transport_info(departure_segment, event_type::kDep);
    return {
        direction::kForward,
        dep_l,
        arr_l,
        dep_time,
        arr_time,
        journey::run_enter_exit{
            rt::run{
                .t_ = transport,
                .stop_range_ = {static_cast<stop_idx_t>(0U),
                                static_cast<stop_idx_t>(
                                    tt_.route_location_seq_
                                        [tt_.transport_route_[transport.t_idx_]]
                                            .size())}},
            dep_stop_idx, arr_stop_idx}};
  };

  // ============
  // (1) Last leg
  // ------------
  auto const find_last_leg = [&]() {
    if (q.dest_match_mode_ == location_match_mode::kIntermodal) {
      for (auto const arr_candidate_segment :
           get_full_segment_range(state_.q_n_[parent])) {
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
        return true;
      }
    } else /* Stop destination -> footpath or direct arrival */ {
      for (auto const arr_candidate_segment :
           get_full_segment_range(state_.q_n_[parent])) {
        tb_debug("dest candidate {}",
                 seg(arr_candidate_segment, state_.q_n_[parent]));

        if (!state_.end_reachable_.test(arr_candidate_segment)) {
          tb_debug("no dest candidate {} => end not reachable",
                   seg(arr_candidate_segment, state_.q_n_[parent]));
          continue;
        }

        auto const [_, _1, arr_l, arr_time] =
            get_transport_info(arr_candidate_segment, event_type::kArr);

        auto const handle_fp = [&](footpath const& fp) {
          if (arr_time + fp.duration() != j.arrival_time() ||
              !has_offset(q.destination_, q.dest_match_mode_, fp.target())) {
            tb_debug(
                "no dest candidate {} arr_l={}: arr_time={} + fp.duration={} = "
                "{} != j.arrival_time={}",
                seg(arr_candidate_segment, state_.q_n_[parent]),
                location{tt_, arr_l}, arr_time, fp.duration(),
                arr_time + fp.duration(), j.arrival_time());
            return false;
          }
          tb_debug("FOUND!");
          j.legs_.push_back({direction::kForward, arr_l, fp.target(), arr_time,
                             j.arrival_time(), fp});
          j.legs_.push_back(get_run_leg(arr_candidate_segment));
          return true;
        };

        if (handle_fp(footpath{arr_l, duration_t{0}})) {
          return true;
        }

        for (auto const fp :
             tt_.locations_.footpaths_out_[state_.tbd_.prf_idx_][arr_l]) {
          if (handle_fp(fp)) {
            return true;
          }
        }
      }
    }
    return false;
  };
  if (!find_last_leg()) {
    throw utl::fail(
        "no last leg found for: transfers={}, start_time={}, dest_time={}",
        j.transfers_, j.departure_time(), j.arrival_time());
  }
  j.dest_ = j.legs_.back().to_;

  // ==================
  // (2) Transport legs
  // ------------------
  parent = state_.q_n_[parent].parent_;
  while (parent != queue_entry::kNoParent) {
    auto const arrival_segment = get_arrival_segment(parent, departure_segment);
    auto const [transport, arr_stop_idx, arr_l, arr_time] =
        get_transport_info(arrival_segment, event_type::kArr);
    auto const fp = get_fp(arr_l, j.legs_.back().from_);
    j.legs_.emplace_back(journey::leg{direction::kForward, arr_l,
                                      j.legs_.back().from_, arr_time,
                                      arr_time + fp.duration(), fp});

    departure_segment = state_.q_n_[parent].segment_range_.from_;
    j.legs_.push_back(get_run_leg(arrival_segment));

    parent = state_.q_n_[parent].parent_;
  }

  // =============
  // (3) First leg
  // -------------
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

template <bool UseLowerBounds>
segment_info query_engine<UseLowerBounds>::seg(segment_idx_t const s,
                                               queue_entry const& qe) const {
  return {tt_, state_.tbd_, s, base_ + qe.transport_query_day_offset_};
}

template <bool UseLowerBounds>
segment_info query_engine<UseLowerBounds>::seg(segment_idx_t const s,
                                               day_idx_t const day) const {
  return {tt_, state_.tbd_, s, day};
}

template struct query_engine<true>;
template struct query_engine<false>;

}  // namespace nigiri::routing::tb