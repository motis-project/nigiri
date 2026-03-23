#include "nigiri/routing/astar/astar_engine.h"

#include <algorithm>
#include <queue>
#include <ranges>

#include "fmt/ranges.h"

#include "utl/enumerate.h"
#include "utl/raii.h"

#include "nigiri/for_each_meta.h"
#include "nigiri/routing/get_earliest_transport.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/raptor/reconstruct.h"
#include "nigiri/routing/tb/segment_info.h"
#include "nigiri/rt/frun.h"
#include "nigiri/special_stations.h"

// #define astar_debug fmt::println
#define astar_debug(...)

namespace nigiri::routing::astar {

template <bool UseLowerBounds>
astar_engine<UseLowerBounds>::astar_engine(
    timetable const& tt,
    rt_timetable const*,
    astar_state& state,
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

  if (dist_to_dest.empty()) {
    is_dest_.for_each_set_bit([&](std::size_t const i) {
      auto const l = location_idx_t{i};
      astar_debug("{} is dest!", loc{tt_, l});
      mark_dest_segments(l, duration_t{0U});
      for (auto const fp :
           tt_.locations_.footpaths_in_[state_.tbd_.prf_idx_][l]) {
        mark_dest_segments(fp.target(), fp.duration());
      }
    });
  } else {
    for (auto const [l_idx, dist] : utl::enumerate(dist_to_dest_)) {
      if (dist != kUnreachable) {
        mark_dest_segments(location_idx_t{l_idx}, duration_t{dist});
      }
    }
  }
}

template <bool UseLowerBounds>
void astar_engine<UseLowerBounds>::execute(unixtime_t const start_time,
                                           std::uint8_t const max_transfers,
                                           unixtime_t const worst_time_at_dest,
                                           profile_idx_t const,
                                           pareto_set<journey>& results) {
  tb_queue_dbg("--- EXECUTE START_TIME={}", start_time);

  auto const up_bnd_min = std::chrono::duration_cast<nigiri::i32_minutes>(
                              worst_time_at_dest - start_time)
                              .count();
  state_.f_min_ = static_cast<double>(up_bnd_min) +
                  static_cast<double>(state_.astar_transfer_penalty_) *
                      static_cast<double>(max_transfers);

  struct open_entry {
    double cost_{0.0};
    tb::queue_idx_t idx_{0U};
    std::uint8_t k_{0U};
  };

  struct open_entry_cmp {
    bool operator()(open_entry const& a, open_entry const& b) const {
      return a.cost_ > b.cost_;
    }
  };

  auto const segment_cost = [&](tb::queue_entry const& qe,
                                std::uint8_t const k) {
    auto const segment = qe.segment_range_.from_;
    auto const t = state_.tbd_.segment_transports_[segment];
    auto const i = static_cast<stop_idx_t>(
        to_idx(segment - state_.tbd_.transport_first_segment_[t] + 1));
    auto const arr_time = tt_.event_time(
        {t, base_ + qe.transport_query_day_offset_}, i, event_type::kArr);
    auto const minutes =
        std::chrono::duration_cast<nigiri::i32_minutes>(arr_time - start_time)
            .count();

    auto const l = stop{tt_.route_location_seq_[tt_.transport_route_[t]][i]}
                       .location_idx();

    if (lb_.empty()) {
      return std::numeric_limits<double>::max();
    }

    auto const lmd = lb_[to_idx(l)];
    auto const cost = static_cast<double>(minutes) +
                      static_cast<double>(state_.astar_transfer_penalty_) *
                          static_cast<double>(k) +
                      lmd;

    if (cost < 0) {
      return std::numeric_limits<double>::max();
    }

    return cost;
  };

  std::priority_queue<open_entry, std::vector<open_entry>, open_entry_cmp> open;
  auto const push_open = [&](tb::queue_idx_t const idx, std::uint8_t const k) {
    open.push({segment_cost(state_.q_n_[idx], k), idx, k});
  };

  for (auto i = tb::queue_idx_t{0U};
       i < static_cast<tb::queue_idx_t>(state_.q_n_.size()); ++i) {
    push_open(i, 0U);
  }

  auto const enqueue_stay = [&](tb::queue_idx_t const parent_idx,
                                std::uint8_t const k) {
    auto const& qe = state_.q_n_[parent_idx];

    auto const next_from = qe.segment_range_.from_ + 1;
    if (next_from >= qe.segment_range_.to_) {
      return;
    }

    auto next = qe;
    next.segment_range_.from_ = next_from;
    next.segment_range_.to_ = qe.segment_range_.to_;
    next.parent_ = parent_idx;

    auto const new_idx = static_cast<tb::queue_idx_t>(state_.q_n_.size());
    state_.q_n_.q_.push_back(next);
    push_open(new_idx, k);
  };

  auto const enqueue_transfers = [&](tb::queue_idx_t const parent_idx,
                                     std::uint8_t const k) {
    if (k >= max_transfers) {
      return;
    }

    auto const size_before = state_.q_n_.size();
    seg_transfers(parent_idx, k);
    auto const size_after = state_.q_n_.size();

    for (auto i = size_before; i < size_after; ++i) {
      push_open(static_cast<tb::queue_idx_t>(i),
                static_cast<std::uint8_t>(k + 1));
    }
  };

  auto max_k = std::uint8_t{0U};

  while (!open.empty()) {
    auto const [cost, idx, k] = open.top();
    open.pop();
    (void)cost;

    if (k > max_transfers) {
      continue;
    }

    auto& qe = state_.q_n_[idx];
    auto const s = qe.segment_range_.from_;

    auto const s_t = state_.tbd_.segment_transports_[s];
    auto const s_i = static_cast<stop_idx_t>(
        to_idx(s - state_.tbd_.transport_first_segment_[s_t] + 1));
    auto const s_arr_time = tt_.event_time(
        {s_t, base_ + qe.transport_query_day_offset_}, s_i, event_type::kArr);
    auto const s_minutes =
        std::chrono::duration_cast<nigiri::i32_minutes>(s_arr_time - start_time)
            .count();
    auto const s_cost = static_cast<double>(s_minutes) +
                        static_cast<double>(state_.astar_transfer_penalty_) *
                            static_cast<double>(k);

    auto const [it, inserted] = state_.best_segment_cost_.emplace(s, s_cost);

    if (!inserted && s_cost >= it->second) {
      stats_.n_enqueue_prevented_by_reached_ += 1;
      continue;
    }

    if (!inserted) {
      it->second = s_cost;
    }

    max_k = std::max(max_k, k);

    seg_dest(start_time, k, idx);

    enqueue_stay(idx, k);
    enqueue_transfers(idx, k);
  }

  if (state_.best_parent_ != tb::queue_entry::kNoParent) {
    results.add({.legs_ = {},
                 .start_time_ = start_time,
                 .dest_time_ = state_.best_dest_time_,
                 .dest_ = location_idx_t::invalid(),
                 .transfers_ = state_.best_k_});
  }

  stats_.n_segments_enqueued_ += state_.q_n_.size();
  stats_.max_transfers_reached_ = max_k >= max_transfers;
}

template <bool UseLowerBounds>
void astar_engine<UseLowerBounds>::seg_dest(unixtime_t const start_time,
                                            std::uint8_t const k,
                                            tb::queue_idx_t const q) {
  auto const& qe = state_.q_n_[q];
  auto const segment = qe.segment_range_.from_;

  if (!state_.end_reachable_.test(segment)) {
    astar_debug("[k={}] reached segment {}  => dest not reachable", k,
                seg(segment, qe));
    return;
  }

  astar_debug("[k={}] reached segment {}  => dest reachable!", k,
              seg(segment, qe));

  auto const t = state_.tbd_.segment_transports_[segment];
  auto const i = static_cast<stop_idx_t>(
      to_idx(segment - state_.tbd_.transport_first_segment_[t] + 1));
  auto const arr_time = tt_.event_time(
      {t, base_ + qe.transport_query_day_offset_}, i, event_type::kArr);
  auto const time_at_dest = arr_time + state_.dist_to_dest_.at(segment);

  auto const minutes =
      std::chrono::duration_cast<nigiri::i32_minutes>(time_at_dest - start_time)
          .count();
  auto const f = static_cast<double>(minutes) +
                 static_cast<double>(state_.astar_transfer_penalty_) *
                     static_cast<double>(k);

  if (f < state_.f_min_) {
    state_.f_min_ = f;
    state_.best_dest_time_ = time_at_dest;
    state_.best_k_ = k;
    state_.best_parent_ = q;
  }
}

template <bool UseLowerBounds>
void astar_engine<UseLowerBounds>::seg_transfers(tb::queue_idx_t const q,
                                                 std::uint8_t const k) {
  auto const qe = state_.q_n_[q];

  auto const tfrs = state_.tbd_.segment_transfers_[qe.segment_range_.from_];

  for (auto const& transfer : tfrs) {
    auto const day = to_idx(base_ + qe.transport_query_day_offset_);
    if (state_.tbd_.bitfields_[transfer.traffic_days_].test(day)) {
      state_.q_n_.enqueue(transfer, q, k + 1, stats_.max_pareto_set_size_);
    }
  }
}

template <bool UseLowerBounds>
void astar_engine<UseLowerBounds>::add_start(location_idx_t const l,
                                             unixtime_t const t) {
  auto const [day, mam] = tt_.day_idx_mam(t);
  for (auto const r : tt_.location_routes_[l]) {
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
      if (query_day_offset < 0 || query_day_offset >= tb::kTBMaxDayOffset) {
        continue;
      }

      auto const transport_first_segment =
          state_.tbd_.transport_first_segment_[et.t_idx_];
      auto const size_before = state_.q_n_.size();
      state_.q_n_.initial_enqueue(
          state_.tbd_, transport_first_segment, transport_first_segment + i, r,
          et.t_idx_, static_cast<tb::query_day_offset_t>(query_day_offset),
          et.day_, stats_.max_pareto_set_size_);
      (void)size_before;
    }
  }
}

template <bool UseLowerBounds>
void astar_engine<UseLowerBounds>::reconstruct(query const& q,
                                               journey& j) const {
  UTL_FINALLY([&]() { std::reverse(begin(j.legs_), end(j.legs_)); })

  astar_debug("reconstruct journey: transfers={}, dep={} arr={}", j.transfers_,
              j.departure_time(), j.arrival_time());

  auto const get_departure_entry = [&](tb::queue_entry const& qe) {
    auto departure = qe;
    auto const dest_seg = qe.segment_range_.from_;

    while (departure.parent_ != tb::queue_entry::kNoParent) {
      auto const departure_candidate = state_.q_n_[departure.parent_];
      auto const candidate_transports =
          state_.tbd_
              .segment_transports_[departure_candidate.segment_range_.from_];
      auto const dest_transports = state_.tbd_.segment_transports_[dest_seg];

      auto const same_transport = candidate_transports == dest_transports;

      if (same_transport) {
        departure = departure_candidate;
        continue;
      }
      break;
    }

    return departure;
  };

  auto const has_offset = [&](std::vector<offset> const& offsets,
                              location_match_mode const match_mode,
                              location_idx_t const l) {
    return utl::any_of(offsets, [&](offset const& o) {
      return matches(tt_, match_mode, o.target(), l);
    });
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
                loc{tt_, from}, loc{tt_, to});
    return *it;
  };

  auto const get_transport_info = [&](tb::queue_entry const& qe,
                                      tb::segment_idx_t const s,
                                      event_type const ev_type)
      -> std::tuple<transport, stop_idx_t, location_idx_t, unixtime_t> {
    auto const d = base_ + qe.transport_query_day_offset_;
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
      [&](tb::queue_entry const& qe, tb::segment_idx_t const departure_segment,
          tb::segment_idx_t const arrival_segment) -> journey::leg {
    astar_debug("run leg:\n\tdeparture segment: {}\n\tarrival segment: {}",
                seg(departure_segment, qe), seg(arrival_segment, qe));

    auto const [transport, arr_stop_idx, arr_l, arr_time] =
        get_transport_info(qe, arrival_segment, event_type::kArr);
    auto const [_, dep_stop_idx, dep_l, dep_time] =
        get_transport_info(qe, departure_segment, event_type::kDep);
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

  auto const find_last_leg = [&]() {
    if (q.dest_match_mode_ == location_match_mode::kIntermodal) {
      auto const arrival = state_.q_n_[state_.best_parent_];
      auto const arrival_segment = arrival.segment_range_.from_;

      if (!state_.end_reachable_.test(arrival_segment)) {
        astar_debug("no dest candidate {} => end not reachable",
                    seg(arrival_segment, arrival));
        return false;
      }

      auto const offset = state_.dist_to_dest_.at(arrival_segment);
      auto const [_, _1, arr_l, arr_time] =
          get_transport_info(arrival, arrival_segment, event_type::kArr);
      if (arr_time + offset != j.arrival_time()) {
        astar_debug("no dest candidate {} => arrival time missmatch",
                    seg(arrival_segment, arrival));
        return false;
      }

      auto const offset_it =
          utl::find_if(q.destination_, [&](routing::offset const& o) {
            return o.target() == arr_l && o.duration() == offset;
          });
      utl::verify(offset_it != end(q.destination_), "offset not found");
      j.legs_.push_back({direction::kForward, arr_l,
                         get_special_station(special_station::kEnd), arr_time,
                         j.arrival_time(), *offset_it});
      return true;
    } else {
      auto const arrival = state_.q_n_[state_.best_parent_];
      auto const arrival_segment = arrival.segment_range_.from_;

      astar_debug("dest candidate {}", seg(arrival_segment, arrival));

      if (!state_.end_reachable_.test(arrival_segment)) {
        astar_debug("no dest candidate {} => end not reachable",
                    seg(arrival_segment, arrival));
        return false;
      }

      auto const [_, _1, arr_l, arr_time] =
          get_transport_info(arrival, arrival_segment, event_type::kArr);

      auto const handle_fp = [&](footpath const& fp) {
        if (arr_time + fp.duration() != j.arrival_time() ||
            !has_offset(q.destination_, q.dest_match_mode_, fp.target())) {
          astar_debug(
              "no dest candidate {} arr_l={}: arr_time={} + fp.duration={} = "
              "{} != j.arrival_time={}",
              seg(arrival_segment, arrival), loc{tt_, arr_l}, arr_time,
              fp.duration(), arr_time + fp.duration(), j.arrival_time());
          return false;
        }
        astar_debug("FOUND!");
        j.legs_.push_back({direction::kForward, arr_l, fp.target(), arr_time,
                           j.arrival_time(), fp});
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
    return false;
  };

  if (!find_last_leg()) {
    throw utl::fail(
        "no last leg found for: transfers={}, start_time={}, dest_time={}",
        j.transfers_, j.departure_time(), j.arrival_time());
  }

  auto current_arrival = state_.best_parent_;
  auto current_arrival_segment =
      state_.q_n_[current_arrival].segment_range_.from_;
  auto current_departure = get_departure_entry(state_.q_n_[current_arrival]);
  auto current_departure_segment = current_departure.segment_range_.from_;

  j.legs_.push_back(get_run_leg(current_departure, current_departure_segment,
                                current_arrival_segment));
  j.dest_ = j.legs_.back().to_;

  current_arrival = current_departure.parent_;

  while (current_arrival != tb::queue_entry::kNoParent) {
    current_departure = get_departure_entry(state_.q_n_[current_arrival]);
    current_departure_segment = current_departure.segment_range_.from_;
    current_arrival_segment = state_.q_n_[current_arrival].segment_range_.from_;

    auto const [transport, arr_stop_idx, arr_l, arr_time] = get_transport_info(
        current_departure, current_arrival_segment, event_type::kArr);
    auto const fp = get_fp(arr_l, j.legs_.back().from_);
    j.legs_.emplace_back(journey::leg{direction::kForward, arr_l,
                                      j.legs_.back().from_, arr_time,
                                      arr_time + fp.duration(), fp});

    j.legs_.push_back(get_run_leg(current_departure, current_departure_segment,
                                  current_arrival_segment));

    current_arrival = current_departure.parent_;
  }

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
        start_time, first_dep_time, loc{tt_, first_dep_l},
        q.start_ | std::views::transform([&](offset const& x) {
          return std::pair{loc{tt_, x.target_}, x.duration()};
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
tb::segment_info astar_engine<UseLowerBounds>::seg(
    tb::segment_idx_t const s, tb::queue_entry const& qe) const {
  return {tt_, state_.tbd_, s, base_ + qe.transport_query_day_offset_};
}

template <bool UseLowerBounds>
tb::segment_info astar_engine<UseLowerBounds>::seg(tb::segment_idx_t const s,
                                                   day_idx_t const day) const {
  return {tt_, state_.tbd_, s, day};
}

template struct astar_engine<true>;
template struct astar_engine<false>;

}  // namespace nigiri::routing::astar