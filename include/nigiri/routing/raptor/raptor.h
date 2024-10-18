#pragma once

#include "nigiri/common/delta_t.h"
#include "nigiri/common/linear_lower_bound.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/raptor/debug.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/routing/raptor/reconstruct.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing {

struct raptor_stats {
  std::uint64_t n_routing_time_{0ULL};
  std::uint64_t n_footpaths_visited_{0ULL};
  std::uint64_t n_routes_visited_{0ULL};
  std::uint64_t n_earliest_trip_calls_{0ULL};
  std::uint64_t n_earliest_arrival_updated_by_route_{0ULL};
  std::uint64_t n_earliest_arrival_updated_by_footpath_{0ULL};
  std::uint64_t fp_update_prevented_by_lower_bound_{0ULL};
  std::uint64_t route_update_prevented_by_lower_bound_{0ULL};
};

template <direction SearchDir, bool Rt>
struct raptor {
  using algo_state_t = raptor_state;
  using algo_stats_t = raptor_stats;

  static constexpr bool kUseLowerBounds = true;
  static constexpr auto const kFwd = (SearchDir == direction::kForward);
  static constexpr auto const kBwd = (SearchDir == direction::kBackward);
  static constexpr auto const kInvalid = kInvalidDelta<SearchDir>;
  static constexpr auto const kUnreachable =
      std::numeric_limits<std::uint16_t>::max();
  static constexpr auto const kIntermodalTarget =
      to_idx(get_special_station(special_station::kEnd));

  static bool is_better(auto a, auto b) { return kFwd ? a < b : a > b; }
  static bool is_better_or_eq(auto a, auto b) { return kFwd ? a <= b : a >= b; }
  static auto get_best(auto a, auto b) { return is_better(a, b) ? a : b; }
  static auto get_best(auto x, auto... y) {
    ((x = get_best(x, y)), ...);
    return x;
  }
  static auto dir(auto a) { return (kFwd ? 1 : -1) * a; }

  raptor(timetable const& tt,
         rt_timetable const* rtt,
         raptor_state& state,
         std::vector<bool>& is_dest,
         std::vector<std::uint16_t>& dist_to_dest,
         std::vector<std::uint16_t>& lb,
         day_idx_t const base,
         clasz_mask_t const allowed_claszes)
      : tt_{tt},
        rtt_{rtt},
        state_{state},
        is_dest_{is_dest},
        dist_to_end_{dist_to_dest},
        lb_{lb},
        base_{base},
        n_days_{tt_.internal_interval_days().size().count()},
        n_locations_{tt_.n_locations()},
        n_routes_{tt.n_routes()},
        n_rt_transports_{Rt ? rtt->n_rt_transports() : 0U},
        allowed_claszes_{allowed_claszes} {
    state_.resize(n_locations_, n_routes_, n_rt_transports_);
    utl::fill(time_at_dest_, kInvalid);
    state_.round_times_.reset(kInvalid);
  }

  algo_stats_t get_stats() const { return stats_; }

  void reset_arrivals() {
    utl::fill(time_at_dest_, kInvalid);
    state_.round_times_.reset(kInvalid);
  }

  void next_start_time() {
    utl::fill(state_.best_, kInvalid);
    utl::fill(state_.tmp_, kInvalid);
    utl::fill(state_.prev_station_mark_, false);
    utl::fill(state_.station_mark_, false);
    utl::fill(state_.route_mark_, false);
    if constexpr (Rt) {
      utl::fill(state_.rt_transport_mark_, false);
    }
  }

  void add_start(location_idx_t const l, unixtime_t const t) {
    trace_upd("adding start {}: {}\n", location{tt_, l}, t);
    state_.best_[to_idx(l)] = unix_to_delta(base(), t);
    state_.round_times_[0U][to_idx(l)] = unix_to_delta(base(), t);
    state_.station_mark_[to_idx(l)] = true;
  }

  void execute(unixtime_t const start_time,
               std::uint8_t const max_transfers,
               unixtime_t const worst_time_at_dest,
               profile_idx_t const prf_idx,
               pareto_set<journey>& results) {
    auto const end_k = std::min(max_transfers, kMaxTransfers) + 1U;

    auto const d_worst_at_dest = unix_to_delta(base(), worst_time_at_dest);
    for (auto& time_at_dest : time_at_dest_) {
      time_at_dest = get_best(d_worst_at_dest, time_at_dest);
    }

    trace_print_init_state();

    for (auto k = 1U; k != end_k; ++k) {

      for (auto i = 0U; i != n_locations_; ++i) {
        state_.best_[i] = get_best(state_.round_times_[k][i], state_.best_[i]);
        if (is_dest_[i]) {
          update_time_at_dest(k, state_.best_[i]);
        }
      }

      auto any_marked = false;
      for (auto i = 0U; i != n_locations_; ++i) {
        if (state_.station_mark_[i]) {
          for (auto const& r : tt_.location_routes_[location_idx_t{i}]) {
            any_marked = true;
            state_.route_mark_[to_idx(r)] = true;
          }
          if constexpr (Rt) {
            for (auto const& rt_t :
                 rtt_->location_rt_transports_[location_idx_t{i}]) {
              any_marked = true;
              state_.rt_transport_mark_[to_idx(rt_t)] = true;
            }
          }
        }
      }

      if (!any_marked) {
        trace_print_state_after_round();
        break;
      }

      std::swap(state_.prev_station_mark_, state_.station_mark_);
      utl::fill(state_.station_mark_, false);

      any_marked = (allowed_claszes_ == all_clasz_allowed())
                       ? loop_routes<false>(k)
                       : loop_routes<true>(k);
      if constexpr (Rt) {
        any_marked |= (allowed_claszes_ == all_clasz_allowed())
                          ? loop_rt_routes<false>(k)
                          : loop_rt_routes<true>(k);
      }

      if (!any_marked) {
        trace_print_state_after_round();
        break;
      }

      utl::fill(state_.route_mark_, false);

      std::swap(state_.prev_station_mark_, state_.station_mark_);
      utl::fill(state_.station_mark_, false);

      update_transfers(k);

      update_footpaths(k, prf_idx);

      update_intermodal_footpaths(k);

      trace_print_state_after_round();
    }

    for (auto i = 0U; i != n_locations_; ++i) {
      auto const is_dest = is_dest_[i];
      if (!is_dest) {
        continue;
      }

      for (auto k = 1U; k != end_k; ++k) {
        auto const dest_time = state_.round_times_[k][i];
        if (dest_time != kInvalid) {
          trace("ADDING JOURNEY: start={}, dest={} @ {}, transfers={}\n",
                start_time, delta_to_unix(base(), state_.round_times_[k][i]),
                location{tt_, location_idx_t{i}}, k - 1);
          auto const [optimal, it, dominated_by] = results.add(
              journey{.legs_ = {},
                      .start_time_ = start_time,
                      .dest_time_ = delta_to_unix(base(), dest_time),
                      .dest_ = location_idx_t{i},
                      .transfers_ = static_cast<std::uint8_t>(k - 1)});
          if (!optimal) {
            trace("  DOMINATED BY: start={}, dest={} @ {}, transfers={}\n",
                  dominated_by->start_time_, dominated_by->dest_time_,
                  location{tt_, dominated_by->dest_}, dominated_by->transfers_);
          }
        }
      }
    }
  }

  void reconstruct(query const& q, journey& j) {
    reconstruct_journey<SearchDir>(tt_, rtt_, q, state_, j, base(), base_);
  }

private:
  date::sys_days base() const {
    return tt_.internal_interval_days().from_ + as_int(base_) * date::days{1};
  }

  template <bool WithClaszFilter>
  bool loop_routes(unsigned const k) {
    auto any_marked = false;
    for (auto r_idx = 0U; r_idx != n_routes_; ++r_idx) {
      auto const r = route_idx_t{r_idx};
      if (state_.route_mark_[r_idx]) {
        if constexpr (WithClaszFilter) {

          if (!is_allowed(allowed_claszes_, tt_.route_clasz_[r])) {
            continue;
          }
        }
        ++stats_.n_routes_visited_;
        trace("┊ ├k={} updating route {}\n", k, r);

        any_marked |= update_route(k, r);
      }
    }
    return any_marked;
  }

  template <bool WithClaszFilter>
  bool loop_rt_routes(unsigned const k) {
    auto any_marked = false;
    for (auto rt_t_idx = 0U; rt_t_idx != n_rt_transports_; ++rt_t_idx) {
      if (state_.rt_transport_mark_[rt_t_idx]) {
        auto const rt_t = rt_transport_idx_t{rt_t_idx};

        if constexpr (WithClaszFilter) {
          if (!is_allowed(allowed_claszes_,
                          rtt_->rt_transport_section_clasz_[rt_t][0])) {
            continue;
          }
        }

        ++stats_.n_routes_visited_;
        trace("┊ ├k={} updating rt transport {}\n", k, rt_t);
        any_marked |= update_rt_transport(k, rt_t);
      }
    }
    return any_marked;
  }

  void update_transfers(unsigned const k) {
    for (auto i = 0U; i != n_locations_; ++i) {
      if (!state_.prev_station_mark_[i]) {
        continue;
      }

      auto const is_dest = is_dest_[i];
      auto const transfer_time =
          (!is_intermodal_dest() && is_dest)
              ? 0
              : dir(tt_.locations_.transfer_time_[location_idx_t{i}]).count();
      auto const fp_target_time =
          static_cast<delta_t>(state_.tmp_[i] + transfer_time);
      if (is_better(fp_target_time, state_.best_[i]) &&
          is_better(fp_target_time, time_at_dest_[k])) {
        if (lb_[i] == kUnreachable ||
            !is_better(fp_target_time + dir(lb_[i]), time_at_dest_[k])) {
          ++stats_.fp_update_prevented_by_lower_bound_;
          continue;
        }

        ++stats_.n_earliest_arrival_updated_by_footpath_;
        state_.round_times_[k][i] = fp_target_time;
        state_.best_[i] = fp_target_time;
        state_.station_mark_[i] = true;
        if (is_dest) {
          update_time_at_dest(k, fp_target_time);
        }
      }
    }
  }

  void update_footpaths(unsigned const k, profile_idx_t const prf_idx) {
    for (auto i = 0U; i != n_locations_; ++i) {
      if (!state_.prev_station_mark_[i]) {
        continue;
      }

      auto const l_idx = location_idx_t{i};
      auto const& fps = kFwd ? tt_.locations_.footpaths_out_[prf_idx][l_idx]
                             : tt_.locations_.footpaths_in_[prf_idx][l_idx];
      for (auto const& fp : fps) {
        ++stats_.n_footpaths_visited_;

        auto const target = to_idx(fp.target());
        auto const fp_target_time =
            clamp(state_.tmp_[i] + dir(fp.duration()).count());

        if (is_better(fp_target_time, state_.best_[target]) &&
            is_better(fp_target_time, time_at_dest_[k])) {
          auto const lower_bound = lb_[to_idx(fp.target())];
          if (lower_bound == kUnreachable ||
              !is_better(fp_target_time + dir(lower_bound), time_at_dest_[k])) {
            ++stats_.fp_update_prevented_by_lower_bound_;
            trace_upd(
                "┊ ├k={} *** LB NO UPD: (from={}, tmp={}) --{}--> (to={}, "
                "best={}) --> update => {}, LB={}, LB_AT_DEST={}, DEST={}\n",
                k, location{tt_, l_idx}, to_unix(state_.tmp_[to_idx(l_idx)]),
                fp.duration(), location{tt_, fp.target()},
                state_.best_[to_idx(fp.target())], fp_target_time, lower_bound,
                to_unix(clamp(fp_target_time + dir(lower_bound))),
                to_unix(time_at_dest_[k]));
            continue;
          }

          trace_upd(
              "┊ ├k={}   footpath: ({}, tmp={}) --{}--> ({}, best={}) --> "
              "update => {}\n",
              k, location{tt_, l_idx}, to_unix(state_.tmp_[to_idx(l_idx)]),
              fp.duration(), location{tt_, fp.target()},
              to_unix(state_.best_[to_idx(fp.target())]), fp_target_time);
          ++stats_.n_earliest_arrival_updated_by_footpath_;
          state_.round_times_[k][to_idx(fp.target())] = fp_target_time;
          state_.best_[to_idx(fp.target())] = fp_target_time;
          state_.station_mark_[to_idx(fp.target())] = true;
          if (is_dest_[to_idx(fp.target())]) {
            update_time_at_dest(k, fp_target_time);
          }
        } else {
          trace(
              "┊ ├k={}   NO FP UPDATE: {} [best={}] --{}--> {} "
              "[best={}, time_at_dest={}]\n",
              k, location{tt_, l_idx}, state_.best_[to_idx(l_idx)],
              fp.duration(), location{tt_, fp.target()},
              state_.best_[to_idx(fp.target())], to_unix(time_at_dest_[k]));
        }
      }
    }
  }

  void update_intermodal_footpaths(unsigned const k) {
    if (dist_to_end_.empty()) {
      return;
    }

    for (auto i = 0U; i != n_locations_; ++i) {
      if ((state_.prev_station_mark_[i] || state_.station_mark_[i]) &&
          dist_to_end_[i] != kUnreachable) {
        auto const end_time = clamp(get_best(state_.best_[i], state_.tmp_[i]) +
                                    dir(dist_to_end_[i]));

        if (is_better(end_time, state_.best_[kIntermodalTarget])) {
          state_.round_times_[k][kIntermodalTarget] = end_time;
          state_.best_[kIntermodalTarget] = end_time;
          update_time_at_dest(k, end_time);
        }

        trace("┊ │k={}  INTERMODAL FOOTPATH: location={}, dist_to_end={}\n", k,
              location{tt_, location_idx_t{i}}, dist_to_end_[i]);
      }
    }
  }

  bool update_rt_transport(unsigned const k, rt_transport_idx_t const rt_t) {
    auto const stop_seq = rtt_->rt_transport_location_seq_[rt_t];
    auto et = false;
    auto any_marked = false;
    for (auto i = 0U; i != stop_seq.size(); ++i) {
      auto const stop_idx =
          static_cast<stop_idx_t>(kFwd ? i : stop_seq.size() - i - 1U);
      auto const stp = stop{stop_seq[stop_idx]};
      auto const l_idx = cista::to_idx(stp.location_idx());
      auto const is_last = i == stop_seq.size() - 1U;

      if ((kFwd && stop_idx != 0U) ||
          (kBwd && stop_idx != stop_seq.size() - 1U)) {
        auto current_best = kInvalid;
        auto const by_transport = rt_time_at_stop(
            rt_t, stop_idx, kFwd ? event_type::kArr : event_type::kDep);
        if (et && (kFwd ? stp.out_allowed() : stp.in_allowed())) {
          current_best = get_best(state_.round_times_[k - 1][l_idx],
                                  state_.tmp_[l_idx], state_.best_[l_idx]);
          if (is_better(by_transport, current_best) &&
              is_better(by_transport, time_at_dest_[k]) &&
              lb_[l_idx] != kUnreachable &&
              is_better(by_transport + dir(lb_[l_idx]), time_at_dest_[k])) {
            trace_upd(
                "┊ │k={}    RT | name={}, dbg={}, time_by_transport={}, BETTER "
                "THAN current_best={} => update, {} marking station {}!\n",
                k, rtt_->transport_name(tt_, rt_t), rtt_->dbg(tt_, rt_t),
                by_transport, current_best,
                !is_better(by_transport, current_best) ? "NOT" : "",
                location{tt_, stp.location_idx()});

            ++stats_.n_earliest_arrival_updated_by_route_;
            state_.tmp_[l_idx] = get_best(by_transport, state_.tmp_[l_idx]);
            state_.station_mark_[l_idx] = true;
            current_best = by_transport;
            any_marked = true;
          }
        }
      }

      if (lb_[l_idx] == kUnreachable) {
        break;
      }

      if (is_last || !(kFwd ? stp.in_allowed() : stp.out_allowed()) ||
          !state_.prev_station_mark_[l_idx]) {
        continue;
      }

      auto const by_transport = rt_time_at_stop(
          rt_t, stop_idx, kFwd ? event_type::kDep : event_type::kArr);
      auto const prev_round_time = state_.round_times_[k - 1][l_idx];
      if (is_better_or_eq(prev_round_time, by_transport)) {
        et = true;
      }
    }
    return any_marked;
  }

  bool update_route(unsigned const k, route_idx_t const r) {
    auto const stop_seq = tt_.route_location_seq_[r];
    bool any_marked = false;
    auto et = transport{};
    for (auto i = 0U; i != stop_seq.size(); ++i) {
      auto const stop_idx =
          static_cast<stop_idx_t>(kFwd ? i : stop_seq.size() - i - 1U);
      auto const stp = stop{stop_seq[stop_idx]};
      auto const l_idx = cista::to_idx(stp.location_idx());
      auto const is_last = i == stop_seq.size() - 1U;

      if (!et.is_valid() && !state_.prev_station_mark_[l_idx]) {
        trace("┊ │k={}  stop_idx={} {}: not marked, no et - skip\n", k,
              stop_idx, location{tt_, location_idx_t{l_idx}});
        continue;
      }

      trace(
          "┊ │k={}  stop_idx={}, location={}, round_times={}, best={}, "
          "tmp={}\n",
          k, stop_idx, location{tt_, stp.location_idx()},
          to_unix(state_.round_times_[k - 1][l_idx]),
          to_unix(state_.best_[l_idx]), to_unix(state_.tmp_[l_idx]));


      auto current_best = kInvalid;
      if (et.is_valid() && (kFwd ? stp.out_allowed() : stp.in_allowed())) {
        auto const by_transport = time_at_stop(
            r, et, stop_idx, kFwd ? event_type::kArr : event_type::kDep);
        current_best = get_best(state_.round_times_[k - 1][l_idx],
                                state_.tmp_[l_idx], state_.best_[l_idx]);
        assert(by_transport != std::numeric_limits<delta_t>::min() &&
               by_transport != std::numeric_limits<delta_t>::max());

        if (is_better(by_transport, current_best) &&
            is_better(by_transport, time_at_dest_[k]) &&
            lb_[l_idx] != kUnreachable &&
            is_better(by_transport + dir(lb_[l_idx]), time_at_dest_[k])) {
          trace_upd(
              "┊ │k={}    name={}, dbg={}, time_by_transport={}, BETTER THAN "
              "current_best={} => update, {} marking station {}!\n",
              k, tt_.transport_name(et.t_idx_), tt_.dbg(et.t_idx_),
              by_transport, current_best,
              !is_better(by_transport, current_best) ? "NOT" : "",
              location{tt_, stp.location_idx()});
          ++stats_.n_earliest_arrival_updated_by_route_;
          state_.tmp_[l_idx] = get_best(by_transport, state_.tmp_[l_idx]);
          state_.station_mark_[l_idx] = true;
          current_best = by_transport;
          any_marked = true;
        } else {
          trace(
              "┊ │k={}    *** NO UPD: at={}, name={}, dbg={}, "
              "time_by_transport={}, current_best=min({}, {}, {})={} => {} - "
              "LB={}, LB_AT_DEST={}, TIME_AT_DEST={} "
              "(is_better(by_transport={}={}, current_best={}={})={}, "
              "is_better(by_transport={}={}, time_at_dest_={}={})={}, "
              "reachable={}, "
              "is_better(lb={}={}, time_at_dest_={}={})={})!\n",
              k, location{tt_, location_idx_t{l_idx}},
              tt_.transport_name(et.t_idx_), tt_.dbg(et.t_idx_),
              to_unix(by_transport), to_unix(state_.round_times_[k - 1][l_idx]),
              to_unix(state_.best_[l_idx]), to_unix(state_.tmp_[l_idx]),
              to_unix(current_best), location{tt_, location_idx_t{l_idx}},
              lb_[l_idx], to_unix(time_at_dest_[k]),
              to_unix(clamp(by_transport + dir(lb_[l_idx]))), by_transport,
              to_unix(by_transport), current_best, to_unix(current_best),
              is_better(by_transport, current_best), by_transport,
              to_unix(by_transport), time_at_dest_[k],
              to_unix(time_at_dest_[k]),
              is_better(by_transport, time_at_dest_[k]),
              lb_[l_idx] != kUnreachable, by_transport + dir(lb_[l_idx]),
              to_unix(clamp(by_transport + dir(lb_[l_idx]))), time_at_dest_[k],
              to_unix(time_at_dest_[k]), to_unix(time_at_dest_[k]),
              is_better(clamp(by_transport + dir(lb_[l_idx])),
                        time_at_dest_[k]));
        }
      } else {
        trace(
            "┊ │k={}    *** NO UPD: no_trip={}, in_allowed={}, "
            "out_allowed={}, label_allowed={}\n",
            k, !et.is_valid(), stp.in_allowed(), stp.out_allowed(),
            (kFwd ? stp.out_allowed() : stp.in_allowed()));
      }
      if (is_last || !(kFwd ? stp.in_allowed() : stp.out_allowed()) ||
          !state_.prev_station_mark_[l_idx]) {
        continue;
      }

      if (lb_[l_idx] == kUnreachable) {
        break;
      }

      auto const et_time_at_stop =
          et.is_valid()
              ? time_at_stop(r, et, stop_idx,
                             kFwd ? event_type::kDep : event_type::kArr)
              : kInvalid;
      auto const prev_round_time = state_.round_times_[k - 1][l_idx];
      assert(prev_round_time != kInvalid);
      if (is_better_or_eq(prev_round_time, et_time_at_stop)) {
        auto const [day, mam] = split(prev_round_time);
        auto const new_et = get_earliest_transport(k, r, stop_idx, day, mam,
                                                   stp.location_idx());
        current_best =
            get_best(current_best, state_.best_[l_idx], state_.tmp_[l_idx]);

        if (new_et.is_valid() &&
            (current_best == kInvalid ||
             is_better_or_eq(
                 time_at_stop(r, new_et, stop_idx,
                              kFwd ? event_type::kDep : event_type::kArr),
                 et_time_at_stop))) {
          et = new_et;
        } else if (new_et.is_valid()) {
          trace("┊ │k={}    update et: no update time_at_stop={}\n", k,
                et_time_at_stop);
        }
      }
    }
    return any_marked;
  }

  transport get_earliest_transport(unsigned const k,
                                   route_idx_t const r,
                                   stop_idx_t const stop_idx,
                                   day_idx_t const day_at_stop,
                                   minutes_after_midnight_t const mam_at_stop,
                                   location_idx_t const l) {
    ++stats_.n_earliest_trip_calls_;

    auto const n_days_to_iterate = std::min(
        kMaxTravelTime.count() / 1440 + 1,
        kFwd ? n_days_ - as_int(day_at_stop) : as_int(day_at_stop) + 1);

    auto const event_times = tt_.event_times_at_stop(
        r, stop_idx, kFwd ? event_type::kDep : event_type::kArr);

    auto const seek_first_day = [&]() {
      return linear_lb(get_begin_it(event_times), get_end_it(event_times),
                       mam_at_stop,
                       [&](delta const a, minutes_after_midnight_t const b) {
                         return is_better(a.mam(), b.count());
                       });
    };
#if defined(NIGIRI_TRACING)
    auto const l_idx =
        stop{tt_.route_location_seq_[r][stop_idx]}.location_idx();

    trace(
        "┊ │k={}    et: current_best_at_stop={}, stop_idx={}, location={}, "
        "n_days_to_iterate={}\n",
        k, tt_.to_unixtime(day_at_stop, mam_at_stop), stop_idx,
        location{tt_, l_idx}, n_days_to_iterate);
#endif

    for (auto i = day_idx_t::value_t{0U}; i != n_days_to_iterate; ++i) {
      auto const ev_time_range =
          it_range{i == 0U ? seek_first_day() : get_begin_it(event_times),
                   get_end_it(event_times)};

      if (ev_time_range.empty()) {
        continue;
      }
      auto const day = kFwd ? day_at_stop + i : day_at_stop - i;
      for (auto it = begin(ev_time_range); it != end(ev_time_range); ++it) {
        auto const t_offset =
            static_cast<std::size_t>(&*it - event_times.data());
        auto const ev = *it;
        auto const ev_mam = ev.mam();

        if (is_better_or_eq(time_at_dest_[k],
                            to_delta(day, ev_mam) + dir(lb_[to_idx(l)]))) {
          trace(
              "┊ │k={}      => name={}, dbg={}, day={}={}, best_mam={}, "
              "transport_mam={}, transport_time={} => TIME AT DEST {} IS "
              "BETTER!\n",
              k, tt_.transport_name(tt_.route_transport_ranges_[r][t_offset]),
              tt_.dbg(tt_.route_transport_ranges_[r][t_offset]), day,
              tt_.to_unixtime(day, 0_minutes), mam_at_stop, ev_mam,
              tt_.to_unixtime(day, duration_t{ev_mam}),
              to_unix(time_at_dest_[k]));
          return {transport_idx_t::invalid(), day_idx_t::invalid()};
        }
        auto const t = tt_.route_transport_ranges_[r][t_offset];

        if (i == 0U && !is_better_or_eq(mam_at_stop.count(), ev_mam)) {
          trace(
              "┊ │k={}      => transport={}, name={}, dbg={}, day={}/{}, "
              "best_mam={}, "
              "transport_mam={}, transport_time={} => NO REACH!\n",
              k, t, tt_.transport_name(t), tt_.dbg(t), i, day, mam_at_stop,
              ev_mam, ev);
          continue;
        }

        auto const ev_day_offset = ev.days();
        auto const start_day =
            static_cast<std::size_t>(as_int(day) - ev_day_offset);
        if (!is_transport_active(t, start_day)) {
          trace(
              "┊ │k={}      => transport={}, name={}, dbg={}, day={}/{}, "
              "ev_day_offset={}, "
              "best_mam={}, "
              "transport_mam={}, transport_time={} => NO TRAFFIC!\n",
              k, t, tt_.transport_name(t), tt_.dbg(t), i, day, ev_day_offset,
              mam_at_stop, ev_mam, ev);
          continue;
        }

        trace(
            "┊ │k={}      => ET FOUND: name={}, dbg={}, at day {} "
            "(day_offset={}) - ev_mam={}, ev_time={}, ev={}\n",
            k, tt_.transport_name(t), tt_.dbg(t), day, ev_day_offset, ev_mam,
            ev, tt_.to_unixtime(day, duration_t{ev_mam}));

        return {t, static_cast<day_idx_t>(as_int(day) - ev_day_offset)};
      }
    }
    return {};
  }

  bool is_transport_active(transport_idx_t const t,
                           std::size_t const day) const {

    if constexpr (Rt) {
      return rtt_->bitfields_[rtt_->transport_traffic_days_[t]].test(day);
    } else {
      if (day >= tt_.bitfields_[tt_.transport_traffic_days_[t]].size()) {
        return false;
      }
      auto const block = tt_.bitfields_[tt_.transport_traffic_days_[t]].blocks_[day / tt_.bitfields_[tt_.transport_traffic_days_[t]].bits_per_block];
      auto const bit = (day % tt_.bitfields_[tt_.transport_traffic_days_[t]].bits_per_block);
      return (block & (std::uint64_t{1U} << bit)) != 0U;
    }
  }

  delta_t time_at_stop(route_idx_t const r,
                       transport const t,
                       stop_idx_t const stop_idx,
                       event_type const ev_type) {
    trace("time at stop: {}\n",
          tt_.to_unixtime(
              t.day_,
              tt_.event_mam(r, t.t_idx_, stop_idx, ev_type).as_duration()));

    return to_delta(t.day_,
                    tt_.event_mam(r, t.t_idx_, stop_idx, ev_type).count());
  }

  delta_t rt_time_at_stop(rt_transport_idx_t const rt_t,
                          stop_idx_t const stop_idx,
                          event_type const ev_type) {
    return to_delta(rtt_->base_day_idx_,
                    rtt_->event_time(rt_t, stop_idx, ev_type));
  }

  delta_t to_delta(day_idx_t const day, std::int16_t const mam) {
    return clamp((as_int(day) - as_int(base_)) * 1440 + mam);
  }

  unixtime_t to_unix(delta_t const t) { return delta_to_unix(base(), t); }

  std::pair<day_idx_t, minutes_after_midnight_t> split(delta_t const x) {
    return split_day_mam(base_, x);
  }

  bool is_intermodal_dest() const { return !dist_to_end_.empty(); }

  void update_time_at_dest(unsigned const k, delta_t const t) {
    for (auto i = k; i != time_at_dest_.size(); ++i) {
      time_at_dest_[i] = get_best(time_at_dest_[i], t);
    }
  }

  int as_int(day_idx_t const d) const { return static_cast<int>(d.v_); }

  template <typename T>
  auto get_begin_it(T const& t) {
    if constexpr (kFwd) {
      return t.begin();
    } else {
      return t.rbegin();
    }
  }

  template <typename T>
  auto get_end_it(T const& t) {
    if constexpr (kFwd) {
      return t.end();
    } else {
      return t.rend();
    }
  }

  timetable const& tt_;
  rt_timetable const* rtt_{nullptr};
  raptor_state& state_;
  std::vector<bool>& is_dest_;
  std::vector<std::uint16_t>& dist_to_end_;
  std::vector<std::uint16_t>& lb_;
  std::array<delta_t, kMaxTransfers + 1> time_at_dest_;
  day_idx_t base_;
  int n_days_;
  raptor_stats stats_;
  std::uint32_t n_locations_, n_routes_, n_rt_transports_;
  clasz_mask_t allowed_claszes_;
};

}  // namespace nigiri::routing
