#pragma once

#include <cassert>

#include "nigiri/common/delta_t.h"
#include "nigiri/common/linear_lower_bound.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/raptor/debug.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/routing/raptor/reconstruct.h"
#include "nigiri/routing/transfer_time_settings.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing {

struct raptor_stats {
  std::map<std::string, std::uint64_t> to_map() const {
    return {
        {"n_routing_time", n_routing_time_},
        {"n_footpaths_visited", n_footpaths_visited_},
        {"n_routes_visited", n_routes_visited_},
        {"n_earliest_trip_calls", n_earliest_trip_calls_},
        {"n_earliest_arrival_updated_by_route",
         n_earliest_arrival_updated_by_route_},
        {"n_earliest_arrival_updated_by_footpath",
         n_earliest_arrival_updated_by_footpath_},
        {"fp_update_prevented_by_lower_bound",
         fp_update_prevented_by_lower_bound_},
        {"route_update_prevented_by_lower_bound",
         route_update_prevented_by_lower_bound_},
    };
  }

  raptor_stats operator+(raptor_stats const& o) const {
    auto copy = *this;
    copy.n_routing_time_ += o.n_routing_time_;
    copy.n_footpaths_visited_ += o.n_footpaths_visited_;
    copy.n_routes_visited_ += o.n_routes_visited_;
    copy.n_earliest_trip_calls_ += o.n_earliest_trip_calls_;
    copy.n_earliest_arrival_updated_by_route_ +=
        o.n_earliest_arrival_updated_by_route_;
    copy.n_earliest_arrival_updated_by_footpath_ +=
        o.n_earliest_arrival_updated_by_footpath_;
    copy.fp_update_prevented_by_lower_bound_ +=
        o.fp_update_prevented_by_lower_bound_;
    copy.route_update_prevented_by_lower_bound_ +=
        o.route_update_prevented_by_lower_bound_;
    return copy;
  }

  std::uint64_t n_routing_time_{0ULL};
  std::uint64_t n_footpaths_visited_{0ULL};
  std::uint64_t n_routes_visited_{0ULL};
  std::uint64_t n_earliest_trip_calls_{0ULL};
  std::uint64_t n_earliest_arrival_updated_by_route_{0ULL};
  std::uint64_t n_earliest_arrival_updated_by_footpath_{0ULL};
  std::uint64_t fp_update_prevented_by_lower_bound_{0ULL};
  std::uint64_t route_update_prevented_by_lower_bound_{0ULL};
};

enum class search_mode { kOneToOne, kOneToAll };

template <direction SearchDir,
          bool Rt,
          via_offset_t Vias,
          search_mode SearchMode>
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
  static constexpr auto const kInvalidArray = []() {
    auto a = std::array<delta_t, Vias + 1>{};
    a.fill(kInvalid);
    return a;
  }();

  static bool is_better(auto a, auto b) { return kFwd ? a < b : a > b; }
  static bool is_better_or_eq(auto a, auto b) { return kFwd ? a <= b : a >= b; }
  static auto get_best(auto a, auto b) { return is_better(a, b) ? a : b; }
  static auto get_best(auto x, auto... y) {
    ((x = get_best(x, y)), ...);
    return x;
  }
  static auto dir(auto a) { return (kFwd ? 1 : -1) * a; }

  raptor(
      timetable const& tt,
      rt_timetable const* rtt,
      raptor_state& state,
      bitvec& is_dest,
      std::array<bitvec, kMaxVias>& is_via,
      std::vector<std::uint16_t>& dist_to_dest,
      hash_map<location_idx_t, std::vector<td_offset>> const& td_dist_to_dest,
      std::vector<std::uint16_t>& lb,
      std::vector<via_stop> const& via_stops,
      day_idx_t const base,
      clasz_mask_t const allowed_claszes,
      bool const require_bike_transport,
      bool const require_car_transport,
      bool const is_wheelchair,
      transfer_time_settings const& tts)
      : tt_{tt},
        rtt_{rtt},
        n_days_{tt_.internal_interval_days().size().count()},
        n_locations_{tt_.n_locations()},
        n_routes_{tt.n_routes()},
        n_rt_transports_{Rt ? rtt->n_rt_transports() : 0U},
        state_{state.resize(n_locations_, n_routes_, n_rt_transports_)},
        tmp_{state_.get_tmp<Vias>()},
        best_{state_.get_best<Vias>()},
        round_times_{state.get_round_times<Vias>()},
        is_dest_{is_dest},
        is_via_{is_via},
        dist_to_end_{dist_to_dest},
        td_dist_to_end_{td_dist_to_dest},
        lb_{lb},
        via_stops_{via_stops},
        base_{base},
        allowed_claszes_{allowed_claszes},
        require_bike_transport_{require_bike_transport},
        require_car_transport_{require_car_transport},
        is_wheelchair_{is_wheelchair},
        transfer_time_settings_{tts} {
    assert(Vias == via_stops_.size());
    reset_arrivals();
    if (!dist_to_end_.empty()) {
      // only used for intermodal queries (dist_to_dest != empty)
      end_reachable_.resize(n_locations_);
      for (auto i = 0U; i != dist_to_end_.size(); ++i) {
        if (dist_to_end_[i] != kUnreachable) {
          end_reachable_.set(i, true);
        }
      }
      for (auto const& [l, _] : td_dist_to_end_) {
        end_reachable_.set(to_idx(l), true);
      }
    }
  }

  algo_stats_t get_stats() const { return stats_; }

  void reset_arrivals() {
    utl::fill(time_at_dest_, kInvalid);
    round_times_.reset(kInvalidArray);
  }

  void next_start_time() {
    utl::fill(best_, kInvalidArray);
    utl::fill(tmp_, kInvalidArray);
    utl::fill(state_.prev_station_mark_.blocks_, 0U);
    utl::fill(state_.station_mark_.blocks_, 0U);
    utl::fill(state_.route_mark_.blocks_, 0U);
    if constexpr (Rt) {
      utl::fill(state_.rt_transport_mark_.blocks_, 0U);
    }
  }

  void add_start(location_idx_t const l, unixtime_t const t) {
    auto const v = (Vias != 0 && is_via_[0][to_idx(l)]) ? 1U : 0U;
    trace_upd(
        "adding start [fwd={}] {}: {}, v={} [current: best={}, round={} => "
        "best={}]\n",
        kFwd, loc{tt_, l}, t, v, to_unix(best_[to_idx(l)][v]),
        to_unix(round_times_[0U][to_idx(l)][v]),
        get_best(t, to_unix(best_[to_idx(l)][v])));
    best_[to_idx(l)][v] =
        get_best(unix_to_delta(base(), t), best_[to_idx(l)][v]);
    round_times_[0U][to_idx(l)][v] =
        get_best(unix_to_delta(base(), t), round_times_[0U][to_idx(l)][v]);
    state_.station_mark_.set(to_idx(l), true);
  }

  void execute(unixtime_t const start_time,
               std::uint8_t const max_transfers,
               unixtime_t const worst_time_at_dest,
               profile_idx_t const prf_idx,
               pareto_set<journey>& results) {
    auto const end_k = std::min(max_transfers, kMaxTransfers) + 2U;

    auto const d_worst_at_dest = unix_to_delta(base(), worst_time_at_dest);
    for (auto& time_at_dest : time_at_dest_) {
      time_at_dest = get_best(d_worst_at_dest, time_at_dest);
    }

    trace_print_init_state();

    for (auto k = 1U; k != end_k; ++k) {
      for (auto i = 0U; i != n_locations_; ++i) {
        for (auto v = 0U; v != Vias + 1; ++v) {
          best_[i][v] = get_best(round_times_[k][i][v], best_[i][v]);
        }
      }
      is_dest_.for_each_set_bit([&](std::uint64_t const i) {
        update_time_at_dest(k, best_[i][Vias]);
      });

      auto any_marked = false;
      state_.station_mark_.for_each_set_bit([&](std::uint64_t const i) {
        for (auto const& r : tt_.location_routes_[location_idx_t{i}]) {
          any_marked = true;
          state_.route_mark_.set(to_idx(r), true);
        }
        if constexpr (Rt) {
          for (auto const& rt_t :
               rtt_->location_rt_transports_[location_idx_t{i}]) {
            any_marked = true;
            state_.rt_transport_mark_.set(to_idx(rt_t), true);
          }
        }
      });

      if (!any_marked) {
        trace_print_state_after_round();
        break;
      }

      std::swap(state_.prev_station_mark_, state_.station_mark_);
      utl::fill(state_.station_mark_.blocks_, 0U);

      any_marked = (allowed_claszes_ == all_clasz_allowed())
                       ? (require_bike_transport_
                              ? (require_car_transport_
                                     ? loop_routes<false, true, true>(k)
                                     : loop_routes<false, true, false>(k))
                              : (require_car_transport_
                                     ? loop_routes<false, false, true>(k)
                                     : loop_routes<false, false, false>(k)))
                       : (require_bike_transport_
                              ? (require_car_transport_
                                     ? loop_routes<true, true, true>(k)
                                     : loop_routes<true, true, false>(k))
                              : (require_car_transport_
                                     ? loop_routes<true, false, true>(k)
                                     : loop_routes<true, false, false>(k)));

      if constexpr (Rt) {
        any_marked |=
            (allowed_claszes_ == all_clasz_allowed())
                ? (require_bike_transport_
                       ? (require_car_transport_
                              ? loop_rt_routes<false, true, true>(k)
                              : loop_rt_routes<false, true, false>(k))
                       : (require_car_transport_
                              ? loop_rt_routes<false, false, true>(k)
                              : loop_rt_routes<false, false, false>(k)))
                : (require_bike_transport_
                       ? (require_car_transport_
                              ? loop_rt_routes<true, true, true>(k)
                              : loop_rt_routes<true, true, false>(k))
                       : (require_car_transport_
                              ? loop_rt_routes<true, false, true>(k)
                              : loop_rt_routes<true, false, false>(k)));
      }

      if (!any_marked) {
        trace_print_state_after_round();
        break;
      }

      utl::fill(state_.route_mark_.blocks_, 0U);

      std::swap(state_.prev_station_mark_, state_.station_mark_);
      utl::fill(state_.station_mark_.blocks_, 0U);

      update_transfers(k);
      update_intermodal_footpaths(k);
      update_footpaths(k, prf_idx);
      update_td_offsets(k, prf_idx);

      trace_print_state_after_round();
    }

    if constexpr (SearchMode == search_mode::kOneToAll) {
      return;
    }

    is_dest_.for_each_set_bit([&](auto const i) {
      for (auto k = 1U; k != end_k; ++k) {
        auto const dest_time = round_times_[k][i][Vias];
        if (dest_time != kInvalid) {
          trace("ADDING JOURNEY: start={}, dest={} @ {}, transfers={}\n",
                start_time, delta_to_unix(base(), round_times_[k][i][Vias]),
                loc{tt_, location_idx_t{i}}, k - 1);
          auto const [optimal, it, dominated_by] = results.add(
              journey{.legs_ = {},
                      .start_time_ = start_time,
                      .dest_time_ = delta_to_unix(base(), dest_time),
                      .dest_ = location_idx_t{i},
                      .transfers_ = static_cast<std::uint8_t>(k - 1)});
          if (!optimal) {
            trace("  DOMINATED BY: start={}, dest={} @ {}, transfers={}\n",
                  dominated_by->start_time_, dominated_by->dest_time_,
                  loc{tt_, dominated_by->dest_}, dominated_by->transfers_);
          }
        }
      }
    });
  }

  void reconstruct(query const& q, journey& j) {
    if constexpr (SearchMode == search_mode::kOneToAll) {
      return;
    }
    trace("reconstruct({} - {}, {} transfers", j.departure_time(),
          j.arrival_time(), j.transfers_);
    reconstruct_journey<SearchDir>(tt_, rtt_, q, state_, j, base(), base_);
  }

private:
  date::sys_days base() const {
    return tt_.internal_interval_days().from_ + as_int(base_) * date::days{1};
  }

  template <bool WithClaszFilter, bool WithBikeFilter, bool WithCarFilter>
  bool loop_routes(unsigned const k) {
    auto any_marked = false;
    state_.route_mark_.for_each_set_bit([&](auto const r_idx) {
      auto const r = route_idx_t{r_idx};

      if constexpr (WithClaszFilter) {
        if (!is_allowed(allowed_claszes_, tt_.route_clasz_[r])) {
          return;
        }
      }

      auto section_bike_filter = false;
      if constexpr (WithBikeFilter) {
        auto const bikes_allowed_on_all_sections =
            tt_.route_bikes_allowed_.test(r_idx * 2);
        if (!bikes_allowed_on_all_sections) {
          auto const bikes_allowed_on_some_sections =
              tt_.route_bikes_allowed_.test(r_idx * 2 + 1);
          if (!bikes_allowed_on_some_sections) {
            return;
          }
          section_bike_filter = true;
        }
      }

      auto section_car_filter = false;
      if constexpr (WithCarFilter) {
        auto const cars_allowed_on_all_sections =
            tt_.route_cars_allowed_.test(r_idx * 2);
        if (!cars_allowed_on_all_sections) {
          auto const cars_allowed_on_some_sections =
              tt_.route_cars_allowed_.test(r_idx * 2 + 1);
          if (!cars_allowed_on_some_sections) {
            return;
          }
          section_car_filter = true;
        }
      }

      ++stats_.n_routes_visited_;
      trace("┊ ├k={} updating route {}\n", k, r);
      any_marked |=
          section_bike_filter
              ? (section_car_filter ? update_route<true, true>(k, r)
                                    : update_route<true, false>(k, r))
              : (section_car_filter ? update_route<false, true>(k, r)
                                    : update_route<false, false>(k, r));
    });
    return any_marked;
  }

  template <bool WithClaszFilter, bool WithBikeFilter, bool WithCarFilter>
  bool loop_rt_routes(unsigned const k) {
    auto any_marked = false;
    state_.rt_transport_mark_.for_each_set_bit([&](auto const rt_t_idx) {
      auto const rt_t = rt_transport_idx_t{rt_t_idx};

      if constexpr (WithClaszFilter) {
        if (!is_allowed(allowed_claszes_,
                        rtt_->rt_transport_section_clasz_[rt_t][0])) {
          return;
        }
      }

      auto section_bike_filter = false;
      if constexpr (WithBikeFilter) {
        auto const bikes_allowed_on_all_sections =
            rtt_->rt_transport_bikes_allowed_.test(rt_t_idx * 2);
        if (!bikes_allowed_on_all_sections) {
          auto const bikes_allowed_on_some_sections =
              rtt_->rt_transport_bikes_allowed_.test(rt_t_idx * 2 + 1);
          if (!bikes_allowed_on_some_sections) {
            return;
          }
          section_bike_filter = true;
        }
      }

      auto section_car_filter = false;
      if constexpr (WithCarFilter) {
        auto const cars_allowed_on_all_sections =
            rtt_->rt_transport_cars_allowed_.test(rt_t_idx * 2);
        if (!cars_allowed_on_all_sections) {
          auto const cars_allowed_on_some_sections =
              rtt_->rt_transport_cars_allowed_.test(rt_t_idx * 2 + 1);
          if (!cars_allowed_on_some_sections) {
            return;
          }
          section_car_filter = true;
        }
      }

      ++stats_.n_routes_visited_;
      trace("┊ ├k={} updating rt transport {}\n", k, rt_t);
      any_marked |=
          section_bike_filter
              ? (section_car_filter ? update_rt_transport<true, true>(k, rt_t)
                                    : update_rt_transport<true, false>(k, rt_t))
              : (section_car_filter
                     ? update_rt_transport<false, true>(k, rt_t)
                     : update_rt_transport<false, false>(k, rt_t));
    });
    return any_marked;
  }

  void update_transfers(unsigned const k) {
    state_.prev_station_mark_.for_each_set_bit([&](auto&& i) {
      for (auto v = 0U; v != Vias + 1; ++v) {
        auto const tmp_time = tmp_[i][v];
        if (tmp_time == kInvalid) {
          continue;
        }

        auto const is_via = v != Vias && is_via_[v][i];
        auto const target_v = is_via ? v + 1 : v;
        auto const is_dest = target_v == Vias && is_dest_[i];
        auto const stay = is_via ? via_stops_[v].stay_ : 0_minutes;

        trace(
            "  loc={}, v={}, tmp={}, is_dest={}, is_via={}, target_v={}, "
            "stay={}\n",
            loc{tt_, location_idx_t{i}}, v, to_unix(tmp_time), is_dest, is_via,
            target_v, stay);

        auto const transfer_time =
            (!is_intermodal_dest() && is_dest)
                ? 0
                : dir(adjusted_transfer_time(
                      transfer_time_settings_,
                      tt_.locations_.transfer_time_[location_idx_t{i}]
                          .count()));
        auto const fp_target_time =
            clamp(tmp_time + transfer_time + dir(stay.count()));

        trace(
            "    transfer_time={}, fp_target_time={}, best@target={}, "
            "dest={}\n",
            transfer_time, to_unix(fp_target_time), to_unix(best_[i][target_v]),
            to_unix(time_at_dest_[k]));

        if (is_better(fp_target_time, best_[i][target_v]) &&
            is_better(fp_target_time, time_at_dest_[k])) {
          if (lb_[i] == kUnreachable ||
              !is_better(fp_target_time + dir(lb_[i]), time_at_dest_[k])) {
            ++stats_.fp_update_prevented_by_lower_bound_;
            return;
          }

          ++stats_.n_earliest_arrival_updated_by_footpath_;
          round_times_[k][i][target_v] = fp_target_time;
          best_[i][target_v] = fp_target_time;
          state_.station_mark_.set(i, true);
          if (is_dest) {
            update_time_at_dest(k, fp_target_time);
          }
        }
      }
    });
  }

  void update_footpaths(unsigned const k, profile_idx_t const prf_idx) {
    state_.prev_station_mark_.for_each_set_bit([&](std::uint64_t const i) {
      auto const l_idx = location_idx_t{i};
      if constexpr (Rt) {
        if (prf_idx != 0U && (kFwd ? rtt_->has_td_footpaths_out_
                                   : rtt_->has_td_footpaths_in_)[prf_idx]
                                 .test(l_idx)) {
          return;
        }
      }

      auto const& fps = kFwd ? tt_.locations_.footpaths_out_[prf_idx][l_idx]
                             : tt_.locations_.footpaths_in_[prf_idx][l_idx];

      for (auto const& fp : fps) {
        ++stats_.n_footpaths_visited_;

        auto const target = to_idx(fp.target());

        for (auto v = 0U; v != Vias + 1; ++v) {
          auto const tmp_time = tmp_[i][v];
          if (tmp_time == kInvalid) {
            continue;
          }

          auto const start_is_via =
              v != Vias && is_via_[v][static_cast<bitvec::size_type>(i)];
          auto const start_v = start_is_via ? v + 1 : v;

          auto const target_is_via =
              start_v != Vias && is_via_[start_v][target];
          auto const target_v = target_is_via ? start_v + 1 : start_v;
          auto stay = 0_minutes;
          if (start_is_via) {
            stay += via_stops_[v].stay_;
          }
          if (target_is_via) {
            stay += via_stops_[start_v].stay_;
          }

          auto const fp_target_time = clamp(
              tmp_time + dir(adjusted_transfer_time(transfer_time_settings_,
                                                    fp.duration().count()) +
                             stay.count()));

          if (is_better(fp_target_time, best_[target][target_v]) &&
              is_better(fp_target_time, time_at_dest_[k])) {
            auto const lower_bound = lb_[target];
            if (lower_bound == kUnreachable ||
                !is_better(fp_target_time + dir(lower_bound),
                           time_at_dest_[k])) {
              ++stats_.fp_update_prevented_by_lower_bound_;
              trace_upd(
                  "┊ ├k={} *** LB NO UPD: (from={}, tmp={}) --{}--> (to={}, "
                  "best={}) --> update => {}, LB={}, LB_AT_DEST={}, DEST={}\n",
                  k, loc{tt_, l_idx}, to_unix(tmp_[to_idx(l_idx)][v]),
                  adjusted_transfer_time(transfer_time_settings_,
                                         fp.duration()),
                  loc{tt_, fp.target()}, best_[target][target_v],
                  to_unix(fp_target_time), lower_bound,
                  to_unix(clamp(fp_target_time + dir(lower_bound))),
                  to_unix(time_at_dest_[k]));
              continue;
            }

            trace_upd(
                "┊ ├k={}   footpath: ({}, tmp={}) --{}--> ({}, best={}) --> "
                "update => {}, v={}->{}, stay={}\n",
                k, loc{tt_, l_idx}, to_unix(tmp_[to_idx(l_idx)][v]),
                adjusted_transfer_time(transfer_time_settings_, fp.duration()),
                loc{tt_, fp.target()}, to_unix(best_[target][target_v]),
                to_unix(fp_target_time), v, target_v, stay);

            ++stats_.n_earliest_arrival_updated_by_footpath_;
            round_times_[k][target][target_v] = fp_target_time;
            best_[target][target_v] = fp_target_time;
            state_.station_mark_.set(target, true);
            if (target_v == Vias && is_dest_[target]) {
              update_time_at_dest(k, fp_target_time);
            }
          } else {
            trace(
                "┊ ├k={}   NO FP UPDATE: {} [best={}] --{}--> {} "
                "[best={}, time_at_dest={}]\n",
                k, loc{tt_, l_idx}, to_unix(best_[to_idx(l_idx)][target_v]),
                adjusted_transfer_time(transfer_time_settings_, fp.duration()),
                loc{tt_, fp.target()}, to_unix(best_[target][target_v]),
                to_unix(time_at_dest_[k]));
          }
        }
      }
    });
  }

  void update_td_offsets(unsigned const k, profile_idx_t const prf_idx) {
    if constexpr (!Rt) {
      return;
    }

    if (prf_idx == 0U) {
      return;
    }

    state_.prev_station_mark_.for_each_set_bit([&](std::uint64_t const i) {
      auto const l_idx = location_idx_t{i};
      if (!(kFwd ? rtt_->has_td_footpaths_out_
                 : rtt_->has_td_footpaths_in_)[prf_idx]
               .test(l_idx)) {
        return;
      }

      auto const& fps = kFwd ? rtt_->td_footpaths_out_[prf_idx][l_idx]
                             : rtt_->td_footpaths_in_[prf_idx][l_idx];

      for (auto v = 0U; v != Vias + 1; ++v) {
        auto const tmp_time = tmp_[i][v];
        if (tmp_time == kInvalid) {
          continue;
        }
        for_each_footpath<
            SearchDir>(fps, to_unix(tmp_time), [&](footpath const fp) {
          ++stats_.n_footpaths_visited_;

          auto const target = to_idx(fp.target());

          auto const start_is_via =
              v != Vias && is_via_[v][static_cast<bitvec::size_type>(i)];
          auto const start_v = start_is_via ? v + 1 : v;

          auto const target_is_via =
              start_v != Vias && is_via_[start_v][target];
          auto const target_v = target_is_via ? start_v + 1 : start_v;
          auto stay = 0_minutes;
          if (start_is_via) {
            stay += via_stops_[v].stay_;
          }
          if (target_is_via) {
            stay += via_stops_[start_v].stay_;
          }

          auto const fp_target_time =
              clamp(tmp_time + dir(fp.duration().count() + stay.count()));

          if (is_better(fp_target_time, best_[target][target_v]) &&
              is_better(fp_target_time, time_at_dest_[k])) {
            auto const lower_bound = lb_[target];
            if (lower_bound == kUnreachable ||
                !is_better(fp_target_time + dir(lower_bound),
                           time_at_dest_[k])) {
              ++stats_.fp_update_prevented_by_lower_bound_;
              trace_upd(
                  "┊ ├k={} *** LB NO TD FP UPD: (from={}, tmp={}) --{}--> "
                  "(to={}, best={}) --> update => {}, LB={}, LB_AT_DEST={}, "
                  "DEST={}\n",
                  k, loc{tt_, l_idx}, to_unix(tmp_[to_idx(l_idx)][v]),
                  fp.duration(), loc{tt_, fp.target()}, best_[target][target_v],
                  fp_target_time, lower_bound,
                  to_unix(clamp(fp_target_time + dir(lower_bound))),
                  to_unix(time_at_dest_[k]));
              return utl::cflow::kContinue;
            }

            trace_upd(
                "┊ ├k={}   td footpath: ({}, tmp={}) --{}--> ({}, best={}) --> "
                "update => {}, v={}->{}, stay={}\n",
                k, loc{tt_, l_idx}, to_unix(tmp_[to_idx(l_idx)][v]),
                fp.duration(), loc{tt_, fp.target()},
                to_unix(best_[target][target_v]), to_unix(fp_target_time), v,
                target_v, stay);

            ++stats_.n_earliest_arrival_updated_by_footpath_;
            round_times_[k][target][target_v] = fp_target_time;
            best_[target][target_v] = fp_target_time;
            state_.station_mark_.set(target, true);
            if (is_dest_[target]) {
              update_time_at_dest(k, fp_target_time);
            }
          } else {
            trace(
                "┊ ├k={}   NO TD FP UPDATE: {} [best={}] --{}--> {} "
                "[best={}, time_at_dest={}]\n",
                k, loc{tt_, l_idx}, best_[to_idx(l_idx)][v],
                adjusted_transfer_time(transfer_time_settings_, fp.duration()),
                loc{tt_, fp.target()}, best_[target][v],
                to_unix(time_at_dest_[k]));
          }

          return utl::cflow::kContinue;
        });
      }
    });
  }

  void update_intermodal_footpaths(unsigned const k) {
    if (dist_to_end_.empty()) {
      return;
    }

    state_.prev_station_mark_.for_each_set_bit([&](auto const i) {
      if (!end_reachable_.test(i)) {
        trace_upd("┊ ├k={}   no end_reachable: {}\n", k,
                  loc{tt_, location_idx_t{i}});
        [[likely]];
        return;
      }

      trace_upd("┊ ├k={}   end_reachable: {}\n", k,
                loc{tt_, location_idx_t{i}});

      auto const l = location_idx_t{i};
      if (dist_to_end_[i] != std::numeric_limits<std::uint16_t>::max()) {
        [[likely]];

        // Case 1: l is last via -> add stay
        if constexpr (Vias != 0U) {
          constexpr auto v = Vias - 1U;
          if (tmp_[i][v] != kInvalid && is_via_[v][i]) {
            auto const end_time = clamp(tmp_[i][v]  //
                                        + dir(via_stops_[v].stay_.count())  //
                                        + dir(dist_to_end_[i]));

            trace_upd(
                "┊ ├k={}, INTERMODAL FOOTPATH FROM LAST VIA: ({}, tmp={}) "
                "--({} +stay={})--> "
                "({}, best={})",
                k, loc{tt_, l}, to_unix(tmp_[to_idx(l)][v]), dist_to_end_[i],
                via_stops_[v].stay_,
                loc{tt_, location_idx_t{kIntermodalTarget}},
                to_unix(best_[kIntermodalTarget][Vias]), to_unix(end_time));

            if (is_better(end_time, best_[kIntermodalTarget][Vias])) {
              round_times_[k][kIntermodalTarget][Vias] = end_time;
              best_[kIntermodalTarget][Vias] = end_time;
              update_time_at_dest(k, end_time);
              trace_upd(" -> update\n");
            } else {
              trace_upd(" -> no update\n");
            }
          }
        }

        // Case 2: l is no via -> don't add stay
        auto const tmp_time = tmp_[i][Vias];
        if (tmp_time == kInvalid) {
          trace_upd("┊ ├k={}, loc={} NOT REACHED\n", k, loc{tt_, l});
          return;
        }

        auto const end_time = clamp(tmp_time + dir(dist_to_end_[i]));

        trace_upd(
            "┊ ├k={}, INTERMODAL FOOTPATH: ({}, tmp={}) --{}--> "
            "({}, best={})",
            k, loc{tt_, l}, to_unix(tmp_[to_idx(l)][Vias]), dist_to_end_[i],
            loc{tt_, location_idx_t{kIntermodalTarget}},
            to_unix(best_[kIntermodalTarget][Vias]), to_unix(end_time));

        if (is_better(end_time, best_[kIntermodalTarget][Vias])) {
          round_times_[k][kIntermodalTarget][Vias] = end_time;
          best_[kIntermodalTarget][Vias] = end_time;
          update_time_at_dest(k, end_time);
          trace_upd(" -> update\n");
        } else {
          trace_upd(" -> no update\n");
        }
      }

      if (auto const it = td_dist_to_end_.find(l); it != end(td_dist_to_end_)) {
        [[unlikely]];

        auto const fp_start_time = tmp_[i][Vias];
        if (fp_start_time == kInvalid) {
          return;
        }
        auto const fp =
            get_td_duration<SearchDir>(it->second, to_unix(fp_start_time));
        if (fp.has_value()) {
          auto const& [duration, _] = *fp;
          auto const end_time = clamp(fp_start_time + dir(duration.count()));

          if (is_better(end_time, best_[kIntermodalTarget][Vias])) {
            round_times_[k][kIntermodalTarget][Vias] = end_time;
            best_[kIntermodalTarget][Vias] = end_time;
            update_time_at_dest(k, end_time);

            trace(
                "┊ │k={}  TD INTERMODAL FOOTPATH: location={}, "
                "start_time={}, dist_to_end={} --> update to {}\n",
                k, loc{tt_, l}, to_unix(fp_start_time), duration,
                to_unix(end_time));
          } else {
            trace(
                "┊ │k={}  TD INTERMODAL FOOTPATH: location={}, "
                "start_time={}, dist_to_end={} --> NO update to {} best={}\n",
                k, loc{tt_, l}, to_unix(fp_start_time), duration,
                to_unix(end_time), best_[kIntermodalTarget][Vias]);
          }
        }
      }
    });
  }

  template <bool WithSectionBikeFilter, bool WithSectionCarFilter>
  bool update_rt_transport(unsigned const k, rt_transport_idx_t const rt_t) {
    auto const stop_seq = rtt_->rt_transport_location_seq_[rt_t];
    auto et = std::array<bool, Vias + 1>{};
    auto v_offset = std::array<std::size_t, Vias + 1>{};
    auto any_marked = false;

    for (auto i = 0U; i != stop_seq.size(); ++i) {
      auto const stop_idx =
          static_cast<stop_idx_t>(kFwd ? i : stop_seq.size() - i - 1U);
      auto const stp = stop{stop_seq[stop_idx]};
      auto const l_idx = cista::to_idx(stp.location_idx());
      auto const is_first = i == 0U;
      auto const is_last = i == stop_seq.size() - 1U;

      if constexpr (WithSectionBikeFilter) {
        if (!is_first &&
            !rtt_->rt_bikes_allowed_per_section_[rt_t][kFwd ? stop_idx - 1
                                                            : stop_idx]) {
          et.fill(false);
          v_offset.fill(0);
        }
      }

      if constexpr (WithSectionCarFilter) {
        if (!is_first &&
            !rtt_->rt_cars_allowed_per_section_[rt_t][kFwd ? stop_idx - 1
                                                           : stop_idx]) {
          et.fill(false);
          v_offset.fill(0);
        }
      }

      if ((kFwd && stop_idx != 0U) ||
          (kBwd && stop_idx != stop_seq.size() - 1U)) {
        auto const by_transport = rt_time_at_stop(
            rt_t, stop_idx, kFwd ? event_type::kArr : event_type::kDep);
        for (auto j = 0U; j != Vias + 1; ++j) {
          auto const v = Vias - j;
          auto target_v = v + v_offset[v];
          if (et[v] && stp.can_finish<SearchDir>(is_wheelchair_)) {
            auto const is_via = target_v != Vias && is_via_[target_v][l_idx];
            auto const is_no_stay_via =
                is_via && via_stops_[target_v].stay_ == 0_minutes;

            if (is_no_stay_via) {
              ++v_offset[v];
              ++target_v;
            }

            auto current_best =
                get_best(round_times_[k - 1][l_idx][target_v],
                         tmp_[l_idx][target_v], best_[l_idx][target_v]);

            if (is_better(by_transport, time_at_dest_[k]) &&
                lb_[l_idx] != kUnreachable &&
                is_better(by_transport + dir(lb_[l_idx]), time_at_dest_[k])) {
              trace_upd(
                  "┊ │k={}    RT | name={}, dbg={}, time_by_transport={}, "
                  "BETTER THAN current_best={} => update, {} marking station "
                  "{}!\n",
                  k, rtt_->default_trip_short_name(tt_, rt_t),
                  rtt_->dbg(tt_, rt_t), to_unix(by_transport),
                  to_unix(current_best),
                  !is_better(by_transport, current_best) ? "NOT" : "",
                  loc{tt_, stp.location_idx()});

              ++stats_.n_earliest_arrival_updated_by_route_;
              tmp_[l_idx][target_v] =
                  get_best(by_transport, tmp_[l_idx][target_v]);
              state_.station_mark_.set(l_idx, true);
              if (is_better(by_transport, current_best)) {
                current_best = by_transport;
              }
              any_marked = true;
            }
          }
        }
      }

      if (lb_[l_idx] == kUnreachable) {
        break;
      }

      if (is_last || !(stp.can_start<SearchDir>(is_wheelchair_)) ||
          !state_.prev_station_mark_[l_idx]) {
        continue;
      }

      auto const by_transport = rt_time_at_stop(
          rt_t, stop_idx, kFwd ? event_type::kDep : event_type::kArr);
      for (auto v = 0U; v != Vias + 1; ++v) {
        auto const target_v = v + v_offset[v];
        auto const prev_round_time = round_times_[k - 1][l_idx][target_v];
        if (is_better_or_eq(prev_round_time, by_transport)) {
          et[v] = true;
          v_offset[v] = 0;
        }
      }
    }
    return any_marked;
  }

  template <bool WithSectionBikeFilter, bool WithSectionCarFilter>
  bool update_route(unsigned const k, route_idx_t const r) {
    auto const stop_seq = tt_.route_location_seq_[r];
    bool any_marked = false;

    auto et = std::array<transport, Vias + 1>{};
    auto v_offset = std::array<std::size_t, Vias + 1>{};

    for (auto i = 0U; i != stop_seq.size(); ++i) {
      auto const stop_idx =
          static_cast<stop_idx_t>(kFwd ? i : stop_seq.size() - i - 1U);
      auto const stp = stop{stop_seq[stop_idx]};
      auto const l_idx = cista::to_idx(stp.location_idx());
      auto const is_first = i == 0U;
      auto const is_last = i == stop_seq.size() - 1U;

      auto current_best = std::array<delta_t, Vias + 1>{};
      current_best.fill(kInvalid);

      // v = via state when entering the transport
      // v + v_offset = via state at the current stop after entering the
      // transport (v_offset > 0 if the transport passes via stops)
      for (auto j = 0U; j != Vias + 1; ++j) {
        auto const v = Vias - j;
        if (!et[v].is_valid() && !state_.prev_station_mark_[l_idx]) {
          trace(
              "┊ │k={} v={}  stop_idx={} {}: not marked, no et - "
              "skip\n",
              k, v, stop_idx, loc{tt_, location_idx_t{l_idx}});
          continue;
        }

        trace(
            "┊ │k={} v={}(+{})  stop_idx={}, location={}, round_times={}, "
            "best={}, "
            "tmp={}\n",
            k, v, v_offset[v], stop_idx, loc{tt_, stp.location_idx()},
            to_unix(round_times_[k - 1][l_idx][v]), to_unix(best_[l_idx][v]),
            to_unix(tmp_[l_idx][v]));

        if constexpr (WithSectionBikeFilter) {
          if (!is_first &&
              !tt_.route_bikes_allowed_per_section_[r][kFwd ? stop_idx - 1
                                                            : stop_idx]) {
            et[v] = {};
            v_offset[v] = 0;
          }
        }

        if constexpr (WithSectionCarFilter) {
          if (!is_first &&
              !tt_.route_cars_allowed_per_section_[r][kFwd ? stop_idx - 1
                                                           : stop_idx]) {
            et[v] = {};
            v_offset[v] = 0;
          }
        }

        auto target_v = v + v_offset[v];

        if (et[v].is_valid() && stp.can_finish<SearchDir>(is_wheelchair_)) {
          auto const by_transport = time_at_stop(
              r, et[v], stop_idx, kFwd ? event_type::kArr : event_type::kDep);

          auto const is_via = target_v != Vias && is_via_[target_v][l_idx];
          auto const is_no_stay_via =
              is_via && via_stops_[target_v].stay_ == 0_minutes;

          if (Vias != 0) {
            trace_upd(
                "┊ │k={} v={}(+{})={} via_count={} is_via_dest={} stay={} "
                "is_via={} is_dest={}\n",
                k, v, v_offset[v], target_v, Vias,
                target_v != Vias ? is_via_[target_v][l_idx] : is_dest_[l_idx],
                via_stops_[target_v].stay_, is_no_stay_via, is_dest_[l_idx]);
          }

          if (is_no_stay_via) {
            ++v_offset[v];
            ++target_v;
          }

          current_best[v] =
              get_best(round_times_[k - 1][l_idx][target_v],
                       tmp_[l_idx][target_v], best_[l_idx][target_v]);

          assert(by_transport != std::numeric_limits<delta_t>::min() &&
                 by_transport != std::numeric_limits<delta_t>::max());
          if (is_better(by_transport, time_at_dest_[k]) &&
              lb_[l_idx] != kUnreachable &&
              is_better(by_transport + dir(lb_[l_idx]), time_at_dest_[k])) {
            trace_upd(
                "┊ │k={} v={}->{}    name={}, dbg={}, time_by_transport={}, "
                "BETTER THAN current_best={} => update, {} marking station "
                "{}!\n",
                k, v, target_v, tt_.transport_name(et[v].t_idx_),
                tt_.dbg(et[v].t_idx_), to_unix(by_transport),
                to_unix(current_best[v]),
                !is_better(by_transport, current_best[v]) ? "NOT" : "",
                loc{tt_, stp.location_idx()});

            ++stats_.n_earliest_arrival_updated_by_route_;
            tmp_[l_idx][target_v] =
                get_best(by_transport, tmp_[l_idx][target_v]);
            state_.station_mark_.set(l_idx, true);
            if (is_better(by_transport, current_best[v])) {
              current_best[v] = by_transport;
            }
            any_marked = true;
          } else {
            trace(
                "┊ │k={} v={}->{}    *** NO UPD: at={}, name={}, dbg={}, "
                "time_by_transport={}, current_best=min({}, {}, {})={} => {} "
                "- "
                "LB={}, LB_AT_DEST={}, TIME_AT_DEST={}, "
                "(is_better(by_transport={}={}, current_best={}={})={}, "
                "is_better(by_transport={}={}, time_at_dest_={}={})={}, "
                "reachable={}, "
                "is_better(lb={}={}, time_at_dest_={}={})={})!\n",
                k, v, target_v, loc{tt_, location_idx_t{l_idx}},
                tt_.transport_name(et[v].t_idx_), tt_.dbg(et[v].t_idx_),
                to_unix(by_transport),
                to_unix(round_times_[k - 1][l_idx][target_v]),
                to_unix(best_[l_idx][target_v]), to_unix(tmp_[l_idx][target_v]),
                to_unix(current_best[v]), loc{tt_, location_idx_t{l_idx}},
                lb_[l_idx], to_unix(time_at_dest_[k]),
                to_unix(clamp(by_transport + dir(lb_[l_idx]))), by_transport,
                to_unix(by_transport), current_best[v],
                to_unix(current_best[v]),
                is_better(by_transport, current_best[v]), by_transport,
                to_unix(by_transport), time_at_dest_[k],
                to_unix(time_at_dest_[k]),
                is_better(by_transport, time_at_dest_[k]),
                lb_[l_idx] != kUnreachable, by_transport + dir(lb_[l_idx]),
                to_unix(clamp(by_transport + dir(lb_[l_idx]))),
                time_at_dest_[k], to_unix(time_at_dest_[k]),
                to_unix(time_at_dest_[k]),
                is_better(clamp(by_transport + dir(lb_[l_idx])),
                          time_at_dest_[k]));
          }
        } else {
          trace(
              "┊ │k={} v={}->{}    *** NO UPD: no_trip={}, in_allowed={}, "
              "out_allowed={}, label_allowed={}\n",
              k, v, target_v, !et[v].is_valid(), stp.in_allowed(),
              stp.out_allowed(), (kFwd ? stp.out_allowed() : stp.in_allowed()));
        }
      }

      if (is_last || !stp.can_start<SearchDir>(is_wheelchair_) ||
          !state_.prev_station_mark_[l_idx]) {
        continue;
      }

      if (lb_[l_idx] == kUnreachable) {
        break;
      }

      for (auto v = 0U; v != Vias + 1; ++v) {
        if (!et[v].is_valid() && !state_.prev_station_mark_[l_idx]) {
          continue;
        }

        auto const target_v = v + v_offset[v];
        auto const et_time_at_stop =
            et[v].is_valid()
                ? time_at_stop(r, et[v], stop_idx,
                               kFwd ? event_type::kDep : event_type::kArr)
                : kInvalid;
        auto const prev_round_time = round_times_[k - 1][l_idx][target_v];
        if (prev_round_time != kInvalid &&
            is_better_or_eq(prev_round_time, et_time_at_stop)) {
          auto const [day, mam] = split(prev_round_time);
          auto const new_et = get_earliest_transport(k, r, stop_idx, day, mam,
                                                     stp.location_idx());
          current_best[v] = get_best(current_best[v], best_[l_idx][target_v],
                                     tmp_[l_idx][target_v]);
          if (new_et.is_valid() &&
              (current_best[v] == kInvalid ||
               is_better_or_eq(
                   time_at_stop(r, new_et, stop_idx,
                                kFwd ? event_type::kDep : event_type::kArr),
                   et_time_at_stop))) {
            et[v] = new_et;
            v_offset[v] = 0;
            trace("┊ │k={} v={}    update et: time_at_stop={}\n", k, v,
                  to_unix(et_time_at_stop));
          } else if (new_et.is_valid()) {
            trace("┊ │k={} v={}    update et: no update time_at_stop={}\n", k,
                  v, to_unix(et_time_at_stop));
          }
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

    auto const event_times = tt_.event_times_at_stop(
        r, stop_idx, kFwd ? event_type::kDep : event_type::kArr);

    auto const seek_first_day = [&]() {
      return linear_lb(get_begin_it(event_times), get_end_it(event_times),
                       mam_at_stop,
                       [&](delta const a, minutes_after_midnight_t const b) {
                         return is_better(a.mam(), b.count());
                       });
    };

    trace("┊ │k={}    et: current_best_at_stop={}, stop_idx={}, location={}\n",
          k, tt_.to_unixtime(day_at_stop, mam_at_stop), stop_idx,
          loc{tt_, stop{tt_.route_location_seq_[r][stop_idx]}.location_idx()});

    auto const n_days_to_iterate = kMaxTravelTime / std::chrono::days{1} + 1U;
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
      return tt_.bitfields_[tt_.transport_traffic_days_[t]].test(day);
    }
  }

  delta_t time_at_stop(route_idx_t const r,
                       transport const t,
                       stop_idx_t const stop_idx,
                       event_type const ev_type) {
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
    if constexpr (SearchMode == search_mode::kOneToAll) {
      return;
    }
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
  int n_days_;
  std::uint32_t n_locations_, n_routes_, n_rt_transports_;
  raptor_state& state_;
  bitvec end_reachable_;
  std::span<std::array<delta_t, Vias + 1>> tmp_;
  std::span<std::array<delta_t, Vias + 1>> best_;
  flat_matrix_view<std::array<delta_t, Vias + 1>> round_times_;
  bitvec const& is_dest_;
  std::array<bitvec, kMaxVias> const& is_via_;
  std::vector<std::uint16_t> const& dist_to_end_;
  hash_map<location_idx_t, std::vector<td_offset>> const& td_dist_to_end_;
  std::vector<std::uint16_t> const& lb_;
  std::vector<via_stop> const& via_stops_;
  std::array<delta_t, kMaxTransfers + 2> time_at_dest_;
  day_idx_t base_;
  raptor_stats stats_;
  clasz_mask_t allowed_claszes_;
  bool require_bike_transport_;
  bool require_car_transport_;
  bool is_wheelchair_;
  transfer_time_settings transfer_time_settings_;
};

}  // namespace nigiri::routing
