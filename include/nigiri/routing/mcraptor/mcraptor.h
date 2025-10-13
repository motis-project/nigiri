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
#include "nigiri/routing/mcraptor/mcraptor_bag.h"

namespace nigiri::routing {

template <direction SearchDir,
          bool Rt,
          via_offset_t Vias,
          search_mode SearchMode>
struct mcraptor {
  using algo_state_t = raptor_state;
  using algo_stats_t = raptor_stats;

  static constexpr bool kUseLowerBounds = true;
  static constexpr auto const kFwd = (SearchDir == direction::kForward);
  static constexpr auto const kInvalid = kInvalidDelta<SearchDir>;
  static constexpr auto const kUnreachable =
      std::numeric_limits<std::uint16_t>::max();

  static bool is_better(auto a, auto b) { return kFwd ? a < b : a > b; }
  static bool is_better_or_eq(auto a, auto b) { return kFwd ? a <= b : a >= b; }
  static auto get_best(auto a, auto b) { return is_better(a, b) ? a : b; }
  static auto get_best(auto x, auto... y) {
    ((x = get_best(x, y)), ...);
    return x;
  }
  static auto dir(auto a) { return (kFwd ? 1 : -1) * a; }

  mcraptor(
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
      bool const is_wheelchair,
      transfer_time_settings const& tts)
      : tt_{tt},
        rtt_{rtt},
        n_days_{tt_.internal_interval_days().size().count()},
        n_locations_{tt_.n_locations()},
        n_routes_{tt.n_routes()},
        n_rt_transports_{Rt ? rtt->n_rt_transports() : 0U},
        state_{state.resize(n_locations_, n_routes_, n_rt_transports_)},
        is_dest_{is_dest},
        is_via_{is_via},
        dist_to_end_{dist_to_dest},
        td_dist_to_end_{td_dist_to_dest},
        lb_{lb},
        via_stops_{via_stops},
        base_{base},
        allowed_claszes_{allowed_claszes},
        require_bike_transport_{require_bike_transport},
        is_wheelchair_{is_wheelchair},
        transfer_time_settings_{tts} {
    assert(Vias == via_stops_.size());
    // only used for intermodal queries (dist_to_dest != empty)
    for (auto i = 0U; i != dist_to_dest.size(); ++i) {
      state_.end_reachable_.set(i, dist_to_dest[i] != kUnreachable);
    }
    for (auto const& [l, _] : td_dist_to_end_) {
      state_.end_reachable_.set(to_idx(l), true);
    }

    tmp_bags_ = vector<mcraptor_bag>{n_locations_, mcraptor_bag()};
    best_bags_ = vector<mcraptor_bag>{n_locations_, mcraptor_bag()};
    round_bags_ = vector<vector<mcraptor_bag>>{
        kMaxTransfers + 1,
        vector<mcraptor_bag>{n_locations_, mcraptor_bag()}};
    dest_bags_ = vector<mcraptor_bag>{kMaxTransfers + 1, mcraptor_bag()};
    reset_arrivals();
  }

  algo_stats_t get_stats() const { return stats_; }

  void reset_arrivals() {
    for (int k = 0; k < kMaxTransfers + 1; ++k) {
      for (auto& round_bag : round_bags_[k]) {
        round_bag.reset();
      }
      dest_bags_[k].reset();
    }
  }

  void next_start_time() {
    for (int l_idx = 0; l_idx < n_locations_; ++l_idx) {
      best_bags_[l_idx].reset();
      tmp_bags_[l_idx].reset();
    }
    utl::fill(state_.prev_station_mark_.blocks_, 0U);
    utl::fill(state_.station_mark_.blocks_, 0U);
    utl::fill(state_.route_mark_.blocks_, 0U);
  }

  void add_start(location_idx_t const l, unixtime_t const t) {
    best_bags_[to_idx(l)].add(mcraptor_label(unix_to_delta(base(), t)));
    round_bags_[0U][to_idx(l)].add(mcraptor_label(unix_to_delta(base(), t)));
    state_.station_mark_.set(to_idx(l), true);
  }

  void execute(unixtime_t const start_time,
               std::uint8_t const max_transfers,
               unixtime_t const worst_time_at_dest,
               profile_idx_t const prf_idx,
               pareto_set<journey>& results) {
    auto const end_k = std::min(max_transfers, kMaxTransfers) + 1U;

    auto const l_worst_at_dest = mcraptor_label(unix_to_delta(base(), worst_time_at_dest));
    for (auto& dest_bag : dest_bags_) {
      if (!dest_bag.dominates_or_equals(l_worst_at_dest)) {
        dest_bag.add(l_worst_at_dest);
      }
    }

    for (auto k = 1U; k != end_k; ++k) {
      for (auto i = 0U; i != n_locations_; ++i) {
        for (auto& round_label : round_bags_[k][i]) {
          if (!best_bags_[i].dominates_or_equals(round_label)) {
            best_bags_[i].add(round_label);
          }
        }
      }

      is_dest_.for_each_set_bit([&](std::uint64_t const i) {
        for (auto& best_label : best_bags_[i]) {
          update_dest_bags(k, best_label);
        }
      });

      auto any_marked = false;
      state_.station_mark_.for_each_set_bit([&](std::uint64_t const i) {
        for (auto const& r : tt_.location_routes_[location_idx_t{i}]) {
          any_marked = true;
          state_.route_mark_.set(to_idx(r), true);
        }
      });

      if (!any_marked) {
        break;
      }

      std::swap(state_.prev_station_mark_, state_.station_mark_);
      utl::fill(state_.station_mark_.blocks_, 0U);

      any_marked = loop_routes(k);

      if (!any_marked) {
        break;
      }

      utl::fill(state_.route_mark_.blocks_, 0U);

      std::swap(state_.prev_station_mark_, state_.station_mark_);
      utl::fill(state_.station_mark_.blocks_, 0U);

      update_transfers(k);
      update_footpaths(k, prf_idx);
    }

    is_dest_.for_each_set_bit([&](auto const i) {
      for (auto k = 1U; k != end_k; ++k) {
        for (auto const& dest_label : round_bags_[k][i]) {
          auto const dest_time = dest_label.arr_t_;
          if (dest_time != kInvalid) {
            auto const [optimal, it, dominated_by] = results.add(
                journey{.legs_ = {},
                        .start_time_ = start_time,
                        .dest_time_ = delta_to_unix(base(), dest_time),
                        .dest_ = location_idx_t{i},
                        .transfers_ = static_cast<std::uint8_t>(k - 1)});
          }
        }
      }
    });
  }

  void reconstruct(query const& q, journey& j) {
    auto l = j.dest_;
    for (auto i = 0U; i <= j.transfers_; ++i) {
      auto const k = j.transfers_ + 1 - i;
      auto [fp_leg, transport_leg] = get_legs(k, l, q.prf_idx_);
      l = kFwd ? transport_leg.from_ : transport_leg.to_;
      // don't add a 0-minute footpath at the end (fwd) or beginning (bwd)
      if (i != 0 || fp_leg.from_ != fp_leg.to_ ||
          fp_leg.dep_time_ != fp_leg.arr_time_) {
        j.add(std::move(fp_leg));
      }
      j.add(std::move(transport_leg));
    }
    std::reverse(begin(j.legs_), end(j.legs_));
  }

private:
  date::sys_days base() const {
    return tt_.internal_interval_days().from_ + as_int(base_) * date::days{1};
  }

  bool loop_routes(unsigned const k) {
    auto any_marked = false;
    state_.route_mark_.for_each_set_bit([&](auto const r_idx) {
      auto const r = route_idx_t{r_idx};

      ++stats_.n_routes_visited_;
      any_marked |= update_route(k, r);
    });
    return any_marked;
  }

  void update_transfers(unsigned const k) {
    state_.prev_station_mark_.for_each_set_bit([&](auto&& i) {
      for (auto tmp_label : tmp_bags_[i]) {
        auto const is_dest = is_dest_[i];

        auto const transfer_time =
            (is_dest)
                ? 0
                : dir(adjusted_transfer_time(
                          transfer_time_settings_,
                          tt_.locations_.transfer_time_[location_idx_t{i}]
                              .count()));
        tmp_label.arr_t_ = static_cast<delta_t>(tmp_label.arr_t_ + transfer_time);

        if (!best_bags_[i].dominates_or_equals(tmp_label) &&
            !dest_bags_[k].dominates_or_equals(tmp_label)) {

          tmp_label.arr_t_ += dir(lb_[i]);

          if (lb_[i] == kUnreachable ||
              dest_bags_[k].dominates_or_equals(tmp_label)) {
            ++stats_.fp_update_prevented_by_lower_bound_;
            return;
          }
          ++stats_.n_earliest_arrival_updated_by_footpath_;

          tmp_label.arr_t_ -= dir(lb_[i]);

          round_bags_[k][i].add(tmp_label);
          best_bags_[i].add(tmp_label);
          state_.station_mark_.set(i, true);
          if (is_dest) {
            update_dest_bags(k, tmp_label);
          }
        }
      }
    });
  }

  void update_footpaths(unsigned const k, profile_idx_t const prf_idx) {
    state_.prev_station_mark_.for_each_set_bit([&](std::uint64_t const i) {
      for (auto& tmp_label : tmp_bags_[i]) {
        auto const tmp_time = tmp_label.arr_t_;
        auto const l_idx = location_idx_t{i};

        auto const& fps =tt_.locations_.footpaths_out_[prf_idx][l_idx];
        for (auto const& fp : fps) {
          ++stats_.n_footpaths_visited_;
          if (tmp_label.arr_t_ == kInvalid) {
            return;
          }

          auto const target = to_idx(fp.target());
          mcraptor_label new_round_label = tmp_label;
          new_round_label.arr_t_  = clamp(
              tmp_time +
              dir(adjusted_transfer_time(
                  transfer_time_settings_, fp.duration().count())));

          if (!best_bags_[target].dominates_or_equals(new_round_label) &&
              !dest_bags_[k].dominates_or_equals(new_round_label)) {
            auto const lower_bound = lb_[target];
            new_round_label.arr_t_ += lower_bound;

            if (lower_bound == kUnreachable ||
                dest_bags_[k].dominates_or_equals(new_round_label)) {
              ++stats_.fp_update_prevented_by_lower_bound_;
              continue;
            }

            ++stats_.n_earliest_arrival_updated_by_footpath_;

            new_round_label.arr_t_ -= lower_bound;
            round_bags_[k][target].add(new_round_label);
            best_bags_[target].add(new_round_label);
            state_.station_mark_.set(target, true);

            if (is_dest_[target]) {
              update_dest_bags(k, new_round_label);
            }
          }
        }
      }
    });
  }

  bool update_route(unsigned const k, route_idx_t const r) {
    auto const stop_seq = tt_.route_location_seq_[r];
    bool any_marked = false;
    mcraptor_bag route_bag;

    for (auto i = 0U; i != stop_seq.size(); ++i) {
      auto const stop_idx =
          static_cast<stop_idx_t>(i);
      auto const stp = stop{stop_seq[stop_idx]};
      auto const l_idx = cista::to_idx(stp.location_idx());
      auto const is_last = i == stop_seq.size() - 1U;

      route_bag.remove_invalid_trips();
      if (route_bag.empty() && !state_.prev_station_mark_[l_idx]) {
        continue;
      }

      if (!route_bag.empty() && stp.can_finish<SearchDir>(false)) {
        for (auto& route_label : route_bag) {
          route_label.arr_t_ = time_at_stop(
              r, route_label.trip_, stop_idx, event_type::kArr);
          assert(route_label.arr_t_ != std::numeric_limits<delta_t>::min() &&
                 route_label.arr_t_ != std::numeric_limits<delta_t>::max());
        }

        for (auto route_label : route_bag) {
          if (round_bags_[k - 1][l_idx].dominates_or_equals(route_label) ||
              tmp_bags_[l_idx].dominates_or_equals(route_label) ||
              best_bags_[l_idx].dominates_or_equals(route_label)) {
            continue;
          }
          route_label.arr_t_ += dir(lb_[l_idx]);

          if (lb_[l_idx] != kUnreachable &&
              !dest_bags_[k].dominates_or_equals(route_label)) {
            ++stats_.n_earliest_arrival_updated_by_route_;

            route_label.arr_t_ -= dir(lb_[l_idx]);
            route_label.fp_l_ = stp.location_idx();

            tmp_bags_[l_idx].add(route_label);
            state_.station_mark_.set(l_idx, true);
            any_marked = true;
          }
        }
      }

      if (is_last ||
          !stp.can_start<SearchDir>(false) ||
          !state_.prev_station_mark_[l_idx] ||
          round_bags_[k - 1][l_idx].empty()) {
        continue;
      }

      if (lb_[l_idx] == kUnreachable) {
        break;
      }

      for (auto& route_label : route_bag) {
        route_label.arr_t_ = time_at_stop(
            r, route_label.trip_, stop_idx, event_type::kDep);
      }

      for (auto round_label : round_bags_[k - 1][l_idx]) {
        if (!route_bag.dominates(round_label)) {
          auto const [day, mam] = split(round_label.arr_t_);
          auto const new_et = get_earliest_transport(
              k, r, stop_idx, day, mam, stp.location_idx());
          round_label.arr_t_ = new_et.is_valid()
                                   ? time_at_stop(
                                         r, new_et, stop_idx, event_type::kDep)
                                   : kInvalid;
          if (new_et.is_valid() &&
              ((best_bags_[l_idx].empty() &&
                tmp_bags_[l_idx].labels_.empty()) ||
               !route_bag.dominates(round_label))) {
            round_label.trip_ = new_et;
            round_label.trip_l_ = stp.location_idx();
            round_label.routeIdx_ = r;
            route_bag.add(round_label);
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
                                   location_idx_t const l,
                                   bool reconstruct = false) {
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

    constexpr auto const kNDaysToIterate = day_idx_t::value_t{2U};
    for (auto i = day_idx_t::value_t{0U}; i != kNDaysToIterate; ++i) {
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

        if (!reconstruct) {
          for (auto const& dest_label : dest_bags_[k]) {
            if (is_better_or_eq(dest_label.arr_t_,
                                to_delta(day, ev_mam) + dir(lb_[to_idx(l)]))) {
              return {transport_idx_t::invalid(), day_idx_t::invalid()};
            }
          }
        }

        auto const t = tt_.route_transport_ranges_[r][t_offset];
        if (i == 0U && !is_better_or_eq(mam_at_stop.count(), ev_mam)) {
          continue;
        }

        auto const ev_day_offset = ev.days();
        auto const start_day =
            static_cast<std::size_t>(as_int(day) - ev_day_offset);
        if (!is_transport_active(t, start_day)) {
          continue;
        }
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

  delta_t to_delta(day_idx_t const day, std::int16_t const mam) {
    return clamp((as_int(day) - as_int(base_)) * 1440 + mam);
  }

  unixtime_t to_unix(delta_t const t) { return delta_to_unix(base(), t); }

  std::pair<day_idx_t, minutes_after_midnight_t> split(delta_t const x) {
    return split_day_mam(base_, x);
  }

  bool is_intermodal_dest() const { return !dist_to_end_.empty(); }

  void update_dest_bags(unsigned const k, mcraptor_label const d_label) {
    if constexpr (SearchMode == search_mode::kOneToAll) {
      return;
    }
    for (auto i = k; i != dest_bags_.size(); ++i) {
      if (!dest_bags_[i].dominates_or_equals(d_label)) {
        dest_bags_[i].add(d_label);
      }
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

  std::pair<journey::leg, journey::leg> get_legs(unsigned const k,
                                                 auto const l,
                                                 auto const prf_idx) {
    assert(round_bags_[k][to_idx(l)].labels_.size() == 1);
    auto label = round_bags_[k][to_idx(l)].get(0);
    auto const& trip_dep_l_idx = label.trip_l_;
    auto const& trip_arr_fp_dep_l_idx = label.fp_l_;
    auto const& route_idx = label.routeIdx_;
    auto const& fp_arr_time = label.arr_t_;

    delta_t trip_dep_time;
    delta_t trip_arr_fp_dep_time;
    stop_idx_t trip_start_idx;
    stop_idx_t trip_dest_idx;
    transport transport;
    footpath footpath{};


    struct transport trip{};
    auto const stop_seq = tt_.route_location_seq_[route_idx];

    for (auto i = 0U; i != stop_seq.size(); ++i) {
      auto const stop_idx = static_cast<stop_idx_t>(i);
      auto const stp = stop{stop_seq[stop_idx]};
      auto const l_idx = stp.location_idx();

      if (trip.is_valid() && (trip_arr_fp_dep_l_idx.v_ == l_idx.v_)) {
        trip_dest_idx = stop_idx;
        transport = trip;
        trip_arr_fp_dep_time = time_at_stop(
            route_idx, trip, stop_idx, event_type::kArr);
        break;
      }

      if (trip_dep_l_idx.v_ == l_idx.v_) {
        trip = label.trip_;
        trip_dep_time = time_at_stop(route_idx, trip, stop_idx,event_type::kDep);
        trip_start_idx = stop_idx;
      }
    }

    if (l == trip_arr_fp_dep_l_idx) {
      footpath = {l, tt_.locations_.transfer_time_[location_idx_t{l}]};
    } else {
      auto const& fps =
          tt_.locations_.footpaths_out_[prf_idx][trip_arr_fp_dep_l_idx];
      for (auto const& fp : fps) {
        if (l == fp.target()) {
          footpath = fp;
          break;
        }
      }
    }

    auto const fp_leg =
        journey::leg{SearchDir, trip_arr_fp_dep_l_idx, l,
                     delta_to_unix(base(), trip_arr_fp_dep_time),
                     delta_to_unix(base(), fp_arr_time),
                     footpath};

    auto const transport_leg =
        journey::leg{SearchDir, trip_dep_l_idx, trip_arr_fp_dep_l_idx,
                     delta_to_unix(base(), trip_dep_time),
                     delta_to_unix(base(), trip_arr_fp_dep_time),
                     journey::run_enter_exit{
                         {.t_ = transport,
                          .stop_range_ =
                              interval<stop_idx_t>{
                                  0,
                                  static_cast<stop_idx_t>(
                                      tt_.route_location_seq_[route_idx].size())
                              }
                         },
                         trip_start_idx, trip_dest_idx}};

    return {fp_leg, transport_leg};
  }

  timetable const& tt_;
  rt_timetable const* rtt_{nullptr};
  int n_days_;
  std::uint32_t n_locations_, n_routes_, n_rt_transports_;
  raptor_state& state_;
  vector<mcraptor_bag> tmp_bags_;
  vector<mcraptor_bag> best_bags_;
  vector<vector<mcraptor_bag>> round_bags_;
  bitvec const& is_dest_;
  std::array<bitvec, kMaxVias> const& is_via_;
  std::vector<std::uint16_t> const& dist_to_end_;
  hash_map<location_idx_t, std::vector<td_offset>> const& td_dist_to_end_;
  std::vector<std::uint16_t> const& lb_;
  std::vector<via_stop> const& via_stops_;
  vector<mcraptor_bag> dest_bags_;
  day_idx_t base_;
  raptor_stats stats_;
  clasz_mask_t allowed_claszes_;
  bool require_bike_transport_;
  bool is_wheelchair_;
  transfer_time_settings transfer_time_settings_;
};

}  // namespace nigiri::routing
