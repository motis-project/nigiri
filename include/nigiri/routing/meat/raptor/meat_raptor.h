#pragma once

#include "utl/timing.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/clasz_mask.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/meat/decision_graph.h"
#include "nigiri/routing/meat/delay.h"
#include "nigiri/routing/meat/raptor/decision_graph_extractor.h"
#include "nigiri/routing/meat/raptor/meat_raptor_state.h"
#include "nigiri/routing/meat/raptor/meat_raptor_stats.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/search.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing::meat::raptor {

template <typename T, typename V>
struct first_dim_accessor {
  auto const& operator[](/*std::uint32_t*/ auto const index) const noexcept {
    return array_[index][vias_];
  }
  T const& array_;
  V const vias_;
};

struct transport_data {
  transport trip_;
  meat_t meat_;
  stop_idx_t exit_stop_;
};

struct meat_raptor {
  using algo_state_t = meat_raptor_state;
  using algo_stats_t = meat_raptor_stats;

  meat_raptor(timetable const& tt,
              meat_raptor_state& state,
              day_idx_t const base,
              clasz_mask_t const allowed_claszes,
              delta_t max_delay = 30,
              double bound_parameter = 1.0,
              meat_t meat_transfer_cost = 0.0,
              double fuzzy_parameter = 0.0 /*TODO not supported jet*/,
              duration_t max_travel_time = kMaxTravelTime)
      : tt_{tt},
        base_{base},
        n_days_{tt_.internal_interval_days().size().count()},
        max_delay_{max_delay},
        bound_parameter_{bound_parameter},
        meat_transfer_cost_{meat_transfer_cost},
        fuzzy_parameter_{fuzzy_parameter},
        allowed_claszes_{allowed_claszes},
        max_travel_time_{max_travel_time},
        fp_prf_idx_{0},
        state_{state.prepare_for_tt(tt_)},
        // mpc_{tt_, state_, base_, allowed_claszes_, prf_idx_, stats_},
        dge_{tt_, base_, state_},
        last_arr_{0} {
    // max_travel_time_ > kMaxTravelTime not supported at the moment
    assert(max_travel_time_ <= kMaxTravelTime);
  }

  algo_stats_t get_stats() const { return stats_; }

  void next_start_time() {
    // mpc_.reset();
    dge_.reset();
    state_.reset();
    stats_.reset();
  }

  void execute(unixtime_t const start_time,
               location_idx_t const start_location,
               location_idx_t const target_location,
               profile_idx_t const prf_idx,
               decision_graph& result_graph) {
    assert(tt_.internal_interval_days().from_ <= start_time);
    UTL_START_TIMING(total_time);
    fp_prf_idx_ = prf_idx;
    auto without_clasz_filter = allowed_claszes_ == all_clasz_allowed();
    auto s_time = to_delta(start_time);

    // - Run raptor that assumes all arrivals are max delayed (+transfer?) by
    // maxDc to determine esa(s,τ,t)
    auto constexpr vias = 0;
    auto constexpr search_dir = direction::kForward;
    auto const q =
        query{.start_time_ = start_time,
              .start_match_mode_ = nigiri::routing::location_match_mode::kExact,
              .dest_match_mode_ = nigiri::routing::location_match_mode::kExact,
              .use_start_footpaths_ = true,
              .start_ = {{start_location, 0_minutes, 0U}},
              .destination_ = {{target_location, 0_minutes, 0U}},
              .td_start_ = {},
              .td_dest_ = {},
              .max_start_offset_ = kMaxTravelTime,
              .max_transfers_ = kMaxTransfers,
              .min_connection_count_ = 0U,
              .extend_interval_earlier_ = false,
              .extend_interval_later_ = false,
              .prf_idx_ = fp_prf_idx_,
              .allowed_claszes_ = allowed_claszes_,
              .require_bike_transport_ = false,
              .transfer_time_settings_ = {},
              .via_stops_ = {}};
    using algo_esa_t =
        nigiri::routing::raptor<search_dir, false, vias, search_type::kESA>;
    using algo_ea_t =
        nigiri::routing::raptor<search_dir, false, vias, search_type::kEA>;
    auto const esa_static_arr_delay =
        max_delay_ /*TODO +1 ? wie beim meat_csa?*/;
    nigiri::routing::search<search_dir, algo_esa_t, search_type::kESA>{
        tt_, nullptr,      state_.s_state_, state_.r_state_,
        q,   std::nullopt, max_delay_,      base_}
        .execute();
    auto const best_arr =
        state_.r_state_.get_best<vias>()[to_idx(target_location)][vias];

    static_assert(kInvalidDelta<search_dir> ==
                  std::numeric_limits<delta_t>::max());
    auto const no_safe_arr_at_dest =
        best_arr == kInvalidDelta<search_dir> ||
        best_arr - s_time - esa_static_arr_delay > max_travel_time_.count();
    if (no_safe_arr_at_dest) {
      result_graph =
          decision_graph{{{start_location, {}, {}}, {target_location, {}, {}}},
                         {},
                         dg_node_idx_t{0},
                         dg_node_idx_t{1},
                         dg_arc_idx_t::invalid()};
      return;
    }

    // - τlast =τ+α·(esa(s,τ,t)−τ)
    auto const esa = clamp(best_arr - esa_static_arr_delay);
    if (bound_parameter_ == std::numeric_limits<double>::max()) {
      last_arr_ = std::numeric_limits<delta_t>::max() - 1;
    } else if (bound_parameter_ == 1.0) {
      last_arr_ = esa;
    } else {
      auto la = std::round(
          s_time + (bound_parameter_ * static_cast<double>(esa - s_time)));
      last_arr_ = static_cast<delta_t>(std::clamp(
          la, static_cast<double>(std::numeric_limits<delta_t>::min()),
          static_cast<double>(std::numeric_limits<delta_t>::max() - 1)));
    }

    // - (one to all raptor with τlast as target pruning value, to determine
    // all, ea (s, τ, ·)) ? ea
    nigiri::routing::search<search_dir, algo_ea_t, search_type::kEA>{
        tt_,          nullptr, state_.s_state_, state_.r_state_, std::move(q),
        std::nullopt, 0,       base_,           last_arr_}
        .execute();

    if (without_clasz_filter) {
      compute_profile_set<false>(start_location, target_location, s_time);
    } else {
      compute_profile_set<true>(start_location, target_location, s_time);
    }

    auto const max_ride_count = std::numeric_limits<int>::max();
    auto const max_arrow_count = std::numeric_limits<int>::max();
    int max_display_delay;
    std::tie(result_graph, max_display_delay) =
        extract_small_sub_decision_graph(dge_, start_location, s_time,
                                         target_location, max_delay_,
                                         max_ride_count, max_arrow_count);
    result_graph.compute_use_probabilities(tt_, max_delay_);

    UTL_STOP_TIMING(total_time);
    stats_.total_duration_ =
        static_cast<std::uint64_t>(UTL_TIMING_MS(total_time));
  }

private:
  int as_int(day_idx_t const d) const { return static_cast<int>(d.v_); }

  date::sys_days base() const {
    return tt_.internal_interval_days().from_ + as_int(base_) * date::days{1};
  }
  delta_t to_delta(unixtime_t const t) const {
    return unix_to_delta(base(), t);
  }

  // unixtime_t to_unix(delta_t const t) const { return delta_to_unix(base(),
  // t); }

  std::pair<day_idx_t, minutes_after_midnight_t> split(delta_t const x) const {
    return split_day_mam(base_, x);
  }

  delta_t tt_to_delta(day_idx_t const day, std::int16_t mam) const {
    return nigiri::tt_to_delta(base_, day, duration_t{mam});
    // return clamp((as_int(day) - as_int(base_)) * 1440 + mam);
  }

  // unixtime_t tt_to_unix(day_idx_t day, minutes_after_midnight_t mam) const {
  //   return tt_.to_unixtime(day, mam);
  // }
  // unixtime_t tt_to_unix(day_idx_t day, std::int16_t mam) const {
  //   return tt_.to_unixtime(day, duration_t{mam});
  // }
  // unixtime_t tt_to_unix(delta d) const {
  //   return tt_.to_unixtime(0, d.as_duration());
  // }
  //  std::pair<day_idx_t, minutes_after_midnight_t> day_idx_mam(unixtime_t
  //  const t){return tt_.day_idx_mam(t);}

  template <bool WithClaszFilter>
  void compute_profile_set(location_idx_t start_location,
                           location_idx_t target_location,
                           delta_t start_time) {
    state_.station_mark_.set(to_idx(target_location), true);
    state_.fp_dis_to_target_[target_location] = 0.0;
    for (auto const& fp :
         tt_.locations_.footpaths_in_[fp_prf_idx_][target_location]) {
      state_.station_mark_.set(to_idx(fp.target()), true);
      state_.fp_dis_to_target_[fp.target()] = fp.duration().count();
    }

    // TODO k != kMaxTransfers + 1 abändern zu true
    // for (auto k = 1U; k != kMaxTransfers + 1; ++k) {
    while (true) {
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
      state_.station_mark_.zero_out();

      any_marked =
          loop_routes<WithClaszFilter>(start_location, target_location);

      if (!any_marked) {
        break;
      }

      state_.route_mark_.zero_out();

      std::swap(state_.prev_station_mark_, state_.station_mark_);
      state_.station_mark_.zero_out();

      // update_transfers(k); do not need it
      update_footpaths(start_location, start_time);
    }
  }

  void update_footpaths(location_idx_t start_location, delta_t start_time) {
    state_.prev_station_mark_.for_each_set_bit([&](auto const i) {
      auto const l_idx = location_idx_t{i};
      if (l_idx == start_location) {
        return;
      }
      state_.station_mark_.set(i, true);
      auto const& fps = tt_.locations_.footpaths_in_[fp_prf_idx_][l_idx];

      for (auto const& pe : state_.profile_set_.for_unsorted_stop(l_idx)) {
        for (auto const& fp : fps) {
          auto const fp_start_location = fp.target();
          auto const fp_dep_time = clamp(pe.dep_time_ - fp.duration().count());

          auto const faster_than_final_fp =
              pe.meat_ < state_.fp_dis_to_target_[fp_start_location] +
                             static_cast<meat_t>(fp_dep_time);
          if (fp_dep_time < start_time || !faster_than_final_fp ||
              fp_start_location == l_idx) {
            continue;
          }

          assert(fp_start_location != l_idx &&
                 "would otherwise invalidate the iterator (pe)");
          auto const added = state_.profile_set_.add_entry(
              fp_start_location,
              profile_entry{
                  fp_dep_time, pe.meat_,
                  walk{fp_start_location, footpath{l_idx, fp.duration()}}});
          if (added) {
            state_.station_mark_.set(to_idx(fp_start_location), true);
          }
        }
      }
    });
  }

  template <bool WithClaszFilter>
  bool loop_routes(location_idx_t start_location,
                   location_idx_t target_location) {
    auto any_marked = false;
    state_.route_mark_.for_each_set_bit([&](auto const r_idx) {
      auto const r = route_idx_t{r_idx};

      if constexpr (WithClaszFilter) {
        if (!is_allowed(allowed_claszes_, tt_.route_clasz_[r])) {
          return;
        }
      }

      //++stats_.n_routes_visited_;
      any_marked |= update_route(r, start_location, target_location);
    });
    return any_marked;
  }

  bool update_route(route_idx_t const r,
                    location_idx_t start_location,
                    location_idx_t target_location) {
    // TODO remove
    (void)target_location;
    auto constexpr vias = 0U;
    auto const ea = first_dim_accessor{state_.r_state_.get_best<vias>(), vias};
    auto const stop_seq = tt_.route_location_seq_[r];
    bool any_marked = false;

    auto active_transports = std::vector<transport_data>{};
    auto outside_bounds = std::pair<transport, transport>{};
    for (auto i = 0U; i != stop_seq.size(); ++i) {
      auto const stop_idx = static_cast<stop_idx_t>(stop_seq.size() - i - 1U);
      auto const stp = stop{stop_seq[stop_idx]};
      auto const l_idx = stp.location_idx();
      auto const l_idx_v = cista::to_idx(l_idx);
      auto const is_last = i == stop_seq.size() - 1U;

      if (active_transports.empty() && !state_.prev_station_mark_[l_idx_v]) {
        continue;
      }

      // TODO remove all transports if dep_time < ea[l_idx_v], or use remove_if
      // in get_transports_with_arr_in_range, for all transports that are not in
      // the range

      if (stp.in_allowed()) {
        auto entry_added_to_profile = false;
        for (auto const& td : active_transports) {
          auto const dep_time =
              time_at_stop(r, td.trip_, stop_idx, event_type::kDep);
          auto const faster_than_walk =
              td.meat_ <
              state_.fp_dis_to_target_[l_idx] + static_cast<meat_t>(dep_time);
          if (dep_time < ea[l_idx_v] || !std::isfinite(td.meat_) ||
              !faster_than_walk) {
            // TODO remove dep_time < ea[l_idx_v], if those cases are removed
            // earlier
            // TODO if this happens: run remove_if dep_time < ea[l_idx_v] after
            // this loop (wahrscheinlich ist remove_if zu teuer)
            continue;
          }
          entry_added_to_profile |= state_.profile_set_.add_entry(
              l_idx, profile_entry{dep_time, td.meat_,
                                   ride(td.trip_, stop_idx, td.exit_stop_)});
        }
        if (entry_added_to_profile) {
          any_marked = true;
          state_.station_mark_.set(l_idx_v, true);
        } else {
          // TODO kann man hier sagen dass auch bei den nächsten
          // entry_added_to_profile==false sein wird (auser der meat wert wird
          // duchrch eine station[update_transports] verbessert)? ich galube der
          // fall ist nicht immer gegeben. active_transports.clear();
        }
      }

      if (l_idx == start_location) {
        break;
      }

      auto const update_transports =
          !is_last && stp.out_allowed() && state_.prev_station_mark_[l_idx_v];
      if (!update_transports) {
        continue;
      }

      // TODO anstelle von min(last_arr_, end_of_tt): min(last_arr_,
      // end_of_tt, (arr_time, so that the trip has a meat_ < inf))
      auto const fp_dis_to_target = state_.fp_dis_to_target_[l_idx];
      auto const range_begin = ea[l_idx_v];
      auto const range_end =
          std::isfinite(fp_dis_to_target)
              ? static_cast<delta_t>(last_arr_ - fp_dis_to_target)
              : state_.profile_set_.last_dep(l_idx).dep_time_;
      get_transports_with_arr_in_range(r, stop_idx, range_begin, range_end,
                                       active_transports, outside_bounds);

      // if state_.prev_station_mark_[l_idx_v] && out_allowed(): check all t
      // in active_transports if the meat value can be improved
      auto const stop_not_empty = !state_.profile_set_.is_stop_empty(l_idx);
      for (auto& td : active_transports) {
        auto const arr_time =
            time_at_stop(r, td.trip_, stop_idx, event_type::kArr);
        if (arr_time < ea[l_idx_v]) {
          // TODO remove, if those cases are removed earlier
          std::cout << "arr_time of transport < ea[l_idx_v]" << std::endl;
          continue;
        }
        auto meat = td.meat_;

        meat = std::min(meat,
                        fp_dis_to_target +
                            static_cast<meat_t>(
                                arr_time) /*TODO add expected value to it?*/);
        // TODO add expected value to it? add final footpath in
        // graph_extractor would have to be changed

        if (stop_not_empty) {
          meat = std::min(
              meat, evaluate_profile(l_idx, arr_time) + meat_transfer_cost_);
        }

        if (meat < td.meat_) {
          td.meat_ = meat;
          td.exit_stop_ = stop_idx;
        }
      }

      // if (lb_[l_idx_v] == kUnreachable) {
      //   break;
      // }
      //
      // auto const et_time_at_stop =
      //    et.is_valid()
      //        ? time_at_stop(r, et, stop_idx,
      //                       kFwd ? event_type::kDep : event_type::kArr)
      //        : kInvalid;
      // auto const prev_round_time = state_.round_times_[k - 1][l_idx_v];
      // assert(prev_round_time != kInvalid);
      // if (is_better_or_eq(prev_round_time, et_time_at_stop)) {
      //  auto const [day, mam] = split(prev_round_time);
      //  auto const new_et = get_earliest_transport(k, r, stop_idx, day, mam,
      //                                             l_idx);
      //  current_best =
      //      get_best(current_best, state_.best_[l_idx_v],
      //      state_.tmp_[l_idx_v]);
      //  if (new_et.is_valid() &&
      //      (current_best == kInvalid ||
      //       is_better_or_eq(
      //           time_at_stop(r, new_et, stop_idx,
      //                        kFwd ? event_type::kDep : event_type::kArr),
      //           et_time_at_stop))) {
      //    et = new_et;
      //  }
      //}
    }
    return any_marked;
  }

  // void update_meat(transport_data& td, route_idx_t const r, stop_idx_t const
  // stop_idx, location_idx_t l_idx){
  //   auto const arr_time =
  //           time_at_stop(r, td.trip_, stop_idx, event_type::kArr);
  //       auto meat = td.meat_;
  //
  //      meat = std::min(meat,
  //                      state_.fp_dis_to_target_[l_idx] +
  //                          static_cast<meat_t>(
  //                              arr_time) /*TODO add expected value to it?*/);
  //      // TODO add expected value to it? add final footpath in
  //      // graph_extractor would have to be changed
  //
  //      if (!state_.profile_set_.is_stop_empty(l_idx)) {
  //        meat = std::min(meat, evaluate_profile(l_idx, arr_time) +
  //                                  meat_transfer_cost_);
  //      }
  //
  //      if (meat < td.meat_) {
  //        td.meat_ = meat;
  //        td.exit_stop_ = stop_idx;
  //      }
  //}

  meat_t evaluate_profile(location_idx_t stop, delta_t when) {
    meat_t meat = 0.0;
    double assigned_prob = 0.0;

    auto i = state_.profile_set_.for_stop_begin(stop, when);
    while (assigned_prob < 1.0) {
      double new_prob =
          delay_prob(clamp(i->dep_time_ - when),
                     tt_.locations_.transfer_time_[stop].count(), max_delay_);
      meat += (new_prob - assigned_prob) * i->meat_;
      assigned_prob = new_prob;
      ++i;
    }
    return meat;
  }

  void get_transports_with_arr_in_range(
      route_idx_t const r,
      stop_idx_t const stop_idx,
      delta_t const range_start_time,
      delta_t const range_end_time,
      std::vector<transport_data>& transports,
      std::pair<transport, transport>& outside_old_bounds) {
    auto [start_day, start_mam] = split(range_start_time);
    auto [end_day, end_mam] = split(range_end_time);
    if (as_int(end_day) >= n_days_) {
      end_day = day_idx_t{n_days_ - 1};
      end_mam = minutes_after_midnight_t{1439};
    }
    assert(start_day <= end_day);
    assert((start_day == end_day && start_mam <= end_mam) ||
           start_day != end_day);

    bool interval_extends_start = false;
    bool interval_extends_end = false;
    auto ub_day = start_day;
    auto ub_mam = start_mam;
    auto lb_day = end_day;
    auto lb_mam = end_mam;
    if (!transports.empty()) {
      auto const [lb, ub] = outside_old_bounds;
      if (lb.is_valid()) {
        auto const arr_lb = time_at_stop(r, lb, stop_idx, event_type::kArr);
        std::tie(lb_day, lb_mam) = split(arr_lb);
        interval_extends_start = range_start_time <= arr_lb;
      }
      if (ub.is_valid()) {
        auto const arr_ub = time_at_stop(r, ub, stop_idx, event_type::kArr);
        std::tie(ub_day, ub_mam) = split(arr_ub);
        interval_extends_end = arr_ub <= tt_to_delta(end_day, end_mam.count());
      }
      if (!interval_extends_start && !interval_extends_end) {
        return;
      }
      if (!interval_extends_start) {
        start_day = ub_day;
        start_mam = ub_mam;
      }
      if (!interval_extends_end) {
        end_day = lb_day;
        end_mam = lb_mam;
      }
    }

    assert(start_day <= end_day);
    assert((start_day == end_day && start_mam <= end_mam) ||
           start_day != end_day);

    auto const event_times =
        tt_.event_times_at_stop(r, stop_idx, event_type::kArr);
    if (event_times.empty()) {
      return;
    }

    // assumes that is_transport_active will be true, most of the times
    // TODO passt dass mit der aktuellen version:
    // eher nicht, geht davon aus, dass komplettes interval noch mal hinzugefügt
    // wird, was sicher nicht der fall sein wird.
    transports.reserve(transports.size() +
                       (static_cast<size_t>(end_day.v_) -
                        static_cast<size_t>(start_day.v_) + 1) *
                           event_times.size());

    // auto const seek_first_day = [&]() {
    //   return linear_lb(event_times.begin(), event_times.end(), start_mam,
    //                    [&](delta const a, minutes_after_midnight_t const b) {
    //                      return a.mam() < b.count();
    //                    });
    // };
    //  auto const seek_last_day = [&]() {
    //    return linear_lb(event_times.begin(), event_times.end(), end_mam,
    //                     [&](delta const a, minutes_after_midnight_t const b)
    //                     {
    //                       return a.mam() <= b.count();
    //                     });
    //  };

    struct bound_info {
      transport_idx_t t_idx_;
      std::int16_t mam_;
    };
    auto low_outside_bound = bound_info{transport_idx_t::invalid(), -1};
    auto up_outside_bound = bound_info{transport_idx_t::invalid(),
                                       std::numeric_limits<int16_t>::max()};
    auto lowest_mam = bound_info{transport_idx_t::invalid(),
                                 std::numeric_limits<int16_t>::max()};
    auto highest_mam = bound_info{transport_idx_t::invalid(), -1};

    for (auto day = start_day; day <= end_day; ++day) {
      if (interval_extends_start && interval_extends_end && lb_day < day &&
          day < ub_day) {
        day = ub_day;
        assert(day <= end_day);
      }
      // auto const ev_time_range =
      //  it_range{day == start_day ? seek_first_day() : event_times.begin(),
      //           event_times.end()};
      //   it_range{day == start_day ? seek_first_day() :
      //   event_times.begin(),day
      //   == end_day ? seek_last_day() : event_times.end()};
      // if (ev_time_range.empty()) {
      //   continue;
      // }
      // for (auto it = begin(ev_time_range); it != end(ev_time_range); ++it) {
      for (auto it = event_times.begin(); it != event_times.end(); ++it) {
        auto const t_offset =
            static_cast<std::size_t>(&*it - event_times.data());
        auto const ev = *it;
        auto const t = tt_.route_transport_ranges_[r][t_offset];
        auto const ev_day_offset = ev.days();
        auto const t_start_day =
            static_cast<std::size_t>(as_int(day) - ev_day_offset);

        if (day == start_day) {
          if (ev.mam() < lowest_mam.mam_) {
            lowest_mam.mam_ = ev.mam();
            lowest_mam.t_idx_ = t;
          }
          if (ev.mam() > highest_mam.mam_) {
            highest_mam.mam_ = ev.mam();
            highest_mam.t_idx_ = t;
          }
          if (low_outside_bound.mam_ < ev.mam() &&
              ev.mam() < start_mam.count()) {
            low_outside_bound.mam_ = ev.mam();
            low_outside_bound.t_idx_ = t;
          }
          if (end_mam.count() < ev.mam() && ev.mam() < up_outside_bound.mam_) {
            up_outside_bound.mam_ = ev.mam();
            up_outside_bound.t_idx_ = t;
          }
        }

        if ((day == end_day && ev.mam() > end_mam.count()) ||
            (day == start_day && ev.mam() < start_mam.count()) ||
            (interval_extends_start && day == lb_day &&
             ev.mam() > lb_mam.count()) ||
            (interval_extends_end && day == ub_day &&
             ev.mam() < ub_mam.count()) ||
            !is_transport_active(t, t_start_day)) {
          continue;
        }
        auto const meat = std::numeric_limits<meat_t>::infinity();
        transports.emplace_back(
            transport{t, static_cast<day_idx_t>(as_int(day) - ev_day_offset)},
            meat, stop_idx);
      }
    }

    if (low_outside_bound.t_idx_ == transport_idx_t::invalid()) {
      if (start_day > 0) {
        outside_old_bounds.first = transport{highest_mam.t_idx_, start_day - 1};
      } else {
        outside_old_bounds.first = transport::invalid();
      }
    } else {
      outside_old_bounds.first = transport{low_outside_bound.t_idx_, start_day};
    }

    if (up_outside_bound.t_idx_ == transport_idx_t::invalid()) {
      if (end_day < day_idx_t{n_days_ - 1}) {
        outside_old_bounds.second = transport{lowest_mam.t_idx_, end_day + 1};
      } else {
        outside_old_bounds.second = transport::invalid();
      }
    } else {
      outside_old_bounds.second = transport{up_outside_bound.t_idx_, end_day};
    }
  }

  bool is_transport_active(transport_idx_t const t,
                           std::size_t const day) const {
    return tt_.bitfields_[tt_.transport_traffic_days_[t]].test(day);
  }

  delta_t time_at_stop(route_idx_t const r,
                       transport const t,
                       stop_idx_t const stop_idx,
                       event_type const ev_type) {
    return tt_to_delta(t.day_,
                       tt_.event_mam(r, t.t_idx_, stop_idx, ev_type).count());
  }

  timetable const& tt_;
  day_idx_t base_;
  int n_days_;
  delta_t max_delay_;
  double bound_parameter_;  // alpha
  meat_t meat_transfer_cost_;
  double fuzzy_parameter_;
  clasz_mask_t allowed_claszes_;
  duration_t max_travel_time_;
  profile_idx_t fp_prf_idx_;
  meat_raptor_state& state_;
  meat_raptor_stats stats_;
  decision_graph_extractor dge_;
  delta_t last_arr_;
};

}  // namespace nigiri::routing::meat::raptor