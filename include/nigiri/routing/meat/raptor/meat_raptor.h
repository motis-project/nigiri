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

struct transport_data {
  transport trip_;
  meat_t meat_;
  stop_idx_t exit_stop_;
};

struct meat_raptor {
  static inline constexpr via_offset_t VIAS = 0;
  using algo_state_t = meat_raptor_state;
  using algo_stats_t = meat_raptor_stats;

  struct vias_accessor {
    auto const& operator[](auto const index) const noexcept {
      return array_[index][VIAS];
    }
    std::span<std::array<delta_t, VIAS + 1>> array_;
  };

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
        dge_{tt_, base_, state_},
        last_arr_{0} {
    // max_travel_time_ > kMaxTravelTime not supported at the moment
    assert(max_travel_time_ <= kMaxTravelTime);
  }

  algo_stats_t get_stats() const { return stats_; }

  void next_start_time() {
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

    // - Run raptor that assumes all arrivals are max delayed by
    // maxDc to determine esa(s,τ,t)
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
        nigiri::routing::raptor<search_dir, false, VIAS, search_type::kESA>;
    using algo_ea_t =
        nigiri::routing::raptor<search_dir, false, VIAS, search_type::kEA>;
    // necessary, so that a connection exist where delay_prob() returns 1
    auto constexpr extra_delay = 1;
    auto const esa_static_arr_delay = clamp(max_delay_ + extra_delay);

    UTL_START_TIMING(esa_time);
    stats_.esa_stats_ =
        nigiri::routing::search<search_dir, algo_esa_t, search_type::kESA>{
            tt_, nullptr,      state_.s_state_,      state_.r_state_,
            q,   std::nullopt, esa_static_arr_delay, base_}
            .execute();
    UTL_STOP_TIMING(esa_time);
    stats_.esa_duration_ = static_cast<std::uint64_t>(UTL_TIMING_MS(esa_time));

    auto const esa =
        state_.r_state_.get_best<VIAS>()[to_idx(target_location)][VIAS];

    static_assert(kInvalidDelta<search_dir> ==
                  std::numeric_limits<delta_t>::max());
    auto const no_safe_arr_at_dest = esa == kInvalidDelta<search_dir> ||
                                     (esa - s_time) > max_travel_time_.count();
    if (no_safe_arr_at_dest) {
      result_graph =
          decision_graph{{{start_location, {}, {}}, {target_location, {}, {}}},
                         {},
                         dg_node_idx_t{0},
                         dg_node_idx_t{1},
                         dg_arc_idx_t::invalid()};
      UTL_STOP_TIMING(total_time);
      stats_.total_duration_ =
          static_cast<std::uint64_t>(UTL_TIMING_MS(total_time));
      return;
    }

    // - τlast =τ+α·(esa(s,τ,t)−τ)
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

    UTL_START_TIMING(ea_time);
    // - (one to all raptor with τlast as target pruning value, to determine
    // all, ea (s, τ, ·))
    stats_.ea_stats_ =
        nigiri::routing::search<search_dir, algo_ea_t, search_type::kEA>{
            tt_,
            nullptr,
            state_.s_state_,
            state_.r_state_,
            std::move(q),
            std::nullopt,
            0,
            base_,
            last_arr_}
            .execute();
    UTL_STOP_TIMING(ea_time);
    stats_.ea_duration_ = static_cast<std::uint64_t>(UTL_TIMING_MS(ea_time));

    UTL_START_TIMING(meat_time);
    if (without_clasz_filter) {
      compute_profile_set<false>(target_location);
    } else {
      compute_profile_set<true>(target_location);
    }
    UTL_STOP_TIMING(meat_time);
    stats_.meat_duration_ =
        static_cast<std::uint64_t>(UTL_TIMING_MS(meat_time));

    auto const max_ride_count = std::numeric_limits<int>::max();
    auto const max_arrow_count = std::numeric_limits<int>::max();
    int max_display_delay;
    UTL_START_TIMING(extract_time);
    std::tie(result_graph, max_display_delay) =
        extract_small_sub_decision_graph(dge_, start_location, s_time,
                                         target_location, max_delay_,
                                         max_ride_count, max_arrow_count);
    stats_.meat_n_e_in_profile_ = state_.profile_set_.compute_entry_amount();
    result_graph.compute_use_probabilities(tt_, max_delay_);
    UTL_STOP_TIMING(extract_time);
    stats_.extract_graph_duration_ =
        static_cast<std::uint64_t>(UTL_TIMING_MS(extract_time));
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

  std::pair<day_idx_t, minutes_after_midnight_t> split(delta_t const x) const {
    return split_day_mam(base_, x);
  }

  delta_t tt_to_delta(day_idx_t const day, std::int16_t mam) const {
    return nigiri::tt_to_delta(base_, day, duration_t{mam});
    // return clamp((as_int(day) - as_int(base_)) * 1440 + mam);
  }

  template <bool WithClaszFilter>
  void compute_profile_set(location_idx_t target_location) {
    auto const ea = vias_accessor{state_.r_state_.get_best<VIAS>()};
    assert(ea[to_idx(target_location)] < std::numeric_limits<delta_t>::max());

    state_.station_mark_.set(to_idx(target_location), true);
    state_.fp_dis_to_target_[target_location] = 0.0;
    for (auto const& fp :
         tt_.locations_.footpaths_in_[fp_prf_idx_][target_location]) {
      if (ea[to_idx(fp.target())] > last_arr_ - fp.duration().count()) {
        continue;
      }
      state_.station_mark_.set(to_idx(fp.target()), true);
      state_.fp_dis_to_target_[fp.target()] = fp.duration().count();
    }

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

      any_marked = loop_routes<WithClaszFilter>();

      if (!any_marked) {
        break;
      }

      state_.route_mark_.zero_out();

      std::swap(state_.prev_station_mark_, state_.station_mark_);
      state_.station_mark_.zero_out();

      update_footpaths();

      std::swap(state_.latest_dep_added_last_round,
                state_.latest_dep_added_current_round);
      utl::fill(state_.latest_dep_added_current_round,
                std::numeric_limits<delta_t>::min());
    }
  }

  void update_footpaths() {
    auto const ea = vias_accessor{state_.r_state_.get_best<VIAS>()};
    state_.prev_station_mark_.for_each_set_bit([&](auto const i) {
      auto const l_idx = location_idx_t{i};
      state_.station_mark_.set(i, true);
      auto const& fps = tt_.locations_.footpaths_in_[fp_prf_idx_][l_idx];

      for (auto const& pe : state_.profile_set_.for_sorted_stop(l_idx)) {
        if (pe.dep_time_ > state_.latest_dep_added_current_round[l_idx]) {
          break;
        }
        for (auto const& fp : fps) {
          auto const fp_start_location = fp.target();
          auto const fp_dep_time = clamp(pe.dep_time_ - fp.duration().count());

          auto const faster_than_final_fp =
              pe.meat_ < state_.fp_dis_to_target_[fp_start_location] +
                             static_cast<meat_t>(fp_dep_time);
          if (fp_dep_time < ea[cista::to_idx(fp_start_location)] ||
              !faster_than_final_fp || fp_start_location == l_idx) {
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
            state_.latest_dep_added_current_round[fp_start_location] = std::max(
                state_.latest_dep_added_current_round[fp_start_location],
                fp_dep_time);
            stats_.meat_n_e_added_to_profile_++;
            stats_.meat_n_fp_added_to_profile_++;
          }
        }
      }
    });
  }

  template <bool WithClaszFilter>
  bool loop_routes() {
    auto any_marked = false;
    state_.route_mark_.for_each_set_bit([&](auto const r_idx) {
      auto const r = route_idx_t{r_idx};

      if constexpr (WithClaszFilter) {
        if (!is_allowed(allowed_claszes_, tt_.route_clasz_[r])) {
          return;
        }
      }

      //++stats_.n_routes_visited_;
      any_marked |= update_route(r);
    });
    return any_marked;
  }

  bool update_route(route_idx_t const r) {
    auto const ea = vias_accessor{state_.r_state_.get_best<VIAS>()};
    auto const stop_seq = tt_.route_location_seq_[r];
    bool any_marked = false;

    auto active_transports = std::vector<transport_data>{};
    auto outside_bounds = std::pair<transport, transport>{};
    for (auto i = 0U; i != stop_seq.size(); ++i) {
      stats_.meat_n_stops_iterated_++;
      auto const stop_idx = static_cast<stop_idx_t>(stop_seq.size() - i - 1U);
      auto const stp = stop{stop_seq[stop_idx]};
      auto const l_idx = stp.location_idx();
      auto const l_idx_v = cista::to_idx(l_idx);
      auto const is_last = i == stop_seq.size() - 1U;

      if (ea[l_idx_v] == std::numeric_limits<delta_t>::max() ||
          (active_transports.empty() && !state_.prev_station_mark_[l_idx_v])) {
        continue;
      }

      if (stp.in_allowed()) {
        auto entry_added_to_profile = false;
        stats_.meat_n_active_transports_iterated_ += active_transports.size();
        for (auto const& td : active_transports) {
          auto const dep_time =
              time_at_stop(r, td.trip_, stop_idx, event_type::kDep);
          auto const faster_than_walk =
              td.meat_ <
              state_.fp_dis_to_target_[l_idx] + static_cast<meat_t>(dep_time);
          if (!std::isfinite(td.meat_) || dep_time < ea[l_idx_v] ||
              !faster_than_walk) {
            continue;
          }
          if (state_.profile_set_.add_entry(
                  l_idx,
                  profile_entry{dep_time, td.meat_,
                                ride(td.trip_, stop_idx, td.exit_stop_)})) {
            state_.latest_dep_added_current_round[l_idx] = std::max(
                state_.latest_dep_added_current_round[l_idx], dep_time);
            entry_added_to_profile = true;
            stats_.meat_n_e_added_to_profile_++;
          }
        }
        if (entry_added_to_profile) {
          any_marked = true;
          state_.station_mark_.set(l_idx_v, true);
        }
      }

      auto const update_transports =
          !is_last && stp.out_allowed() && state_.prev_station_mark_[l_idx_v];
      if (!update_transports) {
        continue;
      }

      auto const fp_dis_to_target = state_.fp_dis_to_target_[l_idx];
      auto const range_begin = ea[l_idx_v];
      auto const range_end =
          std::isfinite(fp_dis_to_target)
              ? (state_.profile_set_.is_stop_empty(l_idx)
                     ? static_cast<delta_t>(last_arr_ - fp_dis_to_target)
                     : std::max(state_.latest_dep_added_last_round[l_idx],
                                state_.latest_dep_added_current_round[l_idx]))
              : std::max(state_.latest_dep_added_last_round[l_idx],
                         state_.latest_dep_added_current_round[l_idx]);
      assert(std::isfinite(fp_dis_to_target) ||
             !state_.profile_set_.is_stop_empty(l_idx));
      assert(range_begin <= range_end);
      get_transports_with_arr_in_range(r, stop_idx, range_begin, range_end,
                                       active_transports, outside_bounds);

      if (active_transports.empty()) {
        continue;
      }

      // if state_.prev_station_mark_[l_idx_v] && out_allowed(): check all t
      // in active_transports if the meat value can be improved
      stats_.meat_n_active_transports_iterated_ += active_transports.size();
      auto const stop_not_empty = !state_.profile_set_.is_stop_empty(l_idx);
      auto usable_trip_in_vec = false;
      for (auto& td : active_transports) {
        auto const arr_time =
            time_at_stop(r, td.trip_, stop_idx, event_type::kArr);
        if (arr_time < ea[l_idx_v]) {
          continue;
        }

        auto meat =
            fp_dis_to_target +
            static_cast<meat_t>(arr_time) /*TODO add expected value to it?*/;
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
        if (std::isfinite(td.meat_)) {
          usable_trip_in_vec = true;
        }
      }

      if (!usable_trip_in_vec) {
        active_transports.clear();
        outside_bounds = std::pair<transport, transport>{};
      }
    }
    return any_marked;
  }

  meat_t evaluate_profile(location_idx_t const stop, delta_t const when) const {
    auto meat = meat_t{0.0};
    auto assigned_prob = 0.0;
    auto transfer_time = tt_.locations_.transfer_time_[stop].count();

    auto i = state_.profile_set_.for_stop_begin(stop, when);
    while (assigned_prob < 1.0) {
      auto new_prob =
          delay_prob(clamp(i->dep_time_ - when), transfer_time, max_delay_);
      // Doesn't seem to be as stable numerically.
      // meat += (new_prob - assigned_prob) * i->meat_;
      meat += new_prob * i->meat_ - assigned_prob * i->meat_;
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
    auto const bounds_exist = !transports.empty();
    if (bounds_exist) {
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

    if (!bounds_exist) {
      transports.reserve((static_cast<size_t>(end_day.v_) -
                          static_cast<size_t>(start_day.v_) + 1) *
                         event_times.size());
    }

    auto const seek_first_first_day = [&]() {
      return linear_lb(event_times.begin(), event_times.end(), start_mam,
                       [&](delta const a, minutes_after_midnight_t const b) {
                         return a.mam() < b.count();
                       });
    };

    auto ub_was_set = false;
    for (auto day = start_day; day <= end_day; ++day) {
      if (interval_extends_start && interval_extends_end && lb_day < day &&
          day < ub_day) {
        day = ub_day;
        assert(day <= end_day);
      }
      auto ev_time_begin = event_times.begin();
      if (day == start_day) {
        ev_time_begin = seek_first_first_day();
        if (!bounds_exist || interval_extends_start) {
          if (ev_time_begin != event_times.begin()) {
            auto const lb_ev_it = ev_time_begin - 1;
            outside_old_bounds.first = transport{
                tt_.route_transport_ranges_[r][static_cast<std::size_t>(
                    &*lb_ev_it - event_times.data())],
                static_cast<day_idx_t>(as_int(start_day) - lb_ev_it->days())};
          } else {
            auto const lb_ev_it = event_times.end() - 1;
            outside_old_bounds.first =
                start_day > 0
                    ? transport{tt_.route_transport_ranges_
                                    [r][static_cast<std::size_t>(
                                        &*lb_ev_it - event_times.data())],
                                static_cast<day_idx_t>(as_int(start_day) - 1 -
                                                       lb_ev_it->days())}
                    : transport::invalid();
          }
        }
      }
      auto const ev_time_range = it_range{ev_time_begin, event_times.end()};
      if (ev_time_range.empty()) {
        continue;
      }

      for (auto it = begin(ev_time_range); it != end(ev_time_range); ++it) {
        auto const t_offset =
            static_cast<std::size_t>(&*it - event_times.data());
        auto const ev = *it;
        auto const t = tt_.route_transport_ranges_[r][t_offset];
        auto const ev_day_offset = ev.days();
        auto const t_start_day =
            static_cast<std::size_t>(as_int(day) - ev_day_offset);

        if (day == end_day && ev.mam() > end_mam.count()) {
          if (!bounds_exist || interval_extends_end) {
            outside_old_bounds.second = transport{
                t, static_cast<day_idx_t>(as_int(day) - ev_day_offset)};
            ub_was_set = true;
          }
          break;
        }

        // TODO break if in middel of not intervall
        if ((interval_extends_start && interval_extends_end &&
             tt_to_delta(lb_day, lb_mam.count()) < tt_to_delta(day, ev.mam()) &&
             tt_to_delta(day, ev.mam()) <
                 tt_to_delta(ub_day, ub_mam.count())) ||
            !is_transport_active(t, t_start_day)) {
          continue;
        }
        auto const meat = std::numeric_limits<meat_t>::infinity();
        transports.emplace_back(
            transport{t, static_cast<day_idx_t>(as_int(day) - ev_day_offset)},
            meat, stop_idx);
      }
    }

    if (!ub_was_set && (interval_extends_end || !bounds_exist)) {
      if (end_day < day_idx_t{n_days_ - 1}) {
        auto const ev_it = event_times.begin();
        auto const t_offset =
            static_cast<std::size_t>(&*ev_it - event_times.data());
        auto const t = tt_.route_transport_ranges_[r][t_offset];
        outside_old_bounds.second = transport{
            t, static_cast<day_idx_t>(as_int(end_day) + 1 - (ev_it->days()))};
      } else {
        outside_old_bounds.second = transport::invalid();
      }
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