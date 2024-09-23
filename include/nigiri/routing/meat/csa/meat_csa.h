#pragma once

#include <cmath>

#include "utl/helpers/algorithm.h"
#include "utl/timing.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/clasz_mask.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/meat/csa/binary_search.h"
#include "nigiri/routing/meat/csa/decision_graph_extractor.h"
#include "nigiri/routing/meat/csa/meat_csa_state.h"
#include "nigiri/routing/meat/csa/meat_csa_stats.h"
#include "nigiri/routing/meat/csa/meat_profile_computer.h"
#include "nigiri/routing/meat/decision_graph.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing::meat::csa {

template <typename ProfileSet>
struct meat_csa {
  using algo_state_t = meat_csa_state<ProfileSet>;
  using algo_stats_t = meat_csa_stats;

  meat_csa(timetable const& tt,
           meat_csa_state<ProfileSet>& state,
           day_idx_t const base,
           clasz_mask_t const allowed_claszes,
           delta_t max_delay = 30,
           double bound_parameter = 1.0,
           meat_t meat_transfer_cost = 0.0,
           double fuzzy_parameter = 0.0)
      : tt_{tt},
        base_{base},
        n_days_{tt_.internal_interval_days().size().count()},
        max_delay_{max_delay},
        bound_parameter_{bound_parameter},
        meat_transfer_cost_{meat_transfer_cost},
        fuzzy_parameter_{fuzzy_parameter},
        allowed_claszes_{allowed_claszes},
        prf_idx_{0},
        state_{state.prepare_for_tt(tt_)},
        mpc_{tt_, state_, base_, allowed_claszes_, prf_idx_, stats_},
        dge_{tt_, base_, state_.profile_set_},
        last_arr_{0} {
    assert(n_days_ >= 0);
    assert(tt.day_idx(tt.internal_interval_days().from_) == 0 &&
           "first day_idx of tt is not 0");
    assert(as_int(tt.day_idx(tt.internal_interval_days().to_)) == n_days_);
  }

  algo_stats_t get_stats() const { return stats_; }

  void next_start_time() {
    mpc_.reset();
    dge_.reset();
    state_.reset();
    stats_.reset();
  }

  void execute(unixtime_t const start_time,
               location_idx_t const start_location,
               location_idx_t const end_location,
               profile_idx_t const prf_idx,
               decision_graph& result_graph) {
    UTL_START_TIMING(total_time);
    prf_idx_ = prf_idx;
    auto without_clasz_filter = allowed_claszes_ == all_clasz_allowed();
    auto s_time = to_delta(start_time);
    auto con_begin = without_clasz_filter ? first_conn_after<false>(s_time)
                                          : first_conn_after<true>(s_time);
    if (con_begin.first == day_idx_t::invalid()) {
      result_graph =
          decision_graph{{{start_location, {}, {}}, {end_location, {}, {}}},
                         {},
                         dg_node_idx_t{0},
                         dg_node_idx_t{1},
                         dg_arc_idx_t::invalid()};
      return;
    }
    auto con_end = without_clasz_filter
                       ? compute_safe_connection_end<false>(
                             con_begin, start_location, s_time, end_location)
                       : compute_safe_connection_end<true>(
                             con_begin, start_location, s_time, end_location);
    if (con_end.first == day_idx_t::invalid()) {
      result_graph =
          decision_graph{{{start_location, {}, {}}, {end_location, {}, {}}},
                         {},
                         dg_node_idx_t{0},
                         dg_node_idx_t{1},
                         dg_arc_idx_t::invalid()};
      return;
    }
    reset_csa_state();
    UTL_START_TIMING(ea_time);
    if (without_clasz_filter) {
      compute_one_to_all_earliest_arrival<false>(con_begin, con_end,
                                                 start_location, s_time);
    } else {
      compute_one_to_all_earliest_arrival<true>(con_begin, con_end,
                                                start_location, s_time);
    }
    UTL_STOP_TIMING(ea_time);
    stats_.ea_duration_ = static_cast<std::uint64_t>(UTL_TIMING_MS(ea_time));

    UTL_START_TIMING(meat_time);
    if (without_clasz_filter) {
      mpc_.template compute_profile_set<false>(
          con_begin, con_end, end_location, s_time, max_delay_,
          fuzzy_parameter_, meat_transfer_cost_, last_arr_);
    } else {
      mpc_.template compute_profile_set<true>(
          con_begin, con_end, end_location, s_time, max_delay_,
          fuzzy_parameter_, meat_transfer_cost_, last_arr_);
    }
    UTL_STOP_TIMING(meat_time);
    stats_.meat_duration_ =
        static_cast<std::uint64_t>(UTL_TIMING_MS(meat_time));

    int max_ride_count = std::numeric_limits<int>::max();
    int max_arrow_count = std::numeric_limits<int>::max();
    int max_display_delay;
    UTL_START_TIMING(extract_time);
    std::tie(result_graph, max_display_delay) =
        extract_small_sub_decision_graph<ProfileSet>(
            dge_, start_location, s_time, end_location, max_delay_,
            max_ride_count, max_arrow_count);
    result_graph.compute_use_probabilities(tt_, max_delay_);
    UTL_STOP_TIMING(extract_time);
    stats_.extract_graph_duration_ =
        static_cast<std::uint64_t>(UTL_TIMING_MS(extract_time));
    UTL_STOP_TIMING(total_time);
    stats_.total_duration_ =
        static_cast<std::uint64_t>(UTL_TIMING_MS(total_time));
  }

private:
  void reset_csa_state() {
    using c_idx_t = connection::trip_con_idx_t;
    utl::fill(state_.ea_, std::numeric_limits<delta_t>::max());
    utl::fill(state_.trip_first_con_, std::numeric_limits<c_idx_t>::max());
  }

  int as_int(day_idx_t const d) const { return static_cast<int>(d.v_); }
  int as_s_int(day_idx_t const d) const {
    static_assert(std::is_same_v<day_idx_t::value_t, std::uint16_t>,
                  "day_idx_t is not uint16_t");
    return static_cast<std::int16_t>(d.v_);
  }
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
  std::pair<day_idx_t, connection_idx_t> first_conn_after(
      delta_t time) {  // compute_conn_begin
    auto [day, mam] = split(time);

    if (as_s_int(day) >= n_days_) {
      return {day_idx_t::invalid(), connection_idx_t::invalid()};
    }

    auto it = tt_.fwd_connections_.begin();
    // assumes fist day of tt is always 0
    if (as_s_int(day) < 0) {
      day = day_idx_t{0};
    } else {
      it = binary_find_first_true(tt_.fwd_connections_.begin(),
                                  tt_.fwd_connections_.end(),
                                  [&](connection const& c) {
                                    return mam.count() <= c.dep_time_.mam();
                                  });

      if (it == tt_.fwd_connections_.end()) {
        it = tt_.fwd_connections_.begin();
        day++;
        if (as_int(day) >= n_days_) {
          return {day_idx_t::invalid(), connection_idx_t::invalid()};
        }
      }
    }

    while (
        !((WithClaszFilter
               ? is_allowed(
                     allowed_claszes_,
                     tt_.route_clasz_[tt_.transport_route_[it->transport_idx_]])
               : true) &&
          tt_.is_connection_active(*it, day))) {
      it++;
      if (it == tt_.fwd_connections_.end()) {
        it = tt_.fwd_connections_.begin();
        day++;
        if (as_int(day) >= n_days_) {
          return {day_idx_t::invalid(), connection_idx_t::invalid()};
        }
      }
    }

    return {day,
            static_cast<connection_idx_t>(it - tt_.fwd_connections_.begin())};
  }

  template <bool WithClaszFilter>
  std::pair<day_idx_t, connection_idx_t> last_conn_before(delta_t time) {
    auto [day, mam] = split(time);

    if (as_s_int(day) < 0) {
      return {day_idx_t::invalid(), connection_idx_t::invalid()};
    }

    auto it = tt_.fwd_connections_.end();
    if (as_int(day) >= n_days_) {
      day = day_idx_t{n_days_ - 1};
    } else {
      it = binary_find_first_true(tt_.fwd_connections_.begin(),
                                  tt_.fwd_connections_.end(),
                                  [&](connection const& c) {
                                    return mam.count() <= c.dep_time_.mam();
                                  });

      if (it == tt_.fwd_connections_.begin()) {
        it = tt_.fwd_connections_.end();
        --day;
        if (as_int(day) < 0) {
          return {day_idx_t::invalid(), connection_idx_t::invalid()};
        }
      }
    }
    --it;

    while (
        !((WithClaszFilter
               ? is_allowed(
                     allowed_claszes_,
                     tt_.route_clasz_[tt_.transport_route_[it->transport_idx_]])
               : true) &&
          tt_.is_connection_active(*it, day))) {
      if (it == tt_.fwd_connections_.begin()) {
        it = tt_.fwd_connections_.end();
        --day;
        if (as_int(day) < 0) {
          return {day_idx_t::invalid(), connection_idx_t::invalid()};
        }
      }
      --it;
    }

    return {day,
            static_cast<connection_idx_t>(it - tt_.fwd_connections_.begin())};
  }

  template <bool WithClaszFilter>
  std::pair<day_idx_t, connection_idx_t> compute_safe_connection_end(
      std::pair<day_idx_t, connection_idx_t> const& conn_begin,
      location_idx_t source_stop,
      delta_t source_time,
      location_idx_t target_stop) {
    UTL_START_TIMING(esa_time);
    auto& esa = state_.ea_;  // earliest safe arrival
    auto& trip_first_con = state_.trip_first_con_;
    update_arr_times(esa, source_time, source_stop,
                     stats_.esa_n_update_arr_time_);
    // TODO remove
    // auto [day, mam] = split(source_time);

    auto constexpr extra_delay = 1;
    delta_t const target_offset =
        tt_.locations_.transfer_time_[target_stop].count() + max_delay_ +
        extra_delay;

    auto conn_end = conn_begin.second;
    auto day = conn_begin.first;
    assert(as_s_int(day) >= 0);
    auto const* conn = &tt_.fwd_connections_[conn_end];
    auto conn_dep_time = tt_to_delta(day, conn->dep_time_.mam());
    while (as_int(day) < n_days_ &&
           conn_dep_time - source_time < kMaxTravelTime.count() &&
           esa[target_stop] > conn_dep_time + target_offset) {
      stats_.esa_n_connections_scanned_++;

      auto const via_trip =
          trip_first_con[conn->transport_idx_] <= conn->trip_con_idx_;
      auto const via_station =
          esa[stop{conn->dep_stop_}.location_idx()] <= conn_dep_time &&
          stop{conn->dep_stop_}.in_allowed() &&
          (WithClaszFilter
               ? is_allowed(allowed_claszes_,
                            tt_.route_clasz_
                                [tt_.transport_route_[conn->transport_idx_]])
               : true);

      if ((via_trip || via_station) && tt_.is_connection_active(*conn, day)) {
        if (!via_trip) {
          trip_first_con[conn->transport_idx_] = conn->trip_con_idx_;
        }
        if (stop{conn->arr_stop_}.out_allowed()) {
          auto const conn_arr_time =
              tt_to_delta(day + static_cast<day_idx_t>(conn->arr_time_.days() -
                                                       conn->dep_time_.days()),
                          conn->arr_time_.mam());
          auto const c_arr_stop_idx = stop{conn->arr_stop_}.location_idx();
          auto const conn_max_arr_time =
              clamp(conn_arr_time +
                    tt_.locations_.transfer_time_[c_arr_stop_idx].count() +
                    max_delay_ + extra_delay);
          update_arr_times(esa, conn_max_arr_time, c_arr_stop_idx,
                           stats_.esa_n_update_arr_time_);
        }
      }

      ++conn_end;
      if (conn_end >= tt_.fwd_connections_.size()) {
        conn_end = connection_idx_t{0};
        ++day;
      }
      conn = &tt_.fwd_connections_[conn_end];
      conn_dep_time = tt_to_delta(day, conn->dep_time_.mam());
    }
    UTL_STOP_TIMING(esa_time);
    stats_.esa_duration_ = static_cast<std::uint64_t>(UTL_TIMING_MS(esa_time));

    if (esa[target_stop] == std::numeric_limits<delta_t>::max() ||
        esa[target_stop] - source_time > kMaxTravelTime.count()) {
      return {day_idx_t::invalid(), connection_idx_t::invalid()};
    } else {
      if (bound_parameter_ == std::numeric_limits<double>::max()) {
        last_arr_ = std::numeric_limits<delta_t>::max() - 1;
        return last_conn_before<WithClaszFilter>(last_arr_);
        // return {day_idx_t{n_days_ - 1},
        //         connection_idx_t{tt_.fwd_connections_.size() - 1}};
      } else if (bound_parameter_ == 1.0) {
        last_arr_ = esa[target_stop];
        if (conn_end == connection_idx_t{0}) {
          return {day - 1, connection_idx_t{tt_.fwd_connections_.size() - 1}};
        } else {
          return {day, conn_end - 1};
        }
      } else {
        auto la = std::round(
            source_time + (bound_parameter_ *
                           static_cast<double>(esa[target_stop] -
                                               target_offset - source_time)));
        last_arr_ = static_cast<delta_t>(std::clamp(
            la, static_cast<double>(std::numeric_limits<delta_t>::min()),
            static_cast<double>(std::numeric_limits<delta_t>::max() - 1)));
        return last_conn_before<WithClaszFilter>(last_arr_);
      }
    }
  }

  void update_arr_times(vector_map<location_idx_t, delta_t>& ea,
                        delta_t const arr_time,
                        location_idx_t const arr_stop_idx,
                        std::uint64_t& n_up_arr_time) {
    if (arr_time < ea[arr_stop_idx]) {
      ++n_up_arr_time;
      ea[arr_stop_idx] = arr_time;
      for (auto const& fp :
           tt_.locations_.footpaths_out_[prf_idx_][arr_stop_idx]) {
        ea[fp.target()] =
            std::min(ea[fp.target()], clamp(arr_time + fp.duration().count()));
      }
    }
  }

  template <bool WithClaszFilter>
  void compute_one_to_all_earliest_arrival(
      std::pair<day_idx_t, connection_idx_t> const& conn_begin,
      std::pair<day_idx_t, connection_idx_t> const& conn_end,
      location_idx_t source_stop,
      delta_t source_time) {
    assert(as_s_int(conn_begin.first) >= 0);
    assert(as_s_int(conn_end.first) >= 0);
    auto& ea = state_.ea_;
    auto& trip_first_con = state_.trip_first_con_;

    update_arr_times(ea, source_time, source_stop,
                     stats_.ea_n_update_arr_time_);

    auto conn = conn_begin;
    auto& day = conn.first;
    auto& con_idx = conn.second;
    while (conn <= conn_end) {
      stats_.ea_n_connections_scanned_++;
      auto const& c = tt_.fwd_connections_[con_idx];
      auto const c_dep_time = tt_to_delta(day, c.dep_time_.mam());
      auto const via_trip = trip_first_con[c.transport_idx_] <= c.trip_con_idx_;
      auto const via_station =
          ea[stop{c.dep_stop_}.location_idx()] <= c_dep_time &&
          stop{c.dep_stop_}.in_allowed() &&
          (WithClaszFilter
               ? is_allowed(
                     allowed_claszes_,
                     tt_.route_clasz_[tt_.transport_route_[c.transport_idx_]])
               : true);

      if ((via_trip || via_station) && tt_.is_connection_active(c, day)) {
        if (!via_trip) {
          trip_first_con[c.transport_idx_] = c.trip_con_idx_;
        }
        if (stop{c.arr_stop_}.out_allowed()) {
          auto const c_arr_time =
              tt_to_delta(day + static_cast<day_idx_t>(c.arr_time_.days() -
                                                       c.dep_time_.days()),
                          c.arr_time_.mam());
          update_arr_times(ea, c_arr_time, stop{c.arr_stop_}.location_idx(),
                           stats_.ea_n_update_arr_time_);
        }
      }

      if (++con_idx >= tt_.fwd_connections_.size()) {
        ++day;
        con_idx = connection_idx_t{0};
      }
    }

    return;
  }

  timetable const& tt_;
  day_idx_t base_;
  int n_days_;
  delta_t max_delay_;
  double bound_parameter_;  // alpha
  meat_t meat_transfer_cost_;
  double fuzzy_parameter_;
  clasz_mask_t allowed_claszes_;
  profile_idx_t prf_idx_;
  meat_csa_state<ProfileSet>& state_;
  meat_profile_computer<ProfileSet> mpc_;
  decision_graph_extractor<ProfileSet> dge_;
  delta_t last_arr_;
  meat_csa_stats stats_;
};

}  // namespace nigiri::routing::meat::csa