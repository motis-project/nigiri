#pragma once

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/clasz_mask.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/meat/csa/binary_search.h"
#include "nigiri/routing/meat/csa/decision_graph_extractor.h"
#include "nigiri/routing/meat/csa/meat_profile_computer.h"
#include "nigiri/routing/meat/decision_graph.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing::meat::csa {

struct meat_csa {
  meat_csa(timetable const& tt,
           day_idx_t const base,
           clasz_mask_t const allowed_claszes,
           delta_t max_delay = 30,
           double bound_parameter = 1.0,
           double fuzzy_parameter = 0.0)
      : tt_{tt},
        end_of_tt_{tt_.internal_interval().to_},
        base_{base},
        n_days_{tt_.internal_interval_days().size().count()},
        max_delay_{max_delay},
        bound_parameter_{bound_parameter},
        fuzzy_parameter_{fuzzy_parameter},
        allowed_claszes_{allowed_claszes},
        prf_idx_{0},
        mpc_{tt_, base_, prf_idx_},
        dge_{tt_, base_} {}
  void execute(unixtime_t const start_time,
               location_idx_t const start_location,
               location_idx_t const end_location,
               // std::uint8_t const max_transfers,
               // unixtime_t const worst_time_at_dest,
               profile_idx_t const prf_idx,  // TODO footpahts
               decision_graph& result_graph) {
    prf_idx_ = prf_idx;
    auto without_clasz_filter = allowed_claszes_ == all_clasz_allowed();
    auto s_time = to_delta(start_time);
    auto con_begin = without_clasz_filter ? first_conn_after<false>(s_time)
                                          : first_conn_after<true>(s_time);
    if (con_begin.first == day_idx_t::invalid()) {
      // No decision graph exists TODO
      assert(false);
      return;
    }
    auto con_end =
        without_clasz_filter
            ? compute_safe_connection_end<false>(
                  con_begin.second, start_location, s_time, end_location)
            : compute_safe_connection_end<true>(
                  con_begin.second, start_location, s_time, end_location);
    if (con_end.first == day_idx_t::invalid()) {
      // No decision graph exists TODO
      assert(false);
      return;
    }
    auto ea = without_clasz_filter
                  ? compute_one_to_all_earliest_arrival<false>(
                        con_begin, con_end, start_location, s_time)
                  : compute_one_to_all_earliest_arrival<true>(
                        con_begin, con_end, start_location, s_time);

    if (without_clasz_filter) {
      mpc_.compute_profile_set<false>(con_begin, con_end, ea, end_location,
                                      max_delay_, fuzzy_parameter_, 1,
                                      allowed_claszes_);
    } else {
      mpc_.compute_profile_set<true>(con_begin, con_end, ea, end_location,
                                     max_delay_, fuzzy_parameter_, 1,
                                     allowed_claszes_);
    }
    // without_clasz_filter  // TODO: was ist mit transfer_cost (aktuell: 1)
    //     ? mpc_.compute_profile_set<false>(con_begin, con_end, ea,
    //     end_location,
    //                                       max_delay_, fuzzy_parameter_, 1,
    //                                       allowed_claszes_)
    //     : mpc_.compute_profile_set<true>(con_begin, con_end, ea,
    //     end_location,
    //                                      max_delay_, fuzzy_parameter_, 1,
    //                                      allowed_claszes_);
    int max_ride_count = std::numeric_limits<int>::max();
    int max_arrow_count = std::numeric_limits<int>::max();
    int max_display_delay;
    std::tie(result_graph, max_display_delay) =
        extract_small_sub_decision_graph(
            dge_, mpc_.get_profile_set(), start_location, s_time, end_location,
            max_delay_, max_ride_count, max_arrow_count);
    mpc_.reset_trip();
    mpc_.reset_stop();
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
  std::pair<day_idx_t, connection_idx_t> first_conn_after(
      delta_t time) {  // compute_conn_begin
    auto [day, mam] = split(time);

    if (day >= n_days_) {
      return {day_idx_t::invalid(), connection_idx_t::invalid()};
    }

    auto it = binary_find_first_true(
        tt_.fwd_connections_.begin(), tt_.fwd_connections_.end(),
        [&](connection const& c) { return mam.count() <= c.dep_time_.mam(); });

    if (it == tt_.fwd_connections_.end()) {
      it = tt_.fwd_connections_.begin();
      day++;
      if (day >= n_days_) {
        return {day_idx_t::invalid(), connection_idx_t::invalid()};
      }
    }
    while (
        !(tt_.is_connection_active(*it, day) &&
          (WithClaszFilter
               ? is_allowed(
                     allowed_claszes_,
                     tt_.route_clasz_[tt_.transport_route_[it->transport_idx_]])
               : true))) {
      it++;
      if (it == tt_.fwd_connections_.end()) {
        it = tt_.fwd_connections_.begin();
        day++;
        if (day >= n_days_) {
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

    if (day < 0) {
      return {day_idx_t::invalid(), connection_idx_t::invalid()};
    }

    auto it = binary_find_first_true(
        tt_.fwd_connections_.begin(), tt_.fwd_connections_.end(),
        [&](connection const& c) { return mam.count() <= c.dep_time_.mam(); });

    if (it == tt_.fwd_connections_.begin()) {
      it = tt_.fwd_connections_.end();
      --day;
      if (day < 0) {
        return {day_idx_t::invalid(), connection_idx_t::invalid()};
      }
    }
    --it;

    while (
        !(tt_.is_connection_active(*it, day) &&
          (WithClaszFilter
               ? is_allowed(
                     allowed_claszes_,
                     tt_.route_clasz_[tt_.transport_route_[it->transport_idx_]])
               : true))) {
      if (it == tt_.fwd_connections_.begin()) {
        it = tt_.fwd_connections_.end();
        --day;
        if (day < 0) {
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
      connection_idx_t conn_begin,
      location_idx_t source_stop,
      delta_t source_time,
      location_idx_t target_stop) {
    vector_map<location_idx_t, delta_t> esa(
        tt_.n_locations(),
        std::numeric_limits<delta_t>::max());  // earliest safe arrival
    vector_map<transport_idx_t, uint16_t> trip_first_con(
        tt_.n_transports(), std::numeric_limits<uint16_t>::max());
    esa[source_stop] = source_time;
    auto [day, mam] = split(source_time);

    delta_t const target_offset =
        tt_.locations_.transfer_time_[target_stop].count() +
        max_delay_;  //??? Warum auch change_time ? Da esa nur relevant ist,
                     // wenn
                     // umstige passieren, kann also schon direkt addierd werden

    connection_idx_t conn_end = conn_begin;
    connection const* conn = &tt_.fwd_connections_[conn_end];
    auto conn_dep_time = tt_to_delta(day, conn->dep_time_.mam());
    while (day < n_days_ && /* test if conn depature < min ( max travel time,
                               end of timetable TODO: n_days_ or end_of_tt_) */
           conn_dep_time - source_time < kMaxTravelTime.count() &&
           esa[target_stop] > conn_dep_time + target_offset) {
      // TODO test ist conn active befor iterate over transfers
      // if (tt_.is_connection_active(*conn,day)){}
      if (tt_.is_connection_active(*conn, day) &&
          (trip_first_con[conn->transport_idx_] <= conn->trip_con_idx_ ||
           (esa[stop{conn->dep_stop_}.location_idx()] <= conn_dep_time &&
            stop{conn->dep_stop_}.in_allowed() &&
            (WithClaszFilter
                 ? is_allowed(allowed_claszes_,
                              tt_.route_clasz_
                                  [tt_.transport_route_[conn->transport_idx_]])
                 : true)))) {
        if (trip_first_con[conn->transport_idx_] > conn->trip_con_idx_) {
          trip_first_con[conn->transport_idx_] = conn->trip_con_idx_;
        }
        if (stop{conn->arr_stop_}.out_allowed()) {
          auto const conn_arr_time = tt_to_delta(
              day + (conn->arr_time_.days() - conn->dep_time_.days()),
              conn->arr_time_.mam());
          auto const c_arr_stop_idx = stop{conn->arr_stop_}.location_idx();
          auto const conn_max_arr_time =
              clamp(conn_arr_time +
                    tt_.locations_.transfer_time_[c_arr_stop_idx].count() +
                    max_delay_);
          update_arr_times(esa, conn_max_arr_time, c_arr_stop_idx);
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

    if (esa[target_stop] == std::numeric_limits<delta_t>::max() ||
        esa[target_stop] - source_time > kMaxTravelTime.count()) {
      return {day_idx_t::invalid(), connection_idx_t::invalid()};
    } else {
      if (bound_parameter_ == std::numeric_limits<double>::max()) {
        return {day_idx_t{n_days_},
                connection_idx_t{tt_.fwd_connections_.size() - 1}};
      } else if (bound_parameter_ == 1.0) {
        if (conn_end == connection_idx_t{0}) {
          return {day - 1, connection_idx_t{tt_.fwd_connections_.size() - 1}};
        } else {
          return {day, conn_end - 1};
        }
      } else {
        return last_conn_before<WithClaszFilter>(
            source_time +
            static_cast<delta_t>(
                bound_parameter_ *
                (esa[target_stop] - target_offset -
                 source_time)));  //??? TODO Warum target_offset hier? da
                                  // sie weiter oben (Zeile 214) addierd
                                  // wurde, und hier nicht gefragt ist? sollte
                                  // da eventuell nur die transfer_time_
                                  // abgezogen werden?);
      }
    }
  }

  void update_arr_times(vector_map<location_idx_t, delta_t>& ea,
                        delta_t const arr_time,
                        location_idx_t const arr_stop_idx) {
    if (arr_time < ea[arr_stop_idx]) {
      ea[arr_stop_idx] = arr_time;
      for (auto const& fp :
           tt_.locations_.footpaths_out_[prf_idx_][arr_stop_idx]) {
        ea[fp.target()] =
            std::min(ea[fp.target()], clamp(arr_time + fp.duration().count()));
      }
    }
  }

  template <bool WithClaszFilter>
  vector_map<location_idx_t, delta_t> compute_one_to_all_earliest_arrival(
      std::pair<day_idx_t, connection_idx_t> const& conn_begin,
      std::pair<day_idx_t, connection_idx_t> const& conn_end,
      location_idx_t source_stop,
      delta_t source_time
      // location_idx_t target_stop
  ) {  // TODO: Warum keine Umsteigezeiten? Da wir auch sehr
       // knappe Verbindungen später im Graph haben wollen?
       // ich brauche trotz dem trip-aware, da sonst
       // in_allowd/out_allowd nicht funktioniren
    vector_map<location_idx_t, delta_t> ea(tt_.n_locations(),
                                           std::numeric_limits<delta_t>::max());
    vector_map<transport_idx_t, uint16_t> trip_first_con(
        tt_.n_transports(), std::numeric_limits<uint16_t>::max());

    ea[source_stop] = source_time;

    auto conn = conn_begin;
    auto& day = conn.first;
    auto& con_idx = conn.second;
    while (conn <= conn_end) {
      auto const& c = tt_.fwd_connections_[con_idx];
      auto const c_dep_time = tt_to_delta(day, c.dep_time_.mam());
      if (tt_.is_connection_active(c, day) &&
          (trip_first_con[c.transport_idx_] <= c.trip_con_idx_ ||
           (ea[stop{c.dep_stop_}.location_idx()] <= c_dep_time &&
            stop{c.dep_stop_}.in_allowed() &&
            (WithClaszFilter
                 ? is_allowed(
                       allowed_claszes_,
                       tt_.route_clasz_[tt_.transport_route_[c.transport_idx_]])
                 : true)))) {
        if (trip_first_con[c.transport_idx_] > c.trip_con_idx_) {
          trip_first_con[c.transport_idx_] = c.trip_con_idx_;
        }
        if (stop{c.arr_stop_}.out_allowed()) {
          auto const c_arr_time =
              tt_to_delta(day + (c.arr_time_.days() - c.dep_time_.days()),
                          c.arr_time_.mam());
          update_arr_times(ea, c_arr_time, stop{c.arr_stop_}.location_idx());
        }
      }

      if (++con_idx >= tt_.fwd_connections_.size()) {
        ++day;
        con_idx = connection_idx_t{0};
      }
    }

    return ea;
  }

  timetable const& tt_;
  unixtime_t end_of_tt_;  // TODO entfernen?
  day_idx_t base_;
  int n_days_;
  delta_t max_delay_;
  double bound_parameter_;  // alpha
  double fuzzy_parameter_;
  clasz_mask_t allowed_claszes_;
  profile_idx_t prf_idx_;
  meat_profile_computer mpc_;
  decision_graph_extractor dge_;
};

}  // namespace nigiri::routing::meat::csa