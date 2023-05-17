#include "nigiri/routing/raptor.h"

#include <algorithm>

#include "fmt/core.h"

#include "utl/enumerate.h"
#include "utl/erase_if.h"
#include "utl/helpers/algorithm.h"
#include "utl/overloaded.h"

#include "nigiri/routing/journey.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/raptor_state.h"
#include "nigiri/routing/reconstruct.h"
#include "nigiri/routing/start_times.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

template <direction SearchDir, typename T>
auto get_begin_it(T const& t) {
  if constexpr (SearchDir == direction::kForward) {
    return t.begin();
  } else {
    return t.rbegin();
  }
}

template <direction SearchDir, typename T>
auto get_end_it(T const& t) {
  if constexpr (SearchDir == direction::kForward) {
    return t.end();
  } else {
    return t.rend();
  }
}

template <direction SearchDir, bool IntermodalTarget>
raptor<SearchDir, IntermodalTarget>::raptor(timetable const& tt,
                                            search_state& search_state,
                                            raptor_state& state,
                                            query q)
    : parent{tt, search_state, q}, state_{state} {
  n_days_ = static_cast<std::uint16_t>(
      this->tt().internal_interval_days().size().count());
}

template <direction SearchDir, bool IntermodalTarget>
bool raptor<SearchDir, IntermodalTarget>::is_better(auto a, auto b) {
  return kFwd ? a < b : a > b;
}

template <direction SearchDir, bool IntermodalTarget>
bool raptor<SearchDir, IntermodalTarget>::is_better_or_eq(auto a, auto b) {
  return kFwd ? a <= b : a >= b;
}

template <direction SearchDir, bool IntermodalTarget>
auto raptor<SearchDir, IntermodalTarget>::get_best(auto a, auto b) {
  return is_better(a, b) ? a : b;
}

template <direction SearchDir, bool IntermodalTarget>
stats const& raptor<SearchDir, IntermodalTarget>::get_stats() const {
  return stats_;
}

template <direction SearchDir, bool IntermodalTarget>
routing_time raptor<SearchDir, IntermodalTarget>::time_at_stop(
    route_idx_t const r,
    transport const t,
    unsigned const stop_idx,
    event_type const ev_type) {
  return {t.day_, tt().event_mam(r, t.t_idx_, stop_idx, ev_type)};
}

template <direction SearchDir, bool IntermodalTarget>
routing_time raptor<SearchDir, IntermodalTarget>::time_at_stop(
    transport const t, unsigned const stop_idx, event_type const ev_type) {
  return {t.day_, tt().event_mam(t.t_idx_, stop_idx, ev_type)};
}

template <direction SearchDir, bool IntermodalTarget>
transport raptor<SearchDir, IntermodalTarget>::get_earliest_transport(
    unsigned const k,
    route_idx_t const r,
    unsigned const stop_idx,
    location_idx_t const l_idx) {
  NIGIRI_COUNT(n_earliest_trip_calls_);

  auto const time = state_.round_times_[k - 1][to_idx(l_idx)];
  auto const [day_at_stop, mam_at_stop] = time.day_idx_mam();

  auto const n_days_to_iterate =
      std::min(kMaxTravelTime.count() / 1440U + 1,
               kFwd ? n_days_ - to_idx(day_at_stop) : to_idx(day_at_stop) + 1U);

  trace(
      "┊ │k={}    et: current_best_at_stop={}, stop_idx={}, "
      "location=(name={}, id={}, idx={}), n_days_to_iterate={}\n",
      k, time, stop_idx, tt().locations_.names_[l_idx].view(),
      tt().locations_.ids_[l_idx].view(), l_idx, n_days_to_iterate);

  auto const event_times = tt().event_times_at_stop(
      r, stop_idx, kFwd ? event_type::kDep : event_type::kArr);

#if defined(NIGIRI_ROUTING_TRACING) && \
    !defined(NIGIRI_ROUTING_TRACING_ONLY_UPDATES)
  for (auto const [t_offset, x] : utl::enumerate(event_times)) {
    auto const t = tt().route_transport_ranges_[r][t_offset];
    trace("┊ │k={}        event_times: transport={}, name={}: {} at {}: {}\n",
          k, t, transport_name(t), kFwd ? "dep" : "arr", location{tt(), l_idx},
          x);
  }
#endif

  auto const seek_first_day = [&, mam_at_stop = mam_at_stop]() {
    return std::lower_bound(
        get_begin_it<SearchDir>(event_times),
        get_end_it<SearchDir>(event_times), mam_at_stop,
        [&](auto&& a, auto&& b) { return is_better(a, b); });
  };

  for (auto i = day_idx_t::value_t{0U}; i != n_days_to_iterate; ++i) {
    auto const day = kFwd ? day_at_stop + i : day_at_stop - i;
    auto const ev_time_range =
        it_range{kFwd /* TODO(felix) make it work for bwd */ && i == 0U
                     ? seek_first_day()
                     : get_begin_it<SearchDir>(event_times),
                 get_end_it<SearchDir>(event_times)};
    if (ev_time_range.empty()) {
      trace("┊ │k={}      day={}/{} -> nothing found -> continue\n", k, i, day);
      continue;
    }

    trace(
        "┊ │k={}      day={}/{}, ev_time_range.size() = {}, "
        "transport_range={}, first={}, last={}\n",
        k, i, day, ev_time_range.size(), tt().route_transport_ranges_[r],
        *ev_time_range.begin(), *(ev_time_range.end() - 1));
    for (auto it = begin(ev_time_range); it != end(ev_time_range); ++it) {
      auto const t_offset = static_cast<std::size_t>(&*it - event_times.data());
      auto const ev = *it;
      trace("┊ │k={}      => t_offset={}\n", k, t_offset);
      auto const ev_mam = minutes_after_midnight_t{
          ev.count() < 1440 ? ev.count() : ev.count() % 1440};
      if (is_better_or_eq(time_at_destination_, routing_time{day, ev_mam})) {
        trace(
            "┊ │k={}      => transport={}, name={}, day={}/{}, best_mam={}, "
            "transport_mam={}, transport_time={} => TIME AT DEST {} IS "
            "BETTER!\n",
            k, tt().route_transport_ranges_[r][t_offset],
            transport_name(tt().route_transport_ranges_[r][t_offset]), i, day,
            mam_at_stop, ev_mam, routing_time{day, ev_mam},
            time_at_destination_);
        return {transport_idx_t::invalid(), day_idx_t::invalid()};
      }

      auto const t = tt().route_transport_ranges_[r][t_offset];
      if (day == day_at_stop && !is_better_or_eq(mam_at_stop, ev_mam)) {
        trace(
            "┊ │k={}      => transport={}, name={}, day={}/{}, best_mam={}, "
            "transport_mam={}, transport_time={} => NO REACH!\n",
            k, t, transport_name(t), i, day, mam_at_stop, ev_mam, ev);
        continue;
      }

      auto const ev_day_offset = static_cast<day_idx_t::value_t>(
          ev.count() < 1440
              ? 0
              : static_cast<cista::base_t<day_idx_t>>(ev.count() / 1440));
      if (!tt().bitfields_[tt().transport_traffic_days_[t]].test(
              static_cast<std::size_t>(to_idx(day) - ev_day_offset))) {
        trace(
            "┊ │k={}      => transport={}, name={}, day={}/{}, "
            "ev_day_offset={}, "
            "best_mam={}, "
            "transport_mam={}, transport_time={} => NO TRAFFIC!\n",
            k, t, transport_name(t), i, day, ev_day_offset, mam_at_stop, ev_mam,
            ev);
        continue;
      }

      trace(
          "┊ │k={}      => ET FOUND: transport={}, name={} at day {} "
          "(day_offset={}) - ev_mam={}, ev_time={}, ev={}\n",
          k, t, transport_name(t), day, ev_day_offset, ev_mam, ev,
          routing_time{day_idx_t{day - ev_day_offset},
                       minutes_after_midnight_t{ev_mam}});
      return {t, static_cast<day_idx_t>(day - ev_day_offset)};
    }
  }
  trace("┊ │k={}    => et: NOT FOUND\n", k);
  return {transport_idx_t::invalid(), day_idx_t::invalid()};
}

template <direction SearchDir, bool IntermodalTarget>
bool raptor<SearchDir, IntermodalTarget>::update_route(unsigned const k,
                                                       route_idx_t const r) {
  auto const stop_seq = tt().route_location_seq_[r];
  bool any_marked = false;

  auto et = transport{};
  for (auto i = 0U; i != stop_seq.size(); ++i) {
    auto const stop_idx =
        static_cast<unsigned>(kFwd ? i : stop_seq.size() - i - 1U);
    auto const stop = timetable::stop{stop_seq[stop_idx]};
    auto const l_idx = cista::to_idx(stop.location_idx());

    if (!et.is_valid() && !state_.prev_station_mark_[l_idx]) {
      trace("┊ │k={}  stop_idx={} ({}): not marked, no et - skip\n", k,
            stop_idx, location{tt(), location_idx_t{l_idx}});
      continue;
    }

    auto current_best =
        get_best(state_.best_[l_idx], state_.round_times_[k][l_idx]);
    auto const transfer_time_offset =
        (kFwd ? 1 : -1) * tt().locations_.transfer_time_[location_idx_t{l_idx}];

    trace(
        "┊ │k={}  stop_idx={}, location=(name={}, id={}, idx={}): "
        "current_best={}, search_dir={}\n",
        k, stop_idx, tt().locations_.names_[location_idx_t{l_idx}].view(),
        tt().locations_.ids_[location_idx_t{l_idx}].view(), l_idx,
        get_best(state_.best_[l_idx], state_.round_times_[k][l_idx]),
        kFwd ? "FWD" : "BWD");

    if (et.is_valid()) {
      auto const is_destination =
          get_search().search_state_.is_destination_[l_idx];
      auto const by_transport_time = time_at_stop(
          r, et, stop_idx, kFwd ? event_type::kArr : event_type::kDep);
      auto const by_transport_time_with_transfer =
          by_transport_time + ((is_destination && !IntermodalTarget ? 0U : 1U) *
                               transfer_time_offset);

      if ((kFwd ? stop.out_allowed() : stop.in_allowed()) &&
          is_better(
              k == 1 ? by_transport_time : by_transport_time_with_transfer,
              state_.best_[l_idx]) &&
          is_better(by_transport_time, time_at_destination_)) {

#ifdef NIGIRI_LOWER_BOUND
        auto const lower_bound =
            get_search().search_state_.travel_time_lower_bound_[to_idx(
                stop.location_idx())];
        if (lower_bound.count() ==
                std::numeric_limits<duration_t::rep>::max() ||
            !is_better(by_transport_time /* _with_transfer TODO try */ +
                           (kFwd ? 1 : -1) * lower_bound,
                       time_at_destination_)) {

#ifdef NIGIRI_ROUTING_TRACING
          auto const trip_idx =
              tt().merged_trips_[tt().transport_to_trip_section_[et.t_idx_]
                                     .front()]
                  .front();
          trace_upd(
              "┊ │k={}    *** LB NO UPD: transport={}, name={}, debug={}:{}, "
              "time_by_transport={} BETTER THAN current_best={} => (name={}, "
              "id={}) - LB={}, LB_AT_DEST={} (unreachable={})!\n",
              k, et, tt().trip_display_names_[trip_idx].view(),
              tt().source_file_names_
                  [tt().trip_debug_[trip_idx].front().source_file_idx_]
                      .view(),
              tt().trip_debug_[trip_idx].front().line_number_from_,
              by_transport_time, current_best,
              tt().locations_.names_[location_idx_t{l_idx}].view(),
              tt().locations_.ids_[location_idx_t{l_idx}].view(), lower_bound,
              by_transport_time + (kFwd ? 1 : -1) * lower_bound,
              lower_bound.count() ==
                  std::numeric_limits<duration_t::rep>::max());
#endif

          NIGIRI_COUNT(route_update_prevented_by_lower_bound_);

          if (state_.prev_station_mark_[l_idx]) {
            goto update_et;
          }
          continue;
        }
#endif

#ifdef NIGIRI_ROUTING_TRACING
        auto const trip_idx =
            tt().merged_trips_[tt().transport_to_trip_section_[et.t_idx_]
                                   .front()]
                .front();
        trace_upd(
            "┊ │k={}    transport={}, name={}, debug={}:{}, "
            "time_by_transport={}, time_by_transport_with_transfer={} BETTER "
            "THAN current_best={} => update, {} marking station {}!\n",
            k, et, tt().trip_display_names_[trip_idx].view(),
            tt().source_file_names_
                [tt().trip_debug_[trip_idx].front().source_file_idx_]
                    .view(),
            tt().trip_debug_[trip_idx].front().line_number_from_,
            by_transport_time, by_transport_time_with_transfer, current_best,
            !is_better(by_transport_time_with_transfer, current_best) ? "NOT"
                                                                      : "",
            location{tt(), stop.location_idx()});
#endif

        NIGIRI_COUNT(n_earliest_arrival_updated_by_route_);
        if (k != 1 ||
            is_better(by_transport_time_with_transfer, state_.best_[l_idx]) ||
            (!state_.station_mark_[l_idx] &&
             is_better(by_transport_time, state_.best_[l_idx]))) {
          state_.best_[l_idx] = by_transport_time_with_transfer;
        }

        if (is_better(by_transport_time_with_transfer, current_best) ||
            (!state_.station_mark_[l_idx] &&
             is_better(by_transport_time, current_best))) {
          any_marked = true;
          state_.station_mark_[l_idx] = true;
          state_.round_times_[k][l_idx] = by_transport_time_with_transfer;
          trace_upd("┊ │k={}    TRANSPORT UPDATE: {} -> {}!\n", k,
                    location{tt(), stop.location_idx()},
                    by_transport_time_with_transfer);
        }

        if constexpr (!IntermodalTarget) {
          if (is_destination) {
            time_at_destination_ =
                get_best(by_transport_time_with_transfer, time_at_destination_);
          }
        }
        current_best = by_transport_time_with_transfer;
      } else {
        trace(
            "┊ │k={}    by_transport={} NOT better than time_at_destination={} "
            "OR current_best={} => no update\n",
            k, by_transport_time, time_at_destination_, current_best);
      }
    }

  update_et:
    if (!state_.prev_station_mark_[l_idx]) {
      trace("┊ │k={}    {} not marked, skipping et update\n", k,
            location{tt(), location_idx_t{l_idx}});
      continue;
    }

    if (i != stop_seq.size() - 1U) {
      auto const et_time_at_stop =
          et.is_valid()
              ? time_at_stop(r, et, stop_idx,
                             kFwd ? event_type::kDep : event_type::kArr)
              : kInvalidTime<SearchDir>;
      if (!(kFwd && (stop_idx == stop_seq.size() - 1 || !stop.in_allowed())) &&
          !(kBwd && (stop_idx == 0 || !stop.out_allowed())) &&
          is_better_or_eq(state_.round_times_[k - 1][l_idx], et_time_at_stop)) {
        trace(
            "┊ │k={}    update et: stop_idx={}, et_valid={}, stop_time={}, "
            "transport_time={}\n",
            k, stop_idx, et.is_valid(), state_.round_times_[k - 1][l_idx],
            et.is_valid()
                ? time_at_stop(et, stop_idx,
                               kFwd ? event_type::kDep : event_type::kArr)
                : kInvalidTime<SearchDir>);
        auto const new_et =
            get_earliest_transport(k, r, stop_idx, location_idx_t{l_idx});
        if (new_et.is_valid() &&
            (current_best == kInvalidTime<SearchDir> ||
             is_better_or_eq(
                 time_at_stop(r, new_et, stop_idx,
                              kFwd ? event_type::kDep : event_type::kArr),
                 et_time_at_stop))) {
          et = new_et;
        } else if (new_et.is_valid()) {
          trace(
              "┊ │k={}    update et: no update time_at_stop_with_transfer={}, "
              "et_time_at_stop={}\n",
              k,
              time_at_stop(r, new_et, stop_idx,
                           kFwd ? event_type::kDep : event_type::kArr) +
                  transfer_time_offset,
              et_time_at_stop);
        }
      } else {
        trace(
            "┊ │k={},    no et update at {}: in_allowed={}, out_allowed={}, "
            "current_best={}, et_time_at_stop={}\n",
            k, location{tt(), location_idx_t{l_idx}}, stop.in_allowed(),
            stop.out_allowed(), current_best, et_time_at_stop);
      }
    } else {
      trace("┊ │k={}    {} last stop, skipping et update\n", k,
            location{tt(), location_idx_t{l_idx}});
    }
  }
  return any_marked;
}

template <direction SearchDir, bool IntermodalTarget>
void raptor<SearchDir, IntermodalTarget>::update_footpaths(unsigned const k) {
  trace_always("┊ ├k={} FOOTPATHS\n", k);

  state_.transport_station_mark_ = state_.station_mark_;

  auto const update_intermodal_dest = [&](location_idx_t const l_idx,
                                          auto&& get_time) {
    if constexpr (IntermodalTarget) {
      if (get_search().search_state_.is_destination_[to_idx(l_idx)]) {
        trace_upd("┊ ├k={}    INTERMODAL TARGET STATION {}\n", k,
                  location{tt(), l_idx});
        for (auto const& o : get_search().q_.destinations_[0]) {
          if (!matches(tt(), location_match_mode::kIntermodal, o.target_,
                       l_idx)) {
            continue;
          }

          auto const intermodal_target_time =
              get_time() + ((kFwd ? 1 : -1) * o.duration_);
          if (is_better(intermodal_target_time, time_at_destination_)) {
            trace_upd(
                "┊ ├k={}      intermodal: {}, best={} --{}--> DEST, best={} "
                "--> update => new_time_at_dest={}\n",
                k, location{tt(), l_idx}, get_time(), o.duration_,
                time_at_destination_, intermodal_target_time);

            state_.round_times_[k][to_idx(get_special_station(
                special_station::kEnd))] = intermodal_target_time;
            time_at_destination_ = intermodal_target_time;
          }
        }
      }
    }
  };

  for (auto l_idx = location_idx_t{0U}; l_idx != tt().n_locations(); ++l_idx) {
    if (!state_.transport_station_mark_[to_idx(l_idx)]) {
      continue;
    }

    update_intermodal_dest(l_idx, [&]() {
      return state_.best_[to_idx(l_idx)] -
             (kFwd ? 1 : -1) * tt().locations_.transfer_time_[l_idx];
    });

    auto const fps = kFwd ? tt().locations_.footpaths_out_[l_idx]
                          : tt().locations_.footpaths_in_[l_idx];
    trace("┊ ├k={} updating footpaths of {}\n", k, location{tt(), l_idx});
    for (auto const& fp : fps) {
      NIGIRI_COUNT(n_footpaths_visited_);

      auto const target = to_idx(fp.target_);
      auto const min =
          get_best(state_.best_[target], state_.round_times_[k][target]);
      auto const fp_target_time =
          state_.best_[to_idx(l_idx)]  //
          + ((kFwd ? 1 : -1) * fp.duration_)  //
          - ((kFwd ? 1 : -1) * tt().locations_.transfer_time_[l_idx]);

      if (is_better(fp_target_time, min) &&
          is_better(fp_target_time, time_at_destination_)) {

#ifdef NIGIRI_LOWER_BOUND
        auto const lower_bound =
            get_search()
                .search_state_.travel_time_lower_bound_[to_idx(fp.target_)];
        if (lower_bound.count() ==
                std::numeric_limits<duration_t::rep>::max() ||
            !is_better(fp_target_time + (kFwd ? 1 : -1) * lower_bound,
                       time_at_destination_)) {

          trace_upd(
              "┊ ├k={} *** LB NO UPD: (name={}, id={}, best={}) --{}--> "
              "(name={}, id={}, best={}) --> update => {}, LB={}, AT_DEST={}\n",
              k, tt().locations_.names_[l_idx].view(),
              tt().locations_.ids_[l_idx].view(),
              state_.round_times_[k][to_idx(l_idx)], fp.duration_,
              tt().locations_.names_[fp.target_].view(),
              tt().locations_.ids_[fp.target_].view(),
              state_.round_times_[k][to_idx(fp.target_)], fp_target_time,
              lower_bound, fp_target_time + (kFwd ? 1 : -1) * lower_bound);

          NIGIRI_COUNT(fp_update_prevented_by_lower_bound_);
          continue;
        }
#endif

        trace_upd(
            "┊ ├k={}   footpath: ({}, best={}) --[{}-{}]--> ({}, best={}) --> "
            "update => {}\n",
            k, location{tt(), l_idx}, state_.round_times_[k][to_idx(l_idx)],
            fp.duration_, tt().locations_.transfer_time_[l_idx],
            location{tt(), fp.target_},
            state_.round_times_[k][to_idx(fp.target_)], fp_target_time);

        NIGIRI_COUNT(n_earliest_arrival_updated_by_footpath_);
        state_.round_times_[k][to_idx(fp.target_)] = fp_target_time;
        state_.station_mark_[to_idx(fp.target_)] = true;

        // Consequence of adding the transfer time at train arrival and
        // subtracting it again from the footpaths:
        //   - READ FROM: best[l_idx] + fp.duration
        //   - WRITE TO: round_times[k]
        //
        // Do NOT update best[fp.target]. Otherwise, we read from where we
        // write. This in combination with the fact that we subtract the
        // transfer time from the footpath time leads to subtracting it 2x,
        // which is wrong.

        update_intermodal_dest(fp.target_, [&]() { return fp_target_time; });

        if constexpr (!IntermodalTarget) {
          if (get_search().search_state_.is_destination_[to_idx(fp.target_)]) {
            time_at_destination_ =
                get_best(time_at_destination_, fp_target_time);
            trace_always("k={} new_time_at_dest = min(old={}, new={})\n", k,
                         time_at_destination_, fp_target_time);
          }
        }
      } else {
        trace(
            "┊ ├k={}   NO FP UPDATE: {} [best={}] --({} - {})--> {} [best={}, "
            "time_at_dest={}]\n",
            k, location{tt(), l_idx}, state_.best_[to_idx(l_idx)], fp.duration_,
            tt().locations_.transfer_time_[l_idx], location{tt(), fp.target_},
            min, time_at_destination_);
      }
    }
  }
}

template <direction SearchDir, bool IntermodalTarget>
unsigned raptor<SearchDir, IntermodalTarget>::end_k() const {
  return std::min(kMaxTransfers, get_search().q_.max_transfers_) + 1U;
}

template <direction SearchDir, bool IntermodalTarget>
void raptor<SearchDir, IntermodalTarget>::rounds() {
  print_state();

  for (auto k = 1U; k != end_k(); ++k) {
    trace_always("┊ round k={}\n", k);

    set_time_at_destination(k);

    auto any_marked = false;
    for (auto l_idx = location_idx_t{0U};
         l_idx != static_cast<cista::base_t<location_idx_t>>(
                      state_.station_mark_.size());
         ++l_idx) {
      if (state_.station_mark_[to_idx(l_idx)]) {
        any_marked = true;
        for (auto const& r : tt().location_routes_[l_idx]) {
          state_.route_mark_[to_idx(r)] = true;
        }
      }
    }

    std::swap(state_.prev_station_mark_, state_.station_mark_);
    utl::fill(state_.station_mark_, false);

    if (!any_marked) {
      trace_always("┊ ╰ k={}, no routes marked, exit\n\n", k);
      return;
    }

    any_marked = false;
    for (auto r_id = 0U; r_id != tt().n_routes(); ++r_id) {
      if (!state_.route_mark_[r_id]) {
        continue;
      }
      trace("┊ ├k={} updating route {}\n", k, r_id);

      NIGIRI_COUNT(n_routes_visited_);
      any_marked |= update_route(k, route_idx_t{r_id});
    }

    utl::fill(state_.route_mark_, false);
    if (!any_marked) {
      trace_always("┊ ╰ k={}, no stations marked, exit\n\n", k);
      return;
    }

    update_footpaths(k);

    trace_always("┊ ╰ round {} done\n", k);
    print_state();
  }
}

template <direction SearchDir, bool IntermodalTarget>
void raptor<SearchDir, IntermodalTarget>::force_print_state(
    char const* comment) {
  auto const empty_rounds = [&](std::uint32_t const l) {
    for (auto k = 0U; k != end_k(); ++k) {
      if (state_.round_times_[k][l] != kInvalidTime<SearchDir>) {
        return false;
      }
    }
    return true;
  };

  fmt::print("INFO: {}, time_at_destination={}\n", comment,
             time_at_destination_);
  for (auto l = 0U; l != tt().n_locations(); ++l) {
    if (!is_special(location_idx_t{l}) &&
        !get_search().search_state_.is_destination_[l] &&
        state_.best_[l] == kInvalidTime<SearchDir> && empty_rounds(l)) {
      continue;
    }

    std::string_view name, track;
    auto const type = tt().locations_.types_.at(location_idx_t{l});
    if (type == location_type::kTrack ||
        type == location_type::kGeneratedTrack) {
      name =
          tt().locations_.names_.at(tt().locations_.parents_[location_idx_t{l}])
              .view();
      track = tt().locations_.names_.at(location_idx_t{l}).view();
    } else {
      name = tt().locations_.names_.at(location_idx_t{l}).view();
      track = "---";
    }
    tt().locations_.names_[location_idx_t{l}].view();
    auto const id = tt().locations_.ids_[location_idx_t{l}].view();
    fmt::print(
        "[{}] {:12} [name={:48}, track={:10}, id={:24}]: ",
        get_search().search_state_.is_destination_[l] ? "X" : "_", l, name,
        track,
        id.substr(0, std::min(std::string_view ::size_type{24U}, id.size())));
    auto const b = state_.best_[l];
    if (b == kInvalidTime<SearchDir>) {
      fmt::print("best=_________, round_times: ");
    } else {
      fmt::print("best={:9}, round_times: ", b);
    }
    for (auto i = 0U; i != kMaxTransfers + 1U; ++i) {
      auto const t = state_.round_times_[i][l];
      if (t != kInvalidTime<SearchDir>) {
        fmt::print("{:9} ", t);
      } else {
        fmt::print("_________ ");
      }
    }
    fmt::print("\n");
  }
}

#ifdef NIGIRI_ROUTING_TRACING
template <direction SearchDir, bool IntermodalTarget>
void raptor<SearchDir, IntermodalTarget>::print_state(char const* comment) {
  force_print_state(comment);
}
#else
template <direction SearchDir, bool IntermodalTarget>
void raptor<SearchDir, IntermodalTarget>::print_state(char const*) {}
#endif

template <direction SearchDir, bool IntermodalTarget>
void raptor<SearchDir, IntermodalTarget>::set_time_at_destination(
    unsigned const k) {
  if (IntermodalTarget) {
    constexpr auto const kDestIdx =
        to_idx(get_special_station(special_station::kEnd));
    trace_always(
        "updating new_time_at_destination= min(round_time={}, best={}, "
        "time_at_dest={})",
        state_.round_times_[k][kDestIdx], state_.best_[kDestIdx],
        time_at_destination_);

    time_at_destination_ =
        get_best(state_.round_times_[k][kDestIdx],
                 get_best(state_.best_[kDestIdx], time_at_destination_));

    trace_always(" = {}\n", time_at_destination_);
  } else {
    for (auto const dest : get_search().search_state_.destinations_.front()) {
      trace_always(
          "updating new_time_at_destination = min(round_time={}, best={}, "
          "time_at_dest={})",
          state_.round_times_[k][to_idx(dest)], state_.best_[to_idx(dest)],
          time_at_destination_);

      time_at_destination_ =
          get_best(state_.round_times_[k][to_idx(dest)],
                   get_best(state_.best_[to_idx(dest)], time_at_destination_));

      trace_always(" = {}\n", time_at_destination_);
    }
  }
}
template <direction SearchDir, bool IntermodalTarget>
void raptor<SearchDir, IntermodalTarget>::reset_state() {
  state_.reset(tt(), kInvalidTime<SearchDir>);
}

template <direction SearchDir, bool IntermodalTarget>
void raptor<SearchDir, IntermodalTarget>::reconstruct_for_destination(
    unixtime_t const start_at_start,
    std::size_t const dest_index,
    location_idx_t const dest) {
  for (auto k = 1U; k != end_k(); ++k) {
    if (state_.round_times_[k][to_idx(dest)] == kInvalidTime<SearchDir>) {
      continue;
    }
    trace_always("ADDING JOURNEY: start={}, dest={} @ {}, transfers={}",
                 start_at_start,
                 state_.round_times_[k][to_idx(dest)].to_unixtime(tt()),
                 location{tt(), dest}, k - 1);
    auto const [optimal, it, dominated_by] =
        get_search().search_state_.results_[dest_index].add(
            journey{.legs_ = {},
                    .start_time_ = start_at_start,
                    .dest_time_ =
                        state_.round_times_[k][to_idx(dest)].to_unixtime(tt()),
                    .dest_ = dest,
                    .transfers_ = static_cast<std::uint8_t>(k - 1)});
    trace_always(" -> {}\n", optimal ? "OPT" : "DISCARD");
    if (optimal) {
      auto const outside_interval =
          holds_alternative<interval<unixtime_t>>(
              get_search().q_.start_time_) &&
          !get_search().search_state_.search_interval_.contains(
              it->start_time_);
      if (!outside_interval &&
          it->travel_time() < get_search().fastest_direct_) {
        try {
          reconstruct_journey<SearchDir>(tt(), get_search().q_, state_, *it);
        } catch (std::exception const& e) {
          get_search().search_state_.results_[dest_index].erase(it);
          log(log_lvl::error, "routing", "reconstruction failed: {}", e.what());
          print_state("RECONSTRUCT FAILED");
        }
      }
    } else {
      trace_always("  DOMINATED BY: start={}, dest={} @ {}, transfers={}\n",
                   dominated_by->start_time_, dominated_by->dest_time_,
                   location{tt(), dominated_by->dest_},
                   dominated_by->transfers_);
    }
  }
}

template <direction SearchDir, bool IntermodalTarget>
void raptor<SearchDir, IntermodalTarget>::reconstruct(
    unixtime_t const start_at_start) {
  if constexpr (IntermodalTarget) {
    reconstruct_for_destination(start_at_start, 0U,
                                get_special_station(special_station::kEnd));
  } else {
    for (auto const [i, t] : utl::enumerate(get_search().q_.destinations_)) {
      for (auto const dest : get_search().search_state_.destinations_[i]) {
        reconstruct_for_destination(start_at_start, i, dest);
      }
    }
  }
}

template struct raptor<direction::kForward, true>;
template struct raptor<direction::kBackward, true>;
template struct raptor<direction::kForward, false>;
template struct raptor<direction::kBackward, false>;

}  // namespace nigiri::routing
