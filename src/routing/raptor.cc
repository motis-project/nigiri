#include "nigiri/routing/raptor.h"

#include "fmt/core.h"

#include "utl/enumerate.h"
#include "utl/equal_ranges_linear.h"
#include "utl/erase_if.h"
#include "utl/overloaded.h"
#include "utl/timing.h"

#include "nigiri/routing/dijkstra.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/reconstruct.h"
#include "nigiri/routing/search_state.h"
#include "nigiri/routing/start_times.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"

#define NIGIRI_LOWER_BOUND

#define NIGIRI_RAPTOR_COUNTING
#ifdef NIGIRI_RAPTOR_COUNTING
#define NIGIRI_COUNT(s) ++stats_.s
#else
#define NIGIRI_COUNT(s)
#endif

// #define NIGIRI_RAPTOR_TRACING
// #define NIGIRI_RAPTOR_TRACING_ONLY_UPDATES

#ifdef NIGIRI_RAPTOR_TRACING

#ifdef NIGIRI_RAPTOR_TRACING_ONLY_UPDATES
#define trace(...)
#else
#define trace(...) fmt::print(std::cout, __VA_ARGS__);
#endif

#define trace_always(...) fmt::print(std::cout, __VA_ARGS__)
#define trace_upd(...) fmt::print(std::cout, __VA_ARGS__)
#else
template <typename... Ts>
void unused(Ts...) {}

#define trace(fmt_str, ...)
#define trace_always(fmt_str, ...)
#define trace_upd(fmt_str, ...)
#endif

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
                                            search_state& state,
                                            query q)
    : tt_{tt},
      n_days_{static_cast<std::uint16_t>(tt_.date_range_.size().count())},
      q_{std::move(q)},
      state_{state} {}

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
  return {t.day_, tt_.event_mam(r, t.t_idx_, stop_idx, ev_type)};
}

template <direction SearchDir, bool IntermodalTarget>
routing_time raptor<SearchDir, IntermodalTarget>::time_at_stop(
    transport const t, unsigned const stop_idx, event_type const ev_type) {
  return {t.day_, tt_.event_mam(t.t_idx_, stop_idx, ev_type)};
}

template <direction SearchDir, bool IntermodalTarget>
transport raptor<SearchDir, IntermodalTarget>::get_earliest_transport(
    unsigned const k,
    route_idx_t const r,
    unsigned const stop_idx,
    location_idx_t const l_idx) {
  NIGIRI_COUNT(n_earliest_trip_calls_);

  auto const time = state_.round_times_[k - 1][to_idx(l_idx)];
  if (time == kInvalidTime<SearchDir>) {
    trace("┊ │    et: location=(name={}, id={}, idx={}) => NOT REACHABLE\n",
          tt_.locations_.names_[l_idx].view(),
          tt_.locations_.ids_[l_idx].view(),
          l_idx) return {transport_idx_t::invalid(), day_idx_t::invalid()};
  }

  auto const [day_at_stop, mam_at_stop] = time.day_idx_mam();

  auto const n_days_to_iterate =
      std::min(kMaxTravelTime / 1440U + 1,
               kFwd ? n_days_ - to_idx(day_at_stop) : to_idx(day_at_stop) + 1U);

  trace(
      "┊ │    et: current_best_at_stop={}, stop_idx={}, "
      "location=(name={}, id={}, idx={}), n_days_to_iterate={}\n",
      time, stop_idx, tt_.locations_.names_[l_idx].view(),
      tt_.locations_.ids_[l_idx].view(), l_idx, n_days_to_iterate)

      auto const event_times = tt_.event_times_at_stop(
          r, stop_idx, kFwd ? event_type::kDep : event_type::kArr);

#if defined(NIGIRI_RAPTOR_TRACING) && \
    !defined(NIGIRI_RAPTOR_TRACING_ONLY_UPDATES)
  for (auto const [t_offset, x] : utl::enumerate(event_times)) {
    auto const t = tt_.route_transport_ranges_[r][t_offset];
    trace("┊ │        event_times: transport={}: {} at {}: {}\n", t,
          kFwd ? "dep" : "arr", location{tt_, l_idx}, x)
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
    auto const ev_time_range = it_range{
        i == 0U ? seek_first_day() : get_begin_it<SearchDir>(event_times),
        get_end_it<SearchDir>(event_times)};
    if (ev_time_range.empty()) {
      continue;
    }
    auto const base =
        static_cast<unsigned>(&*ev_time_range.begin_ - event_times.data());
    for (auto const [t_offset, ev] : utl::enumerate(ev_time_range)) {
      auto const ev_mam = minutes_after_midnight_t{
          ev.count() < 1440 ? ev.count() : ev.count() % 1440};
      if (is_better_or_eq(time_at_destination_, routing_time{day, ev_mam})) {
        return {transport_idx_t::invalid(), day_idx_t::invalid()};
      }

      auto const t = tt_.route_transport_ranges_[r][base + t_offset];
      if (day == day_at_stop && !is_better_or_eq(mam_at_stop, ev_mam)) {
        trace(
            "┊ │      => transport={}, name={}, day={}/{}, best_mam={}, "
            "transport_mam={}, transport_time={} => NO REACH!\n",
            t,
            tt_.trip_display_names_
                [tt_.merged_trips_[tt_.transport_to_trip_section_[t].front()]
                     .front()]
                    .view(),
            i, day, mam_at_stop, ev_mam, ev);
        continue;
      }

      auto const ev_day_offset = static_cast<day_idx_t::value_t>(
          ev.count() < 1440
              ? 0
              : static_cast<cista::base_t<day_idx_t>>(ev.count() / 1440));
      if (!tt_.bitfields_[tt_.transport_traffic_days_[t]].test(
              static_cast<std::size_t>(to_idx(day) - ev_day_offset))) {
        trace(
            "┊ │      => transport={}, name={}, day={}/{}, ev_day_offset={}, "
            "best_mam={}, "
            "transport_mam={}, transport_time={} => NO TRAFFIC!\n",
            t,
            tt_.trip_display_names_
                [tt_.merged_trips_[tt_.transport_to_trip_section_[t].front()]
                     .front()]
                    .view(),
            i, day, ev_day_offset, mam_at_stop, ev_mam, ev);
        continue;
      }

      trace(
          "┊ │      => ET FOUND: transport={}, name={} at day {} "
          "(day_offset={}) - ev_mam={}, ev_time={}, ev={}\n",
          t,
          tt_.trip_display_names_
              [tt_.merged_trips_[tt_.transport_to_trip_section_[t].front()]
                   .front()]
                  .view(),
          day, ev_day_offset, ev_mam, ev,
          routing_time{day_idx_t{day - ev_day_offset},
                       minutes_after_midnight_t{ev_mam}});
      return {t, static_cast<day_idx_t>(day - ev_day_offset)};
    }
  }
  trace("┊ │    => et: NOT FOUND\n");
  return {transport_idx_t::invalid(), day_idx_t::invalid()};
}

template <direction SearchDir, bool IntermodalTarget>
bool raptor<SearchDir, IntermodalTarget>::update_route(unsigned const k,
                                                       route_idx_t const r) {
  auto const stop_seq = tt_.route_location_seq_[r];
  bool any_marked = false;

  auto et = transport{};
  for (auto i = 0U; i != stop_seq.size(); ++i) {
    auto const stop_idx =
        static_cast<unsigned>(kFwd ? i : stop_seq.size() - i - 1U);
    auto const stop = timetable::stop{stop_seq[stop_idx]};
    auto const l_idx = cista::to_idx(stop.location_idx());

    if (!et.is_valid() && !state_.prev_station_mark_[l_idx]) {
      continue;
    }

    auto current_best =
        get_best(state_.best_[l_idx], state_.round_times_[k - 1][l_idx]);
    auto const transfer_time_offset =
        (kFwd ? 1 : -1) * tt_.locations_.transfer_time_[location_idx_t{l_idx}];

    trace(
        "┊ │  stop_idx={}, location=(name={}, id={}, idx={}): "
        "current_best={}\n",
        stop_idx, tt_.locations_.names_[location_idx_t{l_idx}].view(),
        tt_.locations_.ids_[location_idx_t{l_idx}].view(), l_idx, current_best);

    if (et.is_valid()) {
      auto const is_destination = state_.is_destination_[l_idx];
      auto const by_transport_time = time_at_stop(
          r, et, stop_idx, kFwd ? event_type::kArr : event_type::kDep);
      auto const by_transport_time_with_transfer =
          by_transport_time + ((is_destination && !IntermodalTarget ? 0U : 1U) *
                               transfer_time_offset);

      if ((kFwd ? stop.out_allowed() : stop.in_allowed()) &&
          is_better(by_transport_time_with_transfer, current_best) &&
          is_better(by_transport_time_with_transfer, time_at_destination_)) {

#ifdef NIGIRI_LOWER_BOUND
        auto const lower_bound =
            state_.travel_time_lower_bound_[to_idx(stop.location_idx())];
        if (lower_bound.count() ==
                std::numeric_limits<duration_t::rep>::max() ||
            !is_better(by_transport_time + (kFwd ? 1 : -1) * lower_bound,
                       time_at_destination_)) {

#ifdef NIGIRI_RAPTOR_TRACING
          auto const trip_idx =
              tt_.merged_trips_[tt_.transport_to_trip_section_[et.t_idx_]
                                    .front()]
                  .front();
          trace_upd(
              "┊ │    *** LB NO UPD: transport={}, name={}, debug={}:{}, "
              "time_by_transport={} "
              "BETTER THAN "
              "current_best={} => (name={}, id={}) - "
              "LB={}, LB_AT_DEST={}!\n",
              et, tt_.trip_display_names_[trip_idx].view(),
              tt_.source_file_names_
                  [tt_.trip_debug_[trip_idx].front().source_file_idx_]
                      .view(),
              tt_.trip_debug_[trip_idx].front().line_number_from_,
              by_transport_time, current_best,
              tt_.locations_.names_[location_idx_t{l_idx}].view(),
              tt_.locations_.ids_[location_idx_t{l_idx}].view(), lower_bound,
              by_transport_time + (kFwd ? 1 : -1) * lower_bound);
#endif

          NIGIRI_COUNT(route_update_prevented_by_lower_bound_);
          continue;
        }
#endif

#ifdef NIGIRI_RAPTOR_TRACING
        auto const trip_idx =
            tt_.merged_trips_[tt_.transport_to_trip_section_[et.t_idx_].front()]
                .front();
        trace_upd(
            "┊ │    transport={}, name={}, debug={}:{}, time_by_transport={} "
            "BETTER THAN "
            "current_best={} => update, marking station (name={}, id={})!\n",
            et, tt_.trip_display_names_[trip_idx].view(),
            tt_.source_file_names_
                [tt_.trip_debug_[trip_idx].front().source_file_idx_]
                    .view(),
            tt_.trip_debug_[trip_idx].front().line_number_from_,
            by_transport_time, current_best,
            tt_.locations_.names_[location_idx_t{l_idx}].view(),
            tt_.locations_.ids_[location_idx_t{l_idx}].view());
#endif

        NIGIRI_COUNT(n_earliest_arrival_updated_by_route_);
        state_.best_[l_idx] = by_transport_time_with_transfer;
        state_.round_times_[k][l_idx] = by_transport_time_with_transfer;
        state_.station_mark_[l_idx] = true;
        if constexpr (!IntermodalTarget) {
          if (is_destination) {
            time_at_destination_ =
                get_best(by_transport_time_with_transfer, time_at_destination_);
          }
        }
        current_best = by_transport_time_with_transfer;
        any_marked = true;
      } else {
        trace(
            "┊ │    by_transport={} NOT better than time_at_destination={} OR "
            "current_best={} => no update\n",
            by_transport_time_with_transfer, time_at_destination_,
            current_best);
      }
    }

    if (!state_.prev_station_mark_[l_idx]) {
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
          is_better_or_eq(current_best, et_time_at_stop)) {
        trace(
            "┊ │    update et: stop_idx={}, et_valid={}, stop_time={}, "
            "transport_time={}\n",
            stop_idx, et.is_valid(), state_.round_times_[k - 1][l_idx],
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
                              kFwd ? event_type::kDep : event_type::kArr) +
                     transfer_time_offset,
                 et_time_at_stop))) {
          et = new_et;
        }
      }
    }
  }
  return any_marked;
}

template <direction SearchDir, bool IntermodalTarget>
void raptor<SearchDir, IntermodalTarget>::update_footpaths(unsigned const k) {
  trace_always("┊ ├ FOOTPATHS\n");
  for (auto l_idx = location_idx_t{0U}; l_idx != tt_.n_locations(); ++l_idx) {
    if (!state_.station_mark_[to_idx(l_idx)] ||
        state_.best_[to_idx(l_idx)] == kInvalidTime<SearchDir>) {
      continue;
    }

    if constexpr (IntermodalTarget) {
      if (state_.is_destination_[to_idx(l_idx)]) {
        trace_upd("┊ ├  INTERMODAL TARGET STATION {}\n", location{tt_, l_idx});
        for (auto const& o : q_.destinations_[0]) {
          if (!matches(tt_, location_match_mode::kIntermodal, o.target_,
                       l_idx)) {
            trace("┊ ├    {} != {}\n", location{tt_, o.target_},
                  location{tt_, l_idx});
            continue;
          }

          auto const intermodal_target_time =
              state_.best_[to_idx(l_idx)] + ((kFwd ? 1 : -1) * o.duration_);
          if (is_better(intermodal_target_time, time_at_destination_)) {
            trace_upd(
                "┊ ├     intermodal: (name={}, id={}, best={}) --{}--> (DEST, "
                "best={}) --> update => {}\n",
                tt_.locations_.names_[l_idx].view(),
                tt_.locations_.ids_[l_idx].view(),
                state_.round_times_[k][to_idx(l_idx)], o.duration_,
                time_at_destination_, intermodal_target_time);

            state_.round_times_[k][to_idx(get_special_station(
                special_station::kEnd))] = intermodal_target_time;
            time_at_destination_ = intermodal_target_time;
          }
        }
      }
    }

    auto const fps = kFwd ? tt_.locations_.footpaths_out_[l_idx]
                          : tt_.locations_.footpaths_in_[l_idx];
    trace("┊ ├ updating footpaths of {}\n", location{tt_, l_idx});
    for (auto const& fp : fps) {
      NIGIRI_COUNT(n_footpaths_visited_);

      auto const target = to_idx(fp.target_);
      auto const min =
          std::min(state_.best_[target], state_.round_times_[k][target]);
      auto const fp_target_time =
          state_.best_[to_idx(l_idx)]  //
          + ((kFwd ? 1 : -1) * fp.duration_)  //
          - ((kFwd ? 1 : -1) * tt_.locations_.transfer_time_[l_idx]);

      if (is_better(fp_target_time, min) &&
          is_better(fp_target_time, time_at_destination_)) {

#ifdef NIGIRI_LOWER_BOUND
        auto const lower_bound =
            state_.travel_time_lower_bound_[to_idx(fp.target_)];
        if (lower_bound.count() ==
                std::numeric_limits<duration_t::rep>::max() ||
            !is_better(fp_target_time + (kFwd ? 1 : -1) * lower_bound,
                       time_at_destination_)) {

          trace_upd(
              "┊ ├ *** LB NO UPD: (name={}, id={}, best={}) --{}--> (name={}, "
              "id={}, best={}) --> update => {}, LB={}, AT_DEST={}\n",
              tt_.locations_.names_[l_idx].view(),
              tt_.locations_.ids_[l_idx].view(),
              state_.round_times_[k][to_idx(l_idx)], fp.duration_,
              tt_.locations_.names_[fp.target_].view(),
              tt_.locations_.ids_[fp.target_].view(),
              state_.round_times_[k][to_idx(fp.target_)], fp_target_time,
              lower_bound, fp_target_time + (kFwd ? 1 : -1) * lower_bound);

          NIGIRI_COUNT(fp_update_prevented_by_lower_bound_);
          continue;
        }
#endif

        trace_upd(
            "┊ ├ footpath: (name={}, id={}, best={}) --{}--> (name={}, id={}, "
            "best={}) --> update => {}\n",
            tt_.locations_.names_[l_idx].view(),
            tt_.locations_.ids_[l_idx].view(),
            state_.round_times_[k][to_idx(l_idx)], fp.duration_,
            tt_.locations_.names_[fp.target_].view(),
            tt_.locations_.ids_[fp.target_].view(),
            state_.round_times_[k][to_idx(fp.target_)], fp_target_time);

        NIGIRI_COUNT(n_earliest_arrival_updated_by_footpath_);
        state_.round_times_[k][to_idx(fp.target_)] = fp_target_time;
        state_.station_mark_[to_idx(fp.target_)] = true;

        if (state_.is_destination_[to_idx(fp.target_)]) {
          time_at_destination_ = get_best(time_at_destination_, fp_target_time);
        }
      }
    }
  }
}

template <direction SearchDir, bool IntermodalTarget>
unsigned raptor<SearchDir, IntermodalTarget>::end_k() const {
  return std::min(kMaxTransfers, q_.max_transfers_) + 1U;
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
        for (auto const& r : tt_.location_routes_[l_idx]) {
          state_.route_mark_[to_idx(r)] = true;
        }
      }
    }

    std::swap(state_.prev_station_mark_, state_.station_mark_);
    std::fill(begin(state_.station_mark_), end(state_.station_mark_), false);

    if (!any_marked) {
      trace_always("┊ ╰ no routes marked, exit\n\n");
      return;
    }

    any_marked = false;
    for (auto r_id = 0U; r_id != tt_.n_routes(); ++r_id) {
      if (!state_.route_mark_[r_id]) {
        continue;
      }
      trace("┊ ├ updating route {}\n", r_id);

      NIGIRI_COUNT(n_routes_visited_);
      any_marked |= update_route(k, route_idx_t{r_id});
    }

    std::fill(begin(state_.route_mark_), end(state_.route_mark_), false);
    if (!any_marked) {
      trace_always("┊ ╰ no stations marked, exit\n\n");
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

  fmt::print(std::cout, "INFO: {}, time_at_destination={}\n", comment,
             time_at_destination_);
  for (auto l = 0U; l != tt_.n_locations(); ++l) {
    if (!is_special(location_idx_t{l}) && !state_.is_destination_[l] &&
        state_.best_[l] == kInvalidTime<SearchDir> && empty_rounds(l)) {
      continue;
    }

    std::string_view name, track;
    auto const type = tt_.locations_.types_.at(location_idx_t{l});
    if (type == location_type::kTrack) {
      name =
          tt_.locations_.names_.at(tt_.locations_.parents_[location_idx_t{l}])
              .view();
      track = tt_.locations_.names_.at(location_idx_t{l}).view();
    } else {
      name = tt_.locations_.names_.at(location_idx_t{l}).view();
      track = "---";
    }
    tt_.locations_.names_[location_idx_t{l}].view();
    auto const id = tt_.locations_.ids_[location_idx_t{l}].view();
    fmt::print(
        std::cout, "[{}] {:8} [name={:48}, track={:10}, id={:16}]: ",
        state_.is_destination_[l] ? "X" : "_", l, name, track,
        id.substr(0, std::min(std::string_view ::size_type{16U}, id.size())));
    auto const b = state_.best_[l];
    if (b == kInvalidTime<SearchDir>) {
      fmt::print(std::cout, "best=_________, round_times: ");
    } else {
      fmt::print(std::cout, "best={:9}, round_times: ", b);
    }
    for (auto i = 0U; i != kMaxTransfers + 1U; ++i) {
      auto const t = state_.round_times_[i][l];
      if (t != kInvalidTime<SearchDir>) {
        fmt::print(std::cout, "{:9} ", t);
      } else {
        fmt::print(std::cout, "_________ ");
      }
    }
    fmt::print(std::cout, "\n");
  }
}

#ifdef NIGIRI_RAPTOR_TRACING
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
  for (auto const dest : state_.destinations_.front()) {
    time_at_destination_ =
        get_best(state_.round_times_[k][to_idx(dest)],
                 get_best(state_.best_[to_idx(dest)], time_at_destination_));
  }
}

template <direction SearchDir, bool IntermodalTarget>
void raptor<SearchDir, IntermodalTarget>::route() {
  state_.reset(tt_, kInvalidTime<SearchDir>);
  collect_destinations(tt_, q_.destinations_, q_.dest_match_mode_,
                       state_.destinations_, state_.is_destination_);
  state_.results_.resize(
      std::max(state_.results_.size(), state_.destinations_.size()));
  get_starts<SearchDir>(tt_, q_.start_time_, q_.start_, q_.start_match_mode_,
                        q_.use_start_footpaths_, state_.starts_);

#ifdef NIGIRI_LOWER_BOUND

#ifdef NIGIRI_RAPTOR_COUNTING
  UTL_START_TIMING(lb);
#endif
  dijkstra(tt_, q_, kFwd ? tt_.fwd_search_lb_graph_ : tt_.bwd_search_lb_graph_,
           state_.travel_time_lower_bound_);
  for (auto l = location_idx_t{0U}; l != tt_.locations_.children_.size(); ++l) {
    auto const lb = state_.travel_time_lower_bound_[to_idx(l)];
    for (auto const c : tt_.locations_.children_[l]) {
      state_.travel_time_lower_bound_[to_idx(c)] = lb;
    }
  }
#ifdef NIGIRI_RAPTOR_COUNTING
  UTL_STOP_TIMING(lb);
  stats_.lb_time_ = static_cast<std::uint64_t>(UTL_TIMING_MS(lb));
#endif

#ifdef NIGIRI_RAPTOR_TRACING
  for (auto const [l, lb] : utl::enumerate(state_.travel_time_lower_bound_)) {
    if (lb.count() != std::numeric_limits<duration_t::rep>::max()) {
      trace_always("lb {}: {}\n", location{tt_, location_idx_t{l}}, lb.count());
    }
  }
#endif
#endif
  utl::equal_ranges_linear(
      state_.starts_,
      [](start const& a, start const& b) {
        return a.time_at_start_ == b.time_at_start_;
      },
      [&](auto&& from_it, auto&& to_it) {
        std::fill(begin(state_.best_), end(state_.best_),
                  kInvalidTime<SearchDir>);
        for (auto const& s : it_range{from_it, to_it}) {
          trace_always(
              "init: time_at_start={}, time_at_stop={} at (name={} id={})\n",
              s.time_at_start_, s.time_at_stop_,
              tt_.locations_.names_.at(s.stop_).view(),
              tt_.locations_.ids_.at(s.stop_).view());
          state_.round_times_[0U][to_idx(s.stop_)] = {tt_, s.time_at_stop_};
          state_.best_[to_idx(s.stop_)] = {tt_, s.time_at_stop_};
          state_.station_mark_[to_idx(s.stop_)] = true;
        }
        time_at_destination_ = routing_time{tt_, from_it->time_at_stop_} +
                               (kFwd ? 1 : -1) * duration_t{kMaxTravelTime};
        rounds();
        reconstruct(from_it->time_at_start_);
      });
  if (holds_alternative<interval<unixtime_t>>(q_.start_time_)) {
    for (auto& r : state_.results_) {
      utl::erase_if(r, [&](journey const& j) {
        return !q_.start_time_.as<interval<unixtime_t>>().contains(
            j.start_time_);
      });
    }
  }
  state_.search_interval_ = q_.start_time_.apply(utl::overloaded{
      [](interval<unixtime_t> const& start_interval) { return start_interval; },
      [](unixtime_t const start_time) {
        return interval<unixtime_t>{start_time, start_time};
      }});
}

template <direction SearchDir, bool IntermodalTarget>
void raptor<SearchDir, IntermodalTarget>::reconstruct_for_destination(
    unixtime_t const start_at_start,
    std::size_t const i,
    location_idx_t const dest) {
  for (auto k = 1U; k != end_k(); ++k) {
    if (state_.round_times_[k][to_idx(dest)] == kInvalidTime<SearchDir>) {
      continue;
    }
    trace_always("ADDING JOURNEY: start={}, dest={} @ {}, transfers={}",
                 start_at_start,
                 state_.round_times_[k][to_idx(dest)].to_unixtime(tt_),
                 location{tt_, dest}, k - 1);
    auto const [optimal, it] = state_.results_[i].add(journey{
        .legs_ = {},
        .start_time_ = start_at_start,
        .dest_time_ = state_.round_times_[k][to_idx(dest)].to_unixtime(tt_),
        .dest_ = dest,
        .transfers_ = static_cast<std::uint8_t>(k - 1)});
    trace_always(" -> {}\n", optimal ? "OPT" : "DISCARD");
    if (optimal) {
      auto const outside_interval =
          holds_alternative<interval<unixtime_t>>(q_.start_time_) &&
          !q_.start_time_.as<interval<unixtime_t>>().contains(it->start_time_);
      if (!outside_interval) {
        try {
          reconstruct_journey<SearchDir>(tt_, q_, state_, *it);
        } catch (std::exception const& e) {
          state_.results_[i].erase(it);
          log(log_lvl::error, "routing", "reconstruction failed: {}", e.what());
          print_state("RECONSTRUCT FAILED");
        }
      }
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
    for (auto const [i, t] : utl::enumerate(q_.destinations_)) {
      for (auto const dest : state_.destinations_[i]) {
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
