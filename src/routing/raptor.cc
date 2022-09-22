#include "nigiri/routing/raptor.h"

#include "fmt/core.h"

#include "utl/enumerate.h"
#include "utl/equal_ranges_linear.h"
#include "utl/erase_if.h"
#include "utl/overloaded.h"

#include "nigiri/routing/journey.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/reconstruct.h"
#include "nigiri/routing/search_state.h"
#include "nigiri/routing/start_times.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

constexpr auto const kTracing = true;

template <typename... Args>
void trace(char const* fmt_str, Args... args) {
  if constexpr (kTracing) {
    fmt::print(std::cerr, fmt_str, std::forward<Args&&>(args)...);
  }
}

template <direction SearchDir>
raptor<SearchDir>::raptor(timetable& tt, search_state& state, query q)
    : tt_{tt},
      n_days_{static_cast<std::uint16_t>(tt_.date_range_.size().count())},
      q_{std::move(q)},
      state_{state} {}

template <direction SearchDir>
bool raptor<SearchDir>::is_better(auto a, auto b) {
  return kFwd ? a < b : a > b;
}

template <direction SearchDir>
bool raptor<SearchDir>::is_better_or_eq(auto a, auto b) {
  return kFwd ? a <= b : a >= b;
}

template <direction SearchDir>
auto raptor<SearchDir>::get_best(auto a, auto b) {
  return is_better(a, b) ? a : b;
}

template <direction SearchDir>
routing_time raptor<SearchDir>::time_at_stop(transport const& t,
                                             unsigned const stop_idx,
                                             event_type const ev_type) {
  return {t.day_, tt_.event_mam(t.t_idx_, stop_idx, ev_type)};
}

template <direction SearchDir>
transport raptor<SearchDir>::get_earliest_transport(
    unsigned const k,
    route_idx_t const r,
    unsigned const stop_idx,
    location_idx_t const l_idx) {
  auto const time = state_.round_times_[k - 1][to_idx(l_idx)];
  if (time == kInvalidTime) {
    trace("┊ │    et: location=(name={}, id={}, idx={}) => NOT REACHABLE\n",
          tt_.locations_.names_[l_idx].view(),
          tt_.locations_.ids_[l_idx].view(), l_idx);
    return {transport_idx_t::invalid(), day_idx_t::invalid()};
  }

  auto const transport_range = tt_.route_transport_ranges_[r];
  auto const [day_at_stop, mam_at_stop] = time.day_idx_mam();

  auto const n_days_to_iterate =
      std::min(kMaxTravelTime / 1440U + 1,
               kFwd ? n_days_ - to_idx(day_at_stop) : to_idx(day_at_stop) + 1U);

  trace(
      "┊ │    et: current_best_at_stop={}, stop_idx={}, "
      "location=(name={}, id={}, idx={}), n_days_to_iterate={}\n",
      time, stop_idx, tt_.locations_.names_[l_idx].view(),
      tt_.locations_.ids_[l_idx].view(), l_idx, n_days_to_iterate);

  for (auto i = std::uint16_t{0U}; i != n_days_to_iterate; ++i) {
    auto const day = kFwd ? day_at_stop + i : day_at_stop - i;
    for (auto t = kFwd ? transport_range.from_ : transport_range.to_ - 1;
         t != (kFwd ? transport_range.to_ : transport_range.from_ - 1);
         kFwd ? ++t : --t) {
      auto const ev = tt_.event_mam(t, stop_idx,
                                    kFwd ? event_type::kDep : event_type::kArr);
      auto const ev_mam = minutes_after_midnight_t{ev.count() % 1440};
      if (day == day_at_stop && !is_better_or_eq(mam_at_stop, ev_mam)) {
        trace(
            "┊ │    => day={}/{}, best_mam={}, transport_mam={} => NO REACH!\n",
            i, day, mam_at_stop, ev_mam);
        continue;
      }

      auto const ev_day_offset =
          static_cast<cista::base_t<day_idx_t>>(ev.count() / 1440);
      if (!tt_.bitfields_[tt_.transport_traffic_days_[t]].test(to_idx(day) -
                                                               ev_day_offset)) {
        trace(
            "┊ │    => day={}/{}, best_mam={}, transport_mam={} => NO "
            "TRAFFIC!\n",
            i, day, mam_at_stop, ev_mam);
        continue;
      }

      trace("┊ │    => transport_idx={} at day {} (day_offset={}) - ev={}\n", t,
            day, ev_day_offset,
            routing_time{day_idx_t{day - ev_day_offset},
                         minutes_after_midnight_t{ev_mam}});
      return {t, day - ev_day_offset};
    }
  }
  trace("┊ │    => et: NOT FOUND\n");
  return {transport_idx_t::invalid(), day_idx_t::invalid()};
}

template <direction SearchDir>
void raptor<SearchDir>::update_route(unsigned const k, route_idx_t const r) {
  auto const& stop_seq = tt_.route_location_seq_[r];

  auto et = transport{};
  for (auto i = 0U; i != stop_seq.size(); ++i) {
    auto const stop_idx =
        static_cast<unsigned>(kFwd ? i : stop_seq.size() - i - 1U);
    auto const l_idx =
        cista::to_idx(timetable::stop{stop_seq[stop_idx]}.location_idx());
    auto const current_best =
        get_best(state_.best_[l_idx], state_.round_times_[k - 1][l_idx]);

    trace(
        "┊ │  stop_idx={}, location=(name={}, id={}, idx={}): "
        "current_best={}\n",
        stop_idx, tt_.locations_.names_[location_idx_t{l_idx}].view(),
        tt_.locations_.ids_[location_idx_t{l_idx}].view(), l_idx, current_best);

    if (et.is_valid()) {
      auto const by_transport_time =
          time_at_stop(et, stop_idx,
                       kFwd ? event_type::kArr : event_type::kDep) +
          (kFwd ? 1 : -1) *
              tt_.locations_.transfer_time_[location_idx_t{l_idx}];
      if (is_better(by_transport_time, current_best) &&
          is_better(by_transport_time, time_at_destination_)) {
        trace(
            "┊ │    transport={}, time_by_transport={} BETTER THAN "
            "current_best={} => update, marking station (name={}, id={})!\n",
            et, by_transport_time, current_best,
            tt_.locations_.names_[location_idx_t{l_idx}].view(),
            tt_.locations_.ids_[location_idx_t{l_idx}].view());

        state_.best_[l_idx] = by_transport_time;
        state_.round_times_[k][l_idx] = by_transport_time;
        state_.station_mark_[l_idx] = true;
        if (state_.is_destination_[l_idx]) {
          time_at_destination_ = by_transport_time;
        }
      } else {
        trace(
            "┊ │    by_transport={} NOT better than current_best={} => no "
            "update\n",
            by_transport_time, current_best);
      }
    }

    if (!(kFwd && stop_idx == stop_seq.size() - 1) &&
        !(kBwd && stop_idx == 0) &&
        (!et.is_valid() ||
         is_better_or_eq(
             state_.round_times_[k - 1][l_idx],
             time_at_stop(et, stop_idx,
                          kFwd ? event_type::kDep : event_type::kArr)))) {
      trace(
          "┊ │    update et: stop_idx={}, et_valid={}, stop_time={}, "
          "transport_time={}\n",
          stop_idx, et.is_valid(), state_.round_times_[k - 1][l_idx],
          et.is_valid()
              ? time_at_stop(et, stop_idx,
                             kFwd ? event_type::kDep : event_type::kArr)
              : kInvalidTime);
      et = get_earliest_transport(k, r, stop_idx, location_idx_t{l_idx});
    }
  }
}

template <direction SearchDir>
void raptor<SearchDir>::update_footpaths(unsigned const k) {
  trace("┊ ├ FOOTPATHS\n");
  for (auto l_idx = location_idx_t{0U}; l_idx != tt_.n_locations(); ++l_idx) {
    if (!state_.station_mark_[to_idx(l_idx)]) {
      continue;
    }

    auto const fps = kFwd ? tt_.locations_.footpaths_out_[l_idx]
                          : tt_.locations_.footpaths_in_[l_idx];
    trace("┊ ├ updating footpaths of {}\n",
          tt_.locations_.names_[l_idx].view());
    for (auto const& fp : fps) {
      trace("┊ ├ footpath: (name={}, id={}) --{}--> (name={}, id={})\n",
            tt_.locations_.names_[l_idx].view(),
            tt_.locations_.ids_[l_idx].view(), fp.duration_,
            tt_.locations_.names_[fp.target_].view(),
            tt_.locations_.ids_[fp.target_].view());

      auto& time_at_fp_target = state_.best_[to_idx(fp.target_)];
      auto const fp_target_time =
          state_.round_times_[k][to_idx(l_idx)] +
          ((kFwd ? 1 : -1) * fp.duration_) -
          ((kFwd ? 1 : -1) * tt_.locations_.transfer_time_[l_idx]);
      if (is_better(fp_target_time, time_at_fp_target) &&
          is_better(fp_target_time, time_at_destination_)) {
        trace("┊ ├--> UPDATE {} -> {}\n", time_at_fp_target, fp_target_time);
        state_.round_times_[k][to_idx(fp.target_)] = fp_target_time;
        state_.best_[to_idx(fp.target_)] = fp_target_time;
        state_.station_mark_[to_idx(fp.target_)] = true;
        if (state_.is_destination_[to_idx(fp.target_)]) {
          time_at_destination_ = fp_target_time;
        }
      }
    }
  }
}

template <direction SearchDir>
unsigned raptor<SearchDir>::end_k() const {
  return std::min(kMaxTransfers, q_.max_transfers_) + 1U;
}

template <direction SearchDir>
void raptor<SearchDir>::rounds() {
  print_state();

  for (auto k = 1U; k != end_k(); ++k) {
    trace("┊ round k={}\n", k);

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

    if (!any_marked) {
      trace("┊ ╰ nothing marked, exit\n\n");
      break;
    }

    std::fill(begin(state_.station_mark_), end(state_.station_mark_), false);

    for (auto r_id = 0U; r_id != tt_.n_routes(); ++r_id) {
      if (!state_.route_mark_[r_id]) {
        continue;
      }
      trace("┊ ├ updating route {}\n", r_id);
      update_route(k, route_idx_t{r_id});
    }

    std::fill(begin(state_.route_mark_), end(state_.route_mark_), false);

    update_footpaths(k);

    trace("┊ ╰ round {} done\n", k);
    print_state();
  }
}

template <direction SearchDir>
void raptor<SearchDir>::force_print_state(char const* comment) {
  fmt::print(std::cerr, "INFO: {}, time_at_destination={}\n", comment,
             time_at_destination_);
  for (auto l = 0U; l != tt_.n_locations(); ++l) {
    if (state_.best_[l] == kInvalidTime && !state_.is_destination_[l]) {
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
        std::cerr, "[{}] {:8} [name={:48}, track={:10}, id={:16}]: ",
        state_.is_destination_[l] ? "X" : "_", l, name, track,
        id.substr(0, std::min(std::string_view ::size_type{16U}, id.size())));
    auto const b = state_.best_[l];
    if (b == kInvalidTime) {
      fmt::print(std::cerr, "best=_________, round_times: ");
    } else {
      fmt::print(std::cerr, "best={:9}, round_times: ", b);
    }
    for (auto i = 0U; i != kMaxTransfers + 1U; ++i) {
      auto const t = state_.round_times_[i][l];
      if (t != kInvalidTime) {
        fmt::print(std::cerr, "{:9} ", t);
      } else {
        fmt::print(std::cerr, "_________ ");
      }
    }
    fmt::print(std::cerr, "\n");
  }
}

template <direction SearchDir>
void raptor<SearchDir>::print_state(char const* comment) {
  if constexpr (kTracing) {
    force_print_state(comment);
  }
}

template <direction SearchDir>
void raptor<SearchDir>::route() {
  state_.reset(tt_, kInvalidTime);
  collect_destinations(tt_, q_.destinations_, q_.dest_match_mode_,
                       state_.destinations_, state_.is_destination_);
  state_.results_.resize(
      std::max(state_.results_.size(), state_.destinations_.size()));
  get_starts<SearchDir>(tt_, q_.start_time_, q_.start_, q_.start_match_mode_,
                        state_.starts_);
  utl::equal_ranges_linear(
      state_.starts_,
      [](start const& a, start const& b) {
        return a.time_at_start_ == b.time_at_start_;
      },
      [&](auto&& from_it, auto&& to_it) {
        for (auto const& s : it_range{from_it, to_it}) {
          trace("init round: {} = {} at (name={} id={})\n", s.time_at_stop_,
                routing_time{tt_, s.time_at_stop_}.t(),
                tt_.locations_.names_.at(s.stop_).view(),
                tt_.locations_.ids_.at(s.stop_).view());
          state_.round_times_[0U][to_idx(s.stop_)] = {tt_, s.time_at_stop_};
          state_.best_[to_idx(s.stop_)] = {tt_, s.time_at_stop_};
          state_.station_mark_[to_idx(s.stop_)] = true;
        }
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

template <direction SearchDir>
void raptor<SearchDir>::reconstruct(unixtime_t const start_at_start) {
  //  force_print_state("RECONSTRUCT");

  for (auto const [i, t] : utl::enumerate(q_.destinations_)) {
    for (auto const dest : state_.destinations_[i]) {
      for (auto k = 1U; k != end_k(); ++k) {
        if (state_.round_times_[k][to_idx(dest)] == kInvalidTime) {
          continue;
        }
        auto const [optimal, it] = state_.results_[i].add(journey{
            .legs_ = {},
            .start_time_ = start_at_start,
            .dest_time_ = state_.round_times_[k][to_idx(dest)].to_unixtime(tt_),
            .dest_ = dest,
            .transfers_ = static_cast<std::uint8_t>(k - 1)});
        if (optimal) {
          auto const outside_interval =
              holds_alternative<interval<unixtime_t>>(q_.start_time_) &&
              !q_.start_time_.as<interval<unixtime_t>>().contains(
                  it->start_time_);
          if (!outside_interval) {
            try {
              reconstruct_journey<SearchDir>(tt_, q_, state_, *it);
            } catch (std::exception const& e) {
              state_.results_[i].erase(it);
              log(log_lvl::error, "routing", "reconstruction failed: {}",
                  e.what());
              force_print_state("RECONSTRUCT FAILED");
            }
          }
        }
      }
    }
  }
}

template struct raptor<direction::kForward>;
template struct raptor<direction::kBackward>;

}  // namespace nigiri::routing
