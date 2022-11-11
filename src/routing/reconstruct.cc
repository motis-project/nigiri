#include "nigiri/routing/reconstruct.h"

#include "utl/enumerate.h"

#include "nigiri/routing/for_each_meta.h"
#include "nigiri/routing/search_state.h"
#include "nigiri/special_stations.h"

namespace nigiri::routing {

constexpr auto const kTracing = true;
constexpr auto const kTraceStart = true;

bool matches(timetable const& tt,
             location_match_mode const mode,
             location_idx_t const a,
             location_idx_t const b) {
  switch (mode) {
    case location_match_mode::kExact: [[fallthrough]];
    case location_match_mode::kIntermodal: return a == b;
    case location_match_mode::kOnlyChildren: [[fallthrough]];
    case location_match_mode::kEquivalent:
      if (a == b) {
        return true;
      }

      {
        auto matches = false;
        for_each_meta(tt, mode, a, [&](location_idx_t const candidate) {
          matches = matches || (candidate == b);
        });
        return matches;
      }
    default: return true;
  }
}

template <typename... Args>
void trace(char const* fmt_str, Args... args) {
  if constexpr (kTracing) {
    fmt::print(std::cout, fmt_str, std::forward<Args&&>(args)...);
  }
}

template <typename... Args>
void trace_start(char const* fmt_str, Args... args) {
  if constexpr (kTraceStart) {
    fmt::print(std::cout, fmt_str, std::forward<Args&&>(args)...);
  }
}

template <direction SearchDir>
std::optional<journey::leg> find_initial_footpath(timetable const& tt,
                                                  query const& q,
                                                  journey const& j,
                                                  search_state const& state) {
  trace("find_initial_footpath()\n");

  constexpr auto const kFwd = SearchDir == direction::kForward;

  auto const is_journey_start = [&](location_idx_t const candidate_l) {
    return std::any_of(begin(q.start_), end(q.start_), [&](offset const& o) {
      return matches(tt, q.start_match_mode_, o.location_, candidate_l);
    });
  };

  auto const leg_start_location =
      kFwd ? j.legs_.back().from_ : j.legs_.back().to_;
  auto const leg_start_time =
      kFwd ? j.legs_.back().dep_time_ : j.legs_.front().arr_time_;
  if (is_journey_start(leg_start_location)) {
    if (leg_start_time == j.start_time_) {
      trace_start(
          "  leg_start_location={} is a start, time matches ({}) - done\n",
          location{tt, leg_start_location}, j.start_time_);
      return std::nullopt;
    } else {
      trace_start(
          "  leg_start_location={} is a start, times differ ({} vs {})\n",
          location{tt, leg_start_location}, j.start_time_, leg_start_time);

      auto min = offset{
          .location_ = location_idx_t::invalid(),
          .offset_ = duration_t{std::numeric_limits<duration_t::rep>::max()},
          .type_ = 0U};

      for (auto const& o : q.start_) {
        if (!matches(tt, q.start_match_mode_, o.location_,
                     leg_start_location)) {
          trace_start(
              "    location mismatch / time mismatch: {} vs {}, {} vs {} with "
              "offset={}, match_mode={}\n",
              location{tt, o.location_}, location{tt, leg_start_location},
              j.start_time_, leg_start_time, o.offset_, q.start_match_mode_);
          continue;
        }

        if (o.offset_ < min.offset_) {
          trace_start("    location match: {} vs {}, {} vs {} with offset={}\n",
                      location{tt, o.location_},
                      location{tt, leg_start_location}, j.start_time_,
                      leg_start_time, o.offset_);
          min = o;
        }
      }

      utl::verify(min.location_ != location_idx_t::invalid(), "no start found");

      return q.start_match_mode_ == location_match_mode::kIntermodal
                 ? std::make_optional<journey::leg>(
                       SearchDir,
                       kFwd ? get_special_station(special_station::kStart)
                            : min.location_,
                       kFwd ? min.location_
                            : get_special_station(special_station::kEnd),
                       j.start_time_, leg_start_time, min.type_)
                 : std::nullopt;
    }
  }

  trace_start("j_start={} is not a start meta={}, checking footpaths\n",
              location{tt, leg_start_location},
              q.start_match_mode_ == location_match_mode::kEquivalent);
  auto const& footpaths =
      kFwd ? tt.locations_.footpaths_in_[leg_start_location]
           : tt.locations_.footpaths_out_[leg_start_location];
  auto const j_start_time = routing_time{tt, j.start_time_};
  auto const fp_target_time = state.round_times_[0][to_idx(leg_start_location)];
  for (auto const& fp : footpaths) {
    if (is_journey_start(fp.target_) &&
        fp_target_time != kInvalidTime<SearchDir> &&
        std::abs((j_start_time - fp_target_time).count()) ==
            fp.duration_.count()) {
      trace("  -> from={}, j_start={}, leg_start={}, fp_start\n",
            location{tt, fp.target_}, location{tt, leg_start_location},
            j.start_time_, fp_target_time);
      return journey::leg{SearchDir,
                          fp.target_,
                          leg_start_location,
                          j.start_time_,
                          fp_target_time.to_unixtime(tt),
                          footpath_idx_t{}};
    } else {
      trace_start(
          "  no start: {} -> {}  is_journey_start(fp.target_)={} "
          "fp_start_time={}, j_start_time={}, fp_duration={}\n",
          location{tt, fp.target_}, location{tt, leg_start_location},
          is_journey_start(fp.target_), fp_target_time, j_start_time,
          fp.duration_.count());
    }
  }

  throw utl::fail("no valid journey start found");
}

template <direction SearchDir>
void reconstruct_journey(timetable const& tt,
                         query const& q,
                         search_state const& state,
                         journey& j) {
  (void)q;  // TODO(felix) support intermodal start

  constexpr auto const kFwd = SearchDir == direction::kForward;
  auto const is_better_or_eq = [](auto a, auto b) {
    return kFwd ? a <= b : a >= b;
  };

  auto const best = [&](std::uint32_t const k, location_idx_t const l) {
    return std::min(state.best_[to_idx(l)], state.round_times_[k][to_idx(l)]);
  };

  auto const find_entry_in_prev_round =
      [&](unsigned const k, transport const& t, route_idx_t const r,
          std::size_t const from_stop_idx,
          routing_time const time) -> std::optional<journey::leg> {
    auto const& stop_seq = tt.route_location_seq_[r];

    auto const n_stops =
        kFwd ? from_stop_idx + 1 : stop_seq.size() - from_stop_idx;
    for (auto i = 1U; i != n_stops; ++i) {
      auto const stop_idx =
          static_cast<unsigned>(kFwd ? from_stop_idx - i : from_stop_idx + i);
      auto const stop = timetable::stop{stop_seq[stop_idx]};
      auto const l = stop.location_idx();

      if ((kFwd && !stop.in_allowed()) || (!kFwd && !stop.out_allowed())) {
        continue;
      }

      auto const event_time = routing_time{
          t.day_, tt.event_mam(t.t_idx_, stop_idx,
                               kFwd ? event_type::kDep : event_type::kArr)};
      if (is_better_or_eq(state.round_times_[k - 1][to_idx(l)], event_time)) {
        trace("      FOUND ENTRY AT {}: {} <= {}\n", location{tt, l},
              best(k - 1, l), event_time);
        return journey::leg{
            SearchDir,
            timetable::stop{stop_seq[stop_idx]}.location_idx(),
            timetable::stop{stop_seq[from_stop_idx]}.location_idx(),
            event_time.to_unixtime(tt),
            time.to_unixtime(tt),
            journey::transport_enter_exit{
                t, stop_idx, static_cast<unsigned>(from_stop_idx)}};
      } else {
        trace(
            "      ENTRY NOT POSSIBLE AT {}: k={} k-1={}, best_at_stop=min({}, "
            "{})={} > event_time={}\n",
            location{tt, l}, k, k - 1, state.best_[to_idx(l)],
            state.round_times_[k - 1][to_idx(l)], best(k - 1, l), event_time);
      }

      // special case: first stop with meta stations
      if (k == 1 && q.start_match_mode_ == location_match_mode::kEquivalent) {
        for (auto const& eq : tt.locations_.equivalences_[l]) {
          if (is_better_or_eq(state.round_times_[k - 1][to_idx(eq)],
                              event_time)) {
            return journey::leg{
                SearchDir,
                timetable::stop{stop_seq[stop_idx]}.location_idx(),
                timetable::stop{stop_seq[from_stop_idx]}.location_idx(),
                event_time.to_unixtime(tt),
                time.to_unixtime(tt),
                journey::transport_enter_exit{
                    t, stop_idx, static_cast<unsigned>(from_stop_idx)}};
          }
        }
      }
    }

    return std::nullopt;
  };

  auto const get_route_transport =
      [&](unsigned const k, routing_time const time, route_idx_t const r,
          std::size_t const stop_idx) -> std::optional<journey::leg> {
    for (auto const t : tt.route_transport_ranges_[r]) {
      auto const event_mam =
          tt.event_mam(t, stop_idx, kFwd ? event_type::kArr : event_type::kDep);
      trace("  CHECKING TRANSPORT transport={}, name={}, stop={}\n", t,
            tt.trip_display_names_
                [tt.merged_trips_[tt.transport_to_trip_section_[t].front()]
                     .front()]
                    .view(),
            location{tt, timetable::stop{tt.route_location_seq_[r][stop_idx]}
                             .location_idx()});
      if (event_mam.count() % 1440 != time.mam().count()) {
        trace("    -> ev_mam mismatch: transport_ev={} vs footpath = {}\n ",
              duration_t{event_mam.count()}, duration_t{time.mam().count()});
        continue;
      }

      auto const day_offset =
          static_cast<cista::base_t<day_idx_t>>(event_mam.count() / 1440);
      auto const day = time.day() - day_offset;
      if (!tt.bitfields_[tt.transport_traffic_days_[t]].test(to_idx(day))) {
        trace("    -> no traffic on day {}\n ", to_idx(day));
        continue;
      }

      auto leg =
          find_entry_in_prev_round(k, transport{t, day}, r, stop_idx, time);
      if (leg.has_value()) {
        return leg;
      }
      trace("    -> no entry found\n ", to_idx(day));
    }
    return std::nullopt;
  };

  auto const get_transport =
      [&](unsigned const k, location_idx_t const l,
          routing_time const time) -> std::optional<journey::leg> {
    trace(" time={}, adjusted_time={}\n", time, time);
    for (auto const& r : tt.location_routes_[l]) {
      auto const location_seq = tt.route_location_seq_[r];
      for (auto const [i, stop] : utl::enumerate(location_seq)) {
        auto const s = timetable::stop{stop};
        if (s.location_idx() != l ||  //
            (kFwd && (i == 0U || !s.out_allowed())) ||
            (!kFwd && (i == location_seq.size() - 1 || !s.in_allowed()))) {
          continue;
        }

        auto leg = get_route_transport(k, time, r, i);
        if (leg.has_value()) {
          return leg;
        }
      }
    }
    return std::nullopt;
  };

  auto const check_fp = [&](unsigned const k, location_idx_t const l,
                            routing_time const curr_time, footpath const fp)
      -> std::optional<std::pair<journey::leg, journey::leg>> {
    auto const fp_start = curr_time - (kFwd ? fp.duration_ : -fp.duration_);

    trace(
        "round {}: searching for transports at (name={}, id={}) with fp_start "
        "= {}\n ",
        k, tt.locations_.names_.at(fp.target_).view(),
        tt.locations_.ids_.at(fp.target_).view(), fp_start);

    auto const transport_leg = get_transport(k, fp.target_, fp_start);

    if (transport_leg.has_value()) {
      trace("found:\n");
      if constexpr (kTracing) {
        transport_leg->print(std::cout, tt, 1, false);
      }
      trace(" fp leg: {} {} --{}--> {} {}\n", location{tt, l},
            fp_start.to_unixtime(tt), fp.duration_, location{tt, fp.target_},
            curr_time.to_unixtime(tt));

      auto const fp_leg = journey::leg{SearchDir,
                                       fp.target_,
                                       l,
                                       fp_start.to_unixtime(tt),
                                       curr_time.to_unixtime(tt),
                                       footpath_idx_t::invalid()};
      return std::pair{fp_leg, *transport_leg};
    } else {
      trace("nothing found\n");
    }
    return std::nullopt;
  };

  auto const get_legs =
      [&](unsigned const k,
          location_idx_t const l) -> std::pair<journey::leg, journey::leg> {
    trace("CHECKING TRANSFER\n");
    auto const curr_time = state.round_times_[k][to_idx(l)];
    auto transfer_at_same_stop =
        check_fp(k, l, curr_time,
                 footpath{l, (k == j.transfers_ + 1U)
                                 ? 0_minutes
                                 : tt.locations_.transfer_time_[l]});
    if (transfer_at_same_stop.has_value()) {
      return std::move(*transfer_at_same_stop);
    }

    trace("CHECKING FOOTPATHS OF {}\n", tt.locations_.names_.at(l).view());
    auto const fps =
        kFwd ? tt.locations_.footpaths_in_[l] : tt.locations_.footpaths_out_[l];
    for (auto const& fp : fps) {
      trace("FP: (name={}, id={}) --{}--> (name={}, id={})\n",
            tt.locations_.names_.at(l).view(), tt.locations_.ids_.at(l).view(),
            fp.duration_.count(), tt.locations_.names_.at(fp.target_).view(),
            tt.locations_.ids_.at(fp.target_).view());
      auto fp_legs = check_fp(k, l, curr_time, fp);
      if (fp_legs.has_value()) {
        return std::move(*fp_legs);
      }
    }

    throw utl::fail(
        "reconstruction failed at k={}, t={}, stop=(name={}, id={}), time={}",
        k, j.transfers_, tt.locations_.names_[l].view(),
        tt.locations_.ids_[l].view(), curr_time);
  };

  auto l = j.dest_;
  for (auto i = 0U; i <= j.transfers_; ++i) {
    auto const k = j.transfers_ + 1 - i;
    trace("RECONSTRUCT WITH k={}\n", k);
    auto [fp_leg, transport_leg] = get_legs(k, l);
    l = kFwd ? transport_leg.from_ : transport_leg.to_;
    j.add(std::move(fp_leg));
    j.add(std::move(transport_leg));
  }

  if (auto init_fp = find_initial_footpath<SearchDir>(tt, q, j, state);
      init_fp.has_value()) {
    j.add(std::move(*init_fp));
  }

  if constexpr (kFwd) {
    std::reverse(begin(j.legs_), end(j.legs_));
  }
}

template void reconstruct_journey<direction::kForward>(
    timetable const& tt, query const& q, search_state const& state, journey& j);

template void reconstruct_journey<direction::kBackward>(
    timetable const& tt, query const& q, search_state const& state, journey& j);

}  // namespace nigiri::routing
