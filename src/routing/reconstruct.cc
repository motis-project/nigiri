#include "nigiri/routing/reconstruct.h"

#include "utl/enumerate.h"
#include "utl/helpers/algorithm.h"

#include "nigiri/routing/for_each_meta.h"
#include "nigiri/routing/search_state.h"
#include "nigiri/special_stations.h"

namespace nigiri::routing {

constexpr auto const kTracing = false;
constexpr auto const kTraceStart = false;

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
std::optional<journey::leg> find_start_footpath(timetable const& tt,
                                                query const& q,
                                                journey const& j,
                                                search_state const& state) {
  trace("find_start_footpath()\n");

  constexpr auto const kFwd = SearchDir == direction::kForward;

  auto const is_better_or_eq = [](auto a, auto b) {
    return kFwd ? a <= b : a >= b;
  };

  auto const is_journey_start = [&](location_idx_t const candidate_l) {
    return utl::any_of(q.start_, [&](offset const& o) {
      return matches(tt, q.start_match_mode_, o.target_, candidate_l);
    });
  };

  auto const leg_start_location =
      kFwd ? j.legs_.back().from_ : j.legs_.back().to_;
  auto const leg_start_time =
      kFwd ? j.legs_.back().dep_time_ : j.legs_.back().arr_time_;

  if (q.start_match_mode_ != location_match_mode::kIntermodal &&
      is_journey_start(leg_start_location) &&
      is_better_or_eq(j.start_time_, leg_start_time)) {
    trace_start(
        "  leg_start_location={} is a start, time matches ({}) - done\n",
        location{tt, leg_start_location}, j.start_time_);
    return std::nullopt;
  } else {
    trace_start(
        "  direct start excluded intermodal_start={}, is_journey_start({})={}, "
        "leg_start_time={}, journey_start_time={}\n",
        q.start_match_mode_ == location_match_mode::kIntermodal,
        location{tt, leg_start_location}, is_journey_start(leg_start_location),
        leg_start_time, j.start_time_);
  }

  trace_start(
      "j_start={} is not a start meta={}, start={}, checking footpaths\n",
      location{tt, leg_start_location},
      q.start_match_mode_ == location_match_mode::kEquivalent,
      is_journey_start(leg_start_location));
  auto const& footpaths =
      kFwd ? tt.locations_.footpaths_in_[leg_start_location]
           : tt.locations_.footpaths_out_[leg_start_location];
  auto const j_start_time = routing_time{tt, j.start_time_};
  auto const fp_target_time = state.round_times_[0][to_idx(leg_start_location)];

  if (q.start_match_mode_ == location_match_mode::kIntermodal) {
    for (auto const& o : q.start_) {
      if (matches(tt, q.start_match_mode_, o.target_, leg_start_location) &&
          is_better_or_eq(j.start_time_,
                          leg_start_time - (kFwd ? 1 : -1) * o.duration_)) {
        trace_start(
            "  --> start: START -> {}  leg_start_time={}, j_start_time={}, "
            "offset={}\n",
            location{tt, o.target_}, leg_start_time, j.start_time_,
            o.duration_);
        return journey::leg{SearchDir,
                            get_special_station(special_station::kStart),
                            leg_start_location,
                            j.start_time_,
                            j.start_time_ + (kFwd ? 1 : -1) * o.duration_,
                            o};
      } else {
        trace_start(
            "  no start: START -> {}  matches={}, leg_start_location={}, "
            "leg_start_time={}, j_start_time={}, offset={}\n",
            location{tt, o.target_},
            matches(tt, q.start_match_mode_, o.target_, leg_start_location),
            location{tt, leg_start_location}, leg_start_time, j.start_time_,
            o.duration_);
      }

      for (auto const& fp : footpaths) {
        if (matches(tt, q.start_match_mode_, o.target_, fp.target_) &&
            is_better_or_eq(
                j.start_time_,
                leg_start_time -
                    (kFwd ? 1 : -1) * (o.duration_ + fp.duration_))) {
          trace_start(
              "  --> start: START -> {}  leg_start_time={}, j_start_time={}, "
              "offset={}, footpath=({}, {})\n",
              location{tt, o.target_}, leg_start_time, j.start_time_,
              o.duration_, fp.duration_, location{tt, fp.target_});
          return journey::leg{SearchDir,
                              get_special_station(special_station::kStart),
                              leg_start_location,
                              j.start_time_,
                              j.start_time_ + (kFwd ? 1 : -1) * o.duration_,
                              o};
        } else {
          trace_start(
              "  no start: START -> {}  matches={}, leg_start_location={}, "
              "leg_start_time={}, j_start_time={}, offset={}, footpath=({}, "
              "{})\n",
              location{tt, o.target_},
              matches(tt, q.start_match_mode_, o.target_, leg_start_location),
              location{tt, leg_start_location}, leg_start_time, j.start_time_,
              o.duration_, fp.duration_, location{tt, fp.target_});
        }
      }
    }
  } else {
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
                            fp};
      } else {
        trace_start(
            "  no start: {} -> {}  is_journey_start(fp.target_)={} "
            "fp_start_time={}, j_start_time={}, fp_duration={}\n",
            location{tt, fp.target_}, location{tt, leg_start_location},
            is_journey_start(fp.target_), fp_target_time, j_start_time,
            fp.duration_.count());
      }
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
    trace(" time={}\n", time);
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
                                       fp};
      return std::pair{fp_leg, *transport_leg};
    } else {
      trace("nothing found\n");
    }
    return std::nullopt;
  };

  auto const get_legs =
      [&](unsigned const k,
          location_idx_t const l) -> std::pair<journey::leg, journey::leg> {
    auto const curr_time = state.round_times_[k][to_idx(l)];
    if (q.dest_match_mode_ == location_match_mode::kIntermodal &&
        k == j.transfers_ + 1U) {
      trace("  CHECKING INTERMODAL DEST\n");
      for (auto const& dest_offset : q.destinations_[0]) {
        std::optional<std::pair<journey::leg, journey::leg>> ret;
        for_each_meta(
            tt, location_match_mode::kIntermodal, dest_offset.target_,
            [&](location_idx_t const eq) {
              auto const transfer_time = tt.locations_.transfer_time_[eq];
              auto intermodal_dest = check_fp(
                  k, l, curr_time, {eq, dest_offset.duration_ + transfer_time});
              if (intermodal_dest.has_value()) {
                trace(
                    "  found intermodal dest offset END [{}] -> {}: offset={}, "
                    "transfer_time={}\n",
                    curr_time, location{tt, dest_offset.target_},
                    dest_offset.duration_, transfer_time);
                (kFwd ? intermodal_dest->first.dep_time_ += transfer_time
                      : intermodal_dest->first.arr_time_ -= transfer_time);
                intermodal_dest->first.uses_ =
                    offset{eq, dest_offset.duration_ - transfer_time,
                           dest_offset.type_};
                ret = std::move(intermodal_dest);
              } else {
                trace(
                    "  BAD intermodal dest offset: END [{}] -> {}: {}, "
                    "transfer_time={}\n",
                    curr_time, location{tt, dest_offset.target_},
                    dest_offset.duration_, transfer_time);
              }

              for (auto const& fp : kFwd ? tt.locations_.footpaths_in_[eq]
                                         : tt.locations_.footpaths_out_[eq]) {
                auto fp_intermodal_dest = check_fp(
                    k, l, curr_time,
                    {fp.target_, dest_offset.duration_ + fp.duration_});
                if (fp_intermodal_dest.has_value()) {
                  trace(
                      "  found intermodal+footpath dest offset END [{}] -> {}: "
                      "offset={}, "
                      "transfer_time={}\n",
                      curr_time, location{tt, fp.target_}, fp.duration_,
                      transfer_time);
                  fp_intermodal_dest->first.uses_ = offset{
                      eq, fp.duration_ - transfer_time, dest_offset.type_};
                  ret = std::move(fp_intermodal_dest);
                } else {
                  trace(
                      "  BAD intermodal+footpath dest offset: {}@{} --{}--> "
                      "{}@{} --{}--> END@{} (type={})\n",
                      location{tt, fp.target_},
                      state.round_times_[k][to_idx(fp.target_)], fp.duration_,
                      location{tt, eq}, state.round_times_[k][to_idx(eq)],
                      dest_offset.duration_, curr_time, dest_offset.type_);
                }
              }
            });
        if (ret.has_value()) {
          return std::move(*ret);
        }
      }

      throw utl::fail(
          "intermodal destination reconstruction failed at k={}, t={}, "
          "stop=(name={}, id={}), time={}",
          k, j.transfers_, tt.locations_.names_[l].view(),
          tt.locations_.ids_[l].view(), curr_time);
    }

    trace("CHECKING TRANSFER\n");
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

  auto init_fp = find_start_footpath<SearchDir>(tt, q, j, state);
  if (init_fp.has_value()) {
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
