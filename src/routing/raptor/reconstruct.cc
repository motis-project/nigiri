#include "nigiri/routing/raptor/reconstruct.h"

#include "utl/enumerate.h"
#include "utl/helpers/algorithm.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/for_each_meta.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/special_stations.h"

namespace nigiri::routing {

constexpr auto const kTracing = false;
constexpr auto const kTraceStart = false;

template <typename... Args>
void trace_rc(char const* fmt_str, Args... args) {
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

bool is_journey_start(timetable const& tt,
                      query const& q,
                      location_idx_t const candidate_l) {
  return utl::any_of(q.start_, [&](offset const& o) {
    return matches(tt, q.start_match_mode_, o.target_, candidate_l);
  });
}

template <direction SearchDir>
std::optional<journey::leg> find_start_footpath(timetable const& tt,
                                                query const& q,
                                                journey const& j,
                                                raptor_state const& state,
                                                date::sys_days const base) {
  trace_rc("find_start_footpath()\n");

  constexpr auto const kFwd = SearchDir == direction::kForward;

  auto const is_better_or_eq = [](auto a, auto b) {
    return kFwd ? a <= b : a >= b;
  };

  auto const is_ontrip = holds_alternative<unixtime_t>(q.start_time_);
  auto const start_matches = [&](delta_t const a, delta_t const b) {
    return is_ontrip ? is_better_or_eq(a, b) : a == b;
  };

  auto const leg_start_location =
      kFwd ? j.legs_.back().from_ : j.legs_.back().to_;
  auto const leg_start_time =
      kFwd ? j.legs_.back().dep_time_ : j.legs_.back().arr_time_;

  if (q.start_match_mode_ != location_match_mode::kIntermodal &&
      is_journey_start(tt, q, leg_start_location) &&
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
        location{tt, leg_start_location},
        is_journey_start(tt, q, leg_start_location), leg_start_time,
        j.start_time_);
  }

  trace_start(
      "j_start={} is not a start meta={}, start={}, checking footpaths\n",
      location{tt, leg_start_location},
      q.start_match_mode_ == location_match_mode::kEquivalent,
      is_journey_start(tt, q, leg_start_location));
  auto const& footpaths =
      kFwd ? tt.locations_.footpaths_in_[leg_start_location]
           : tt.locations_.footpaths_out_[leg_start_location];
  auto const j_start_time = unix_to_delta(base, j.start_time_);
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
      if (is_journey_start(tt, q, fp.target_) &&
          fp_target_time != kInvalidDelta<SearchDir> &&
          start_matches(j_start_time + (kFwd ? 1 : -1) * fp.duration_.count(),
                        fp_target_time)) {
        trace_rc(
            "  -> from={}, j_start={}, journey_start={}, fp_target_time={}, "
            "duration={}\n",
            location{tt, fp.target_}, location{tt, leg_start_location},
            j.start_time_, fp_target_time, fp.duration_);
        return journey::leg{SearchDir,
                            fp.target_,
                            leg_start_location,
                            j.start_time_,
                            delta_to_unix(base, fp_target_time),
                            fp};
      } else {
        trace_start(
            "  no start: {} -> {}  is_journey_start(fp.target_)={} "
            "fp_start_time={}, j_start_time={}, fp_duration={}\n",
            location{tt, fp.target_}, location{tt, leg_start_location},
            is_journey_start(tt, q, fp.target_), fp_target_time, j_start_time,
            fp.duration_.count());
      }
    }
  }

  throw utl::fail("no valid journey start found");
}

template <direction SearchDir>
void reconstruct_journey(timetable const& tt,
                         query const& q,
                         raptor_state const& raptor_state,
                         journey& j,
                         date::sys_days const base,
                         day_idx_t const base_day_idx) {
  constexpr auto const kFwd = SearchDir == direction::kForward;
  auto const is_better_or_eq = [](auto a, auto b) {
    return kFwd ? a <= b : a >= b;
  };

  auto const is_ontrip = holds_alternative<unixtime_t>(q.start_time_);
  auto const start_matches = [&](delta_t const a, delta_t const b) {
    return is_ontrip ? is_better_or_eq(a, b) : a == b;
  };

  auto const best = [&](std::uint32_t const k, location_idx_t const l) {
    return std::min(raptor_state.best_[to_idx(l)],
                    raptor_state.round_times_[k][to_idx(l)]);
  };

  auto const find_entry_in_prev_round =
      [&](unsigned const k, transport const& t, route_idx_t const r,
          std::size_t const from_stop_idx,
          delta_t const time) -> std::optional<journey::leg> {
    auto const& stop_seq = tt.route_location_seq_[r];

    auto const n_stops =
        kFwd ? from_stop_idx + 1 : stop_seq.size() - from_stop_idx;
    for (auto i = 1U; i != n_stops; ++i) {
      auto const stop_idx =
          static_cast<unsigned>(kFwd ? from_stop_idx - i : from_stop_idx + i);
      auto const stp = stop{stop_seq[stop_idx]};
      auto const l = stp.location_idx();

      if ((kFwd && !stp.in_allowed()) || (!kFwd && !stp.out_allowed())) {
        continue;
      }

      auto const event_time =
          tt_to_delta(base_day_idx, t.day_,
                      tt.event_mam(t.t_idx_, stop_idx,
                                   kFwd ? event_type::kDep : event_type::kArr)
                          .as_duration());
      if (is_better_or_eq(raptor_state.round_times_[k - 1][to_idx(l)],
                          event_time)) {
        trace_rc(
            "      FOUND ENTRY AT name={}, dbg={}, location={}: {} <= {}\n",
            tt.transport_name(t.t_idx_), tt.dbg(t.t_idx_), location{tt, l},
            best(k - 1, l), event_time);
        return journey::leg{
            SearchDir,
            stop{stop_seq[stop_idx]}.location_idx(),
            stop{stop_seq[from_stop_idx]}.location_idx(),
            delta_to_unix(base, event_time),
            delta_to_unix(base, time),
            journey::transport_enter_exit{
                t, stop_idx, static_cast<unsigned>(from_stop_idx)}};
      } else {
        trace_rc(
            "      ENTRY NOT POSSIBLE AT {}: k={} k-1={}, best_at_stop=min({}, "
            "{})={}={} > event_time={}={}\n",
            location{tt, l}, k, k - 1, raptor_state.best_[to_idx(l)],
            raptor_state.round_times_[k - 1][to_idx(l)], best(k - 1, l),
            delta_to_unix(base, best(k - 1, l)), event_time,
            tt.event_time(t, stop_idx,
                          kFwd ? event_type::kDep : event_type::kArr));
      }

      // special case: first stop with meta stations
      if (k == 1 && q.start_match_mode_ == location_match_mode::kEquivalent) {
        if (is_journey_start(tt, q, l) &&
            start_matches(raptor_state.round_times_[k - 1][to_idx(l)],
                          event_time)) {
          trace_rc(
              "      ENTRY AT META={}, ORIG={}, name={}, dbg={}: k={} k-1={}, "
              "best_at_stop=min({}, {})={} <= event_time={}\n",
              location{tt, l}, location{tt, l}, tt.transport_name(t.t_idx_),
              tt.dbg(t.t_idx_), k, k - 1, raptor_state.best_[to_idx(l)],
              raptor_state.round_times_[k - 1][to_idx(l)], best(k - 1, l),
              event_time);
          return journey::leg{
              SearchDir,
              stop{stop_seq[stop_idx]}.location_idx(),
              stop{stop_seq[from_stop_idx]}.location_idx(),
              delta_to_unix(base, event_time),
              delta_to_unix(base, time),
              journey::transport_enter_exit{
                  t, stop_idx, static_cast<unsigned>(from_stop_idx)}};
        }
      }
    }

    return std::nullopt;
  };

  auto const get_route_transport =
      [&](unsigned const k, delta_t const time, route_idx_t const r,
          std::size_t const stop_idx) -> std::optional<journey::leg> {
    auto const [day, mam] = split_day_mam(base_day_idx, time);

    for (auto const t : tt.route_transport_ranges_[r]) {
      auto const event_mam =
          tt.event_mam(t, stop_idx, kFwd ? event_type::kArr : event_type::kDep);
      trace_rc(
          "  CHECKING TRANSPORT name={}, dbg={}, stop={}, time={} (day={}, "
          "mam={}), traffic_day={}, event_mam={}\n",
          tt.transport_name(t), tt.dbg(t),
          location{tt,
                   stop{tt.route_location_seq_[r][stop_idx]}.location_idx()},
          delta_to_unix(base, time), day, mam,
          static_cast<int>(to_idx(day)) - event_mam.count() / 1440, event_mam);

      if (minutes_after_midnight_t{event_mam.count() % 1440} != mam) {
        trace_rc("    -> ev_mam mismatch: transport_ev={} vs footpath = {}\n ",
                 duration_t{event_mam.count()}, duration_t{mam});
        continue;
      }

      auto const traffic_day = static_cast<std::size_t>(
          static_cast<int>(to_idx(day)) - event_mam.count() / 1440);
      if (!tt.bitfields_[tt.transport_traffic_days_[t]].test(traffic_day)) {
        trace_rc("    -> no traffic on day {}\n ", traffic_day);
        continue;
      }

      auto leg = find_entry_in_prev_round(
          k, transport{t, day_idx_t{traffic_day}}, r, stop_idx, time);
      if (leg.has_value()) {
        return leg;
      }
      trace_rc("    -> no entry found\n ", traffic_day);
    }
    return std::nullopt;
  };

  auto const get_transport =
      [&](unsigned const k, location_idx_t const l,
          delta_t const time) -> std::optional<journey::leg> {
    trace_rc(" time={}\n", delta_to_unix(base, time));
    for (auto const& r : tt.location_routes_[l]) {
      auto const location_seq = tt.route_location_seq_[r];
      for (auto const [i, s] : utl::enumerate(location_seq)) {
        auto const stp = stop{s};
        if (stp.location_idx() != l ||  //
            (kFwd && (i == 0U || !stp.out_allowed())) ||
            (!kFwd && (i == location_seq.size() - 1 || !stp.in_allowed()))) {
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
                            delta_t const curr_time, footpath const fp)
      -> std::optional<std::pair<journey::leg, journey::leg>> {
    auto const fp_start = static_cast<delta_t>(
        curr_time - (kFwd ? fp.duration_ : -fp.duration_).count());

    trace_rc(
        "round {}: searching for transports at {} with curr_time={} --{}--> "
        "fp_start={}\n ",
        k, location{tt, fp.target_}, delta_to_unix(base, curr_time),
        fp.duration_, delta_to_unix(base, fp_start));

    auto const transport_leg = get_transport(k, fp.target_, fp_start);

    if (transport_leg.has_value()) {
      trace_rc("found:\n");
      if constexpr (kTracing) {
        transport_leg->print(std::cout, tt, 1, true);
      }
      trace_rc(" fp leg: {} {} --{}--> {} {}\n", location{tt, l},
               delta_to_unix(base, fp_start), fp.duration_,
               location{tt, fp.target_}, delta_to_unix(base, curr_time));

      auto const fp_leg = journey::leg{SearchDir,
                                       fp.target_,
                                       l,
                                       delta_to_unix(base, fp_start),
                                       delta_to_unix(base, curr_time),
                                       fp};
      return std::pair{fp_leg, *transport_leg};
    } else {
      trace_rc("nothing found\n");
    }
    return std::nullopt;
  };

  auto const get_legs =
      [&](unsigned const k,
          location_idx_t const l) -> std::pair<journey::leg, journey::leg> {
    auto const curr_time = raptor_state.round_times_[k][to_idx(l)];
    if (q.dest_match_mode_ == location_match_mode::kIntermodal &&
        k == j.transfers_ + 1U) {
      trace_rc("  CHECKING INTERMODAL DEST\n");
      for (auto const& dest_offset : q.destination_) {
        std::optional<std::pair<journey::leg, journey::leg>> ret;
        for_each_meta(
            tt, location_match_mode::kIntermodal, dest_offset.target_,
            [&](location_idx_t const eq) {
              auto intermodal_dest =
                  check_fp(k, l, curr_time, {eq, dest_offset.duration_});
              if (intermodal_dest.has_value()) {
                trace_rc(
                    "  found intermodal dest offset END [{}] -> {}: "
                    "offset={}\n",
                    curr_time, location{tt, dest_offset.target_},
                    dest_offset.duration_);
                intermodal_dest->first.uses_ =
                    offset{eq, dest_offset.duration_, dest_offset.type_};
                ret = std::move(intermodal_dest);
              } else {
                trace_rc("  BAD intermodal dest offset: END [{}] -> {}: {}\n",
                         curr_time, location{tt, dest_offset.target_},
                         dest_offset.duration_);
              }

              for (auto const& fp : kFwd ? tt.locations_.footpaths_in_[eq]
                                         : tt.locations_.footpaths_out_[eq]) {
                auto fp_intermodal_dest = check_fp(
                    k, l, curr_time,
                    {fp.target_, dest_offset.duration_ + fp.duration_});
                if (fp_intermodal_dest.has_value()) {
                  trace_rc(
                      "  found intermodal+footpath dest offset END [{}] -> {}: "
                      "offset={}\n",
                      curr_time, location{tt, fp.target_}, fp.duration_);
                  fp_intermodal_dest->first.uses_ =
                      offset{eq, fp.duration_, dest_offset.type_};
                  ret = std::move(fp_intermodal_dest);
                } else {
                  trace_rc(
                      "  BAD intermodal+footpath dest offset: {}@{} --{}--> "
                      "{}@{} --{}--> END@{} (type={})\n",
                      location{tt, fp.target_},
                      raptor_state.round_times_[k][to_idx(fp.target_)],
                      fp.duration_, location{tt, eq},
                      raptor_state.round_times_[k][to_idx(eq)],
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
          "stop={}, time={}",
          k, j.transfers_, location{tt, l}, curr_time);
    }

    trace_rc("CHECKING TRANSFER\n");
    auto transfer_at_same_stop =
        check_fp(k, l, curr_time,
                 footpath{l, (k == j.transfers_ + 1U)
                                 ? 0_i8_minutes
                                 : tt.locations_.transfer_time_[l]});
    if (transfer_at_same_stop.has_value()) {
      return std::move(*transfer_at_same_stop);
    }

    trace_rc("CHECKING FOOTPATHS OF {}\n", tt.locations_.names_.at(l).view());
    auto const fps =
        kFwd ? tt.locations_.footpaths_in_[l] : tt.locations_.footpaths_out_[l];
    for (auto const& fp : fps) {
      trace_rc("FP: (name={}, id={}) --{}--> (name={}, id={})\n",
               tt.locations_.names_.at(l).view(),
               tt.locations_.ids_.at(l).view(), fp.duration_.count(),
               tt.locations_.names_.at(fp.target_).view(),
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
    trace_rc("RECONSTRUCT WITH k={}\n", k);
    auto [fp_leg, transport_leg] = get_legs(k, l);
    l = kFwd ? transport_leg.from_ : transport_leg.to_;
    j.add(std::move(fp_leg));
    j.add(std::move(transport_leg));
  }

  auto init_fp = find_start_footpath<SearchDir>(tt, q, j, raptor_state, base);
  if (init_fp.has_value()) {
    j.add(std::move(*init_fp));
  }

  if constexpr (kFwd) {
    std::reverse(begin(j.legs_), end(j.legs_));
  }

  if (kTracing) {
    j.print(std::cout, tt, true);
  }
}

template void reconstruct_journey<direction::kForward>(timetable const&,
                                                       query const&,
                                                       raptor_state const&,
                                                       journey&,
                                                       date::sys_days const,
                                                       day_idx_t const);

template void reconstruct_journey<direction::kBackward>(timetable const&,
                                                        query const&,
                                                        raptor_state const&,
                                                        journey&,
                                                        date::sys_days const,
                                                        day_idx_t const);

}  // namespace nigiri::routing
