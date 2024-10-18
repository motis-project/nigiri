#include "nigiri/routing/raptor/reconstruct.h"

#include "utl/enumerate.h"
#include "utl/helpers/algorithm.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/for_each_meta.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/special_stations.h"

namespace nigiri::routing {

bool is_journey_start(timetable const& tt,
                      query const& q,
                      location_idx_t const candidate_l) {
  return utl::any_of(q.start_, [&](offset const& o) {
    return matches(tt, q.start_match_mode_, o.target(), candidate_l);
  });
}

template <direction SearchDir>
std::optional<journey::leg> find_start_footpath(timetable const& tt,
                                                query const& q,
                                                journey const& j,
                                                raptor_state const& state,
                                                date::sys_days const base) {
  trace_rc_find_start_footpath;

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
    trace_rc_direct_start_found;
    return std::nullopt;
  } else {
    trace_rc_direct_start_excluded;
  }

  trace_rc_checking_start_fp;

  auto const& footpaths =
      kFwd ? tt.locations_.footpaths_in_[q.prf_idx_][leg_start_location]
           : tt.locations_.footpaths_out_[q.prf_idx_][leg_start_location];
  auto const j_start_time = unix_to_delta(base, j.start_time_);
  auto const fp_target_time = state.round_times_[0][to_idx(leg_start_location)];

  if (q.start_match_mode_ == location_match_mode::kIntermodal) {
    for (auto const& o : q.start_) {
      if (matches(tt, q.start_match_mode_, o.target(), leg_start_location) &&
          is_better_or_eq(j.start_time_,
                          leg_start_time - (kFwd ? 1 : -1) * o.duration())) {
        trace_rc_intermodal_start_found;
        return journey::leg{SearchDir,
                            get_special_station(special_station::kStart),
                            leg_start_location,
                            j.start_time_,
                            j.start_time_ + (kFwd ? 1 : -1) * o.duration(),
                            o};
      } else {
        trace_rc_intermodal_no_match;
      }

      for (auto const& fp : footpaths) {
        if (matches(tt, q.start_match_mode_, o.target(), fp.target()) &&
            is_better_or_eq(
                j.start_time_,
                leg_start_time -
                    (kFwd ? 1 : -1) * (o.duration() + fp.duration()))) {
          trace_rc_intermodal_fp_start_found;
          return journey::leg{SearchDir,
                              get_special_station(special_station::kStart),
                              leg_start_location,
                              j.start_time_,
                              j.start_time_ + (kFwd ? 1 : -1) * o.duration(),
                              o};
        } else {
          trace_rc_intermodal_fp_no_match;
        }
      }
    }
  } else {
    for (auto const& fp : footpaths) {
      if (is_journey_start(tt, q, fp.target()) &&
          fp_target_time != kInvalidDelta<SearchDir> &&
          start_matches(j_start_time + (kFwd ? 1 : -1) * fp.duration().count(),
                        fp_target_time)) {
        trace_rc_fp_start_found;
        return journey::leg{SearchDir,
                            fp.target(),
                            leg_start_location,
                            j.start_time_,
                            delta_to_unix(base, fp_target_time),
                            fp};
      } else {
        trace_rc_fp_start_no_match;
      }
    }
  }

  throw utl::fail("no valid journey start found");
}

template <direction SearchDir>
std::optional<journey::leg> find_start_footpath_gpu(timetable const& tt,
                                                query const& q,
                                                journey const& j,
                                                mem const& state,
                                                date::sys_days const base) {
  trace_rc_find_start_footpath;

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
    trace_rc_direct_start_found;
    return std::nullopt;
  } else {
    trace_rc_direct_start_excluded;
  }

  trace_rc_checking_start_fp;

  auto const& footpaths =
      kFwd ? tt.locations_.footpaths_in_[q.prf_idx_][leg_start_location]
           : tt.locations_.footpaths_out_[q.prf_idx_][leg_start_location];
  auto const j_start_time = unix_to_delta(base, j.start_time_);
  auto const fp_target_time = state.host_.round_times_[0* state.host_.column_count_round_times_ + to_idx(leg_start_location)];

  if (q.start_match_mode_ == location_match_mode::kIntermodal) {
    for (auto const& o : q.start_) {
      if (matches(tt, q.start_match_mode_, o.target(), leg_start_location) &&
          is_better_or_eq(j.start_time_,
                          leg_start_time - (kFwd ? 1 : -1) * o.duration())) {
        trace_rc_intermodal_start_found;
        return journey::leg{SearchDir,
                            get_special_station(special_station::kStart),
                            leg_start_location,
                            j.start_time_,
                            j.start_time_ + (kFwd ? 1 : -1) * o.duration(),
                            o};
      } else {
        trace_rc_intermodal_no_match;
      }

      for (auto const& fp : footpaths) {
        if (matches(tt, q.start_match_mode_, o.target(), fp.target()) &&
            is_better_or_eq(
                j.start_time_,
                leg_start_time -
                    (kFwd ? 1 : -1) * (o.duration() + fp.duration()))) {
          trace_rc_intermodal_fp_start_found;
          return journey::leg{SearchDir,
                              get_special_station(special_station::kStart),
                              leg_start_location,
                              j.start_time_,
                              j.start_time_ + (kFwd ? 1 : -1) * o.duration(),
                              o};
        } else {
          trace_rc_intermodal_fp_no_match;
        }
      }
    }
  } else {
    for (auto const& fp : footpaths) {
      if (is_journey_start(tt, q, fp.target()) &&
          fp_target_time != kInvalidDelta<SearchDir> &&
          start_matches(j_start_time + (kFwd ? 1 : -1) * fp.duration().count(),
                        fp_target_time)) {
        trace_rc_fp_start_found;
        return journey::leg{SearchDir,
                            fp.target(),
                            leg_start_location,
                            j.start_time_,
                            delta_to_unix(base, fp_target_time),
                            fp};
      } else {
        trace_rc_fp_start_no_match;
      }
    }
  }

  throw utl::fail("no valid journey start found");
}
template <direction SearchDir>
void reconstruct_journey(timetable const& tt,
                         rt_timetable const* rtt,
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

#if defined(NIGIRI_TRACE_RECONSTRUCT)
  auto const best = [&](std::uint32_t const k, location_idx_t const l) {
    return std::min(raptor_state.best_[to_idx(l)],
                    raptor_state.round_times_[k][to_idx(l)]);
  };
#endif

  auto const find_entry_in_prev_round =
      [&](unsigned const k, rt::run const& r, stop_idx_t const from_stop_idx,
          delta_t const time) -> std::optional<journey::leg> {
    auto const fr = rt::frun{tt, rtt, r};
    auto const n_stops = kFwd ? from_stop_idx + 1U : fr.size() - from_stop_idx;
    for (auto i = 1U; i != n_stops; ++i) {
      auto const stop_idx =
          static_cast<stop_idx_t>(kFwd ? from_stop_idx - i : from_stop_idx + i);
      auto const stp = fr[stop_idx];
      auto const l = stp.get_location_idx();

      if ((kFwd && !stp.in_allowed()) || (!kFwd && !stp.out_allowed())) {
        continue;
      }

      auto const event_time = unix_to_delta(
          base, stp.time(kFwd ? event_type::kDep : event_type::kArr));
      if (is_better_or_eq(raptor_state.round_times_[k - 1][to_idx(l)],
                          event_time) ||
          // special case: first stop with meta stations
          (k == 1 && q.start_match_mode_ == location_match_mode::kEquivalent &&
           is_journey_start(tt, q, l) &&
           start_matches(raptor_state.round_times_[k - 1][to_idx(l)],
                         event_time))) {
        trace_rc_transport_entry_found;
        return journey::leg{
            SearchDir,
            fr[stop_idx].get_location_idx(),
            fr[from_stop_idx].get_location_idx(),
            delta_to_unix(base, event_time),
            delta_to_unix(base, time),
            journey::run_enter_exit{r, stop_idx, from_stop_idx}};
      } else {
        trace_rc_transport_entry_not_possible;
      }
    }

    return std::nullopt;
  };

  auto const is_transport_active = [&](transport_idx_t const t,
                                       std::size_t const day) {
    if (rtt != nullptr) {
      return rtt->bitfields_[rtt->transport_traffic_days_[t]].test(day);
    } else {
      return tt.bitfields_[tt.transport_traffic_days_[t]].test(day);
    }
  };

  auto const get_route_transport =
      [&](unsigned const k, delta_t const time, route_idx_t const r,
          stop_idx_t const stop_idx) -> std::optional<journey::leg> {
    auto const [day, mam] = split_day_mam(base_day_idx, time);

    for (auto const t : tt.route_transport_ranges_[r]) {
      auto const event_mam =
          tt.event_mam(t, stop_idx, kFwd ? event_type::kArr : event_type::kDep);
      trace_rc_transport;

      if (minutes_after_midnight_t{event_mam.count() % 1440} != mam) {
        trace_rc_transport_mam_mismatch;
        continue;
      }

      auto const traffic_day = static_cast<std::size_t>(
          static_cast<int>(to_idx(day)) - event_mam.count() / 1440);
      if (!is_transport_active(t, traffic_day)) {
        trace_rc_transport_no_traffic;
        continue;
      }

      auto leg =
          find_entry_in_prev_round(k,
                                   {.t_ = transport{t, day_idx_t{traffic_day}},
                                    .stop_range_ = interval<stop_idx_t>{0, 0}},
                                   stop_idx, time);
      if (leg.has_value()) {
        return leg;
      }
      trace_rc_transport_not_found;
    }

    return std::nullopt;
  };

  auto const get_transport =
      [&](unsigned const k, location_idx_t const l,
          delta_t const time) -> std::optional<journey::leg> {
    trace_reconstruct(" time={}\n", delta_to_unix(base, time));

    if (rtt != nullptr) {
      for (auto const& rt_t : rtt->location_rt_transports_[l]) {
        auto const location_seq = rtt->rt_transport_location_seq_[rt_t];
        for (auto const [i, s] : utl::enumerate(location_seq)) {
          auto const stp = stop{s};
          auto const stop_idx = static_cast<stop_idx_t>(i);
          auto const fr = rt::frun{
              tt, rtt,
              rt::run{.stop_range_ = interval<stop_idx_t>{0, 0}, .rt_ = rt_t}};
          if (stp.location_idx() != l ||  //
              (kFwd && (i == 0U || !stp.out_allowed())) ||
              (!kFwd && (i == location_seq.size() - 1 || !stp.in_allowed())) ||
              delta_to_unix(base, time) !=
                  fr[stop_idx].time(kFwd ? event_type::kArr
                                         : event_type::kDep)) {
            continue;
          }

          auto leg = find_entry_in_prev_round(k, fr, stop_idx, time);
          if (leg.has_value()) {
            return leg;
          }
        }
      }
    }

    for (auto const& r : tt.location_routes_[l]) {
      auto const location_seq = tt.route_location_seq_[r];
      for (auto const [i, s] : utl::enumerate(location_seq)) {
        auto const stp = stop{s};
        if (stp.location_idx() != l ||  //
            (kFwd && (i == 0U || !stp.out_allowed())) ||
            (!kFwd && (i == location_seq.size() - 1 || !stp.in_allowed()))) {
          continue;
        }

        auto leg = get_route_transport(k, time, r, static_cast<stop_idx_t>(i));
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
        curr_time - (kFwd ? fp.duration() : -fp.duration()).count());
    trace_rc_check_fp;
    auto const transport_leg = get_transport(k, fp.target(), fp_start);

    if (transport_leg.has_value()) {
      trace_rc_legs_found;
      auto const fp_leg = journey::leg{SearchDir,
                                       fp.target(),
                                       l,
                                       delta_to_unix(base, fp_start),
                                       delta_to_unix(base, curr_time),
                                       fp};
      return std::pair{fp_leg, *transport_leg};
    } else {
      trace_reconstruct("nothing found\n");
    }
    return std::nullopt;
  };

  auto const get_legs =
      [&](unsigned const k,
          location_idx_t const l) -> std::pair<journey::leg, journey::leg> {
    auto const curr_time = raptor_state.round_times_[k][to_idx(l)];
    if (q.dest_match_mode_ == location_match_mode::kIntermodal &&
        k == j.transfers_ + 1U) {
      trace_reconstruct("  CHECKING INTERMODAL DEST\n");
      for (auto const& dest_offset : q.destination_) {
        std::optional<std::pair<journey::leg, journey::leg>> ret;
        for_each_meta(
            tt, location_match_mode::kIntermodal, dest_offset.target_,
            [&](location_idx_t const eq) {
              auto intermodal_dest =
                  check_fp(k, l, curr_time, {eq, dest_offset.duration_});
              if (intermodal_dest.has_value()) {
                trace_rc_intermodal_dest_match;
                intermodal_dest->first.uses_ =
                    offset{eq, dest_offset.duration_, dest_offset.type_};
                ret = std::move(intermodal_dest);
              } else {
                trace_rc_intermodal_dest_mismatch;
              }

              for (auto const& fp :
                   kFwd ? tt.locations_.footpaths_in_[q.prf_idx_][eq]
                        : tt.locations_.footpaths_out_[q.prf_idx_][eq]) {
                auto fp_intermodal_dest = check_fp(
                    k, l, curr_time,
                    {fp.target(), dest_offset.duration_ + fp.duration()});
                if (fp_intermodal_dest.has_value()) {
                  trace_rc_fp_intermodal_dest_match;
                  fp_intermodal_dest->first.uses_ =
                      offset{eq, fp.duration(), dest_offset.type_};
                  ret = std::move(fp_intermodal_dest);
                } else {
                  trace_rc_fp_intermodal_dest_mismatch;
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

    trace_reconstruct("CHECKING TRANSFER AT {}\n", location{tt, l});
    auto transfer_at_same_stop =
        check_fp(k, l, curr_time,
                 footpath{l, (k == j.transfers_ + 1U)
                                 ? 0_i8_minutes
                                 : tt.locations_.transfer_time_[l]});
    if (transfer_at_same_stop.has_value()) {
      return std::move(*transfer_at_same_stop);
    }

    trace_reconstruct("CHECKING FOOTPATHS OF {}\n", location{tt, l});
    auto const footpaths = kFwd ? tt.locations_.footpaths_in_[q.prf_idx_][l]
                                : tt.locations_.footpaths_out_[q.prf_idx_][l];
    for (auto const& fp : footpaths) {
      auto fp_legs = check_fp(k, l, curr_time, fp);
      if (fp_legs.has_value()) {
        return std::move(*fp_legs);
      }
    }

    throw utl::fail("reconstruction failed at k={}, t={}, stop={}, time={}", k,
                    j.transfers_, location{tt, l}, curr_time);
  };

  auto l = j.dest_;
  for (auto i = 0U; i <= j.transfers_; ++i) {
    auto const k = j.transfers_ + 1 - i;
    trace_reconstruct("RECONSTRUCT WITH k={}\n", k);
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

#if defined(NIGIRI_TRACE_RECUSTRUCT)
  j.print(std::cout, tt, true);
#endif
}

template <direction SearchDir>
void reconstruct_journey_gpu(timetable const& tt,
                         rt_timetable const* rtt,
                         query const& q,
                         mem const& state,
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

#if defined(NIGIRI_TRACE_RECONSTRUCT)
  auto const best = [&](std::uint32_t const k, location_idx_t const l) {
    return std::min(state.host_.best_[to_idx(l)],
                    state.host_.round_times_[k* state.host_.column_count_round_times_ +to_idx(l)]);
  };
#endif

  auto const find_entry_in_prev_round =
      [&](unsigned const k, rt::run const& r, stop_idx_t const from_stop_idx,
          delta_t const time) -> std::optional<journey::leg> {
    auto const fr = rt::frun{tt, rtt, r};
    auto const n_stops = kFwd ? from_stop_idx + 1U : fr.size() - from_stop_idx;
    for (auto i = 1U; i != n_stops; ++i) {
      auto const stop_idx =
          static_cast<stop_idx_t>(kFwd ? from_stop_idx - i : from_stop_idx + i);
      auto const stp = fr[stop_idx];
      auto const l = stp.get_location_idx();

      if ((kFwd && !stp.in_allowed()) || (!kFwd && !stp.out_allowed())) {
        continue;
      }

      auto const event_time = unix_to_delta(
          base, stp.time(kFwd ? event_type::kDep : event_type::kArr));
      if (is_better_or_eq(state.host_.round_times_[(k - 1)* state.host_.column_count_round_times_ +to_idx(l)],
                          event_time) ||
          // special case: first stop with meta stations
          (k == 1 && q.start_match_mode_ == location_match_mode::kEquivalent &&
           is_journey_start(tt, q, l) &&
           start_matches(state.host_.round_times_[(k - 1)* state.host_.column_count_round_times_ +to_idx(l)],
                         event_time))) {
        trace_rc_transport_entry_found;
        return journey::leg{
            SearchDir,
            fr[stop_idx].get_location_idx(),
            fr[from_stop_idx].get_location_idx(),
            delta_to_unix(base, event_time),
            delta_to_unix(base, time),
            journey::run_enter_exit{r, stop_idx, from_stop_idx}};
      } else {
        trace_rc_transport_entry_not_possible_gpu;
      }
    }

    return std::nullopt;
  };

  auto const is_transport_active = [&](transport_idx_t const t,
                                       std::size_t const day) {
    if (rtt != nullptr) {
      return rtt->bitfields_[rtt->transport_traffic_days_[t]].test(day);
    } else {
      return tt.bitfields_[tt.transport_traffic_days_[t]].test(day);
    }
  };

  auto const get_route_transport =
      [&](unsigned const k, delta_t const time, route_idx_t const r,
          stop_idx_t const stop_idx) -> std::optional<journey::leg> {
    auto const [day, mam] = split_day_mam(base_day_idx, time);

    for (auto const t : tt.route_transport_ranges_[r]) {
      auto const event_mam =
          tt.event_mam(t, stop_idx, kFwd ? event_type::kArr : event_type::kDep);
      trace_rc_transport;

      if (minutes_after_midnight_t{event_mam.count() % 1440} != mam) {
        trace_rc_transport_mam_mismatch;
        continue;
      }

      auto const traffic_day = static_cast<std::size_t>(
          static_cast<int>(to_idx(day)) - event_mam.count() / 1440);
      if (!is_transport_active(t, traffic_day)) {
        trace_rc_transport_no_traffic;
        continue;
      }

      auto leg =
          find_entry_in_prev_round(k,
                                   {.t_ = transport{t, day_idx_t{traffic_day}},
                                    .stop_range_ = interval<stop_idx_t>{0, 0}},
                                   stop_idx, time);
      if (leg.has_value()) {
        return leg;
      }
      trace_rc_transport_not_found;
    }

    return std::nullopt;
  };

  auto const get_transport =
      [&](unsigned const k, location_idx_t const l,
          delta_t const time) -> std::optional<journey::leg> {
    trace_reconstruct(" time={}\n", delta_to_unix(base, time));

    if (rtt != nullptr) {
      for (auto const& rt_t : rtt->location_rt_transports_[l]) {
        auto const location_seq = rtt->rt_transport_location_seq_[rt_t];
        for (auto const [i, s] : utl::enumerate(location_seq)) {
          auto const stp = stop{s};
          auto const stop_idx = static_cast<stop_idx_t>(i);
          auto const fr = rt::frun{
              tt, rtt,
              rt::run{.stop_range_ = interval<stop_idx_t>{0, 0}, .rt_ = rt_t}};
          if (stp.location_idx() != l ||  //
              (kFwd && (i == 0U || !stp.out_allowed())) ||
              (!kFwd && (i == location_seq.size() - 1 || !stp.in_allowed())) ||
              delta_to_unix(base, time) !=
                  fr[stop_idx].time(kFwd ? event_type::kArr
                                         : event_type::kDep)) {
            continue;
          }

          auto leg = find_entry_in_prev_round(k, fr, stop_idx, time);
          if (leg.has_value()) {
            return leg;
          }
        }
      }
    }

    for (auto const& r : tt.location_routes_[l]) {
      auto const location_seq = tt.route_location_seq_[r];
      for (auto const [i, s] : utl::enumerate(location_seq)) {
        auto const stp = stop{s};
        if (stp.location_idx() != l ||  //
            (kFwd && (i == 0U || !stp.out_allowed())) ||
            (!kFwd && (i == location_seq.size() - 1 || !stp.in_allowed()))) {
          continue;
        }

        auto leg = get_route_transport(k, time, r, static_cast<stop_idx_t>(i));
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
        curr_time - (kFwd ? fp.duration() : -fp.duration()).count());
    trace_rc_check_fp;
    auto const transport_leg = get_transport(k, fp.target(), fp_start);

    if (transport_leg.has_value()) {
      trace_rc_legs_found;
      auto const fp_leg = journey::leg{SearchDir,
                                       fp.target(),
                                       l,
                                       delta_to_unix(base, fp_start),
                                       delta_to_unix(base, curr_time),
                                       fp};
      return std::pair{fp_leg, *transport_leg};
    } else {
      trace_reconstruct("nothing found\n");
    }
    return std::nullopt;
  };

  auto const get_legs =
      [&](unsigned const k,
          location_idx_t const l) -> std::pair<journey::leg, journey::leg> {
    auto const curr_time = state.host_.round_times_[k* state.host_.column_count_round_times_ +to_idx(l)];
    if (q.dest_match_mode_ == location_match_mode::kIntermodal &&
        k == j.transfers_ + 1U) {
      trace_reconstruct("  CHECKING INTERMODAL DEST\n");
      for (auto const& dest_offset : q.destination_) {
        std::optional<std::pair<journey::leg, journey::leg>> ret;
        for_each_meta(
            tt, location_match_mode::kIntermodal, dest_offset.target_,
            [&](location_idx_t const eq) {
              auto intermodal_dest =
                  check_fp(k, l, curr_time, {eq, dest_offset.duration_});
              if (intermodal_dest.has_value()) {
                trace_rc_intermodal_dest_match;
                intermodal_dest->first.uses_ =
                    offset{eq, dest_offset.duration_, dest_offset.type_};
                ret = std::move(intermodal_dest);
              } else {
                trace_rc_intermodal_dest_mismatch;
              }

              for (auto const& fp :
                   kFwd ? tt.locations_.footpaths_in_[q.prf_idx_][eq]
                        : tt.locations_.footpaths_out_[q.prf_idx_][eq]) {
                auto fp_intermodal_dest = check_fp(
                    k, l, curr_time,
                    {fp.target(), dest_offset.duration_ + fp.duration()});
                if (fp_intermodal_dest.has_value()) {
                  trace_rc_fp_intermodal_dest_match;
                  fp_intermodal_dest->first.uses_ =
                      offset{eq, fp.duration(), dest_offset.type_};
                  ret = std::move(fp_intermodal_dest);
                } else {
                  trace_rc_fp_intermodal_dest_mismatch_gpu;
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

    trace_reconstruct("CHECKING TRANSFER AT {}\n", location{tt, l});
    auto transfer_at_same_stop =
        check_fp(k, l, curr_time,
                 footpath{l, (k == j.transfers_ + 1U)
                                 ? 0_i8_minutes
                                 : tt.locations_.transfer_time_[l]});
    if (transfer_at_same_stop.has_value()) {
      return std::move(*transfer_at_same_stop);
    }

    trace_reconstruct("CHECKING FOOTPATHS OF {}\n", location{tt, l});
    auto const footpaths = kFwd ? tt.locations_.footpaths_in_[q.prf_idx_][l]
                                : tt.locations_.footpaths_out_[q.prf_idx_][l];
    for (auto const& fp : footpaths) {
      auto fp_legs = check_fp(k, l, curr_time, fp);
      if (fp_legs.has_value()) {
        return std::move(*fp_legs);
      }
    }

    throw utl::fail("reconstruction failed at k={}, t={}, stop={}, time={}", k,
                    j.transfers_, location{tt, l}, curr_time);
  };

  auto l = j.dest_;
  for (auto i = 0U; i <= j.transfers_; ++i) {
    auto const k = j.transfers_ + 1 - i;
    trace_reconstruct("RECONSTRUCT WITH k={}\n", k);
    auto [fp_leg, transport_leg] = get_legs(k, l);
    l = kFwd ? transport_leg.from_ : transport_leg.to_;
    j.add(std::move(fp_leg));
    j.add(std::move(transport_leg));
  }

  auto init_fp = find_start_footpath_gpu<SearchDir>(tt, q, j, state, base);
  if (init_fp.has_value()) {
    j.add(std::move(*init_fp));
  }

  if constexpr (kFwd) {
    std::reverse(begin(j.legs_), end(j.legs_));
  }

#if defined(NIGIRI_TRACE_RECUSTRUCT)
  j.print(std::cout, tt, true);
#endif
}
template void reconstruct_journey<direction::kForward>(timetable const&,
                                                       rt_timetable const*,
                                                       query const&,
                                                       raptor_state const&,
                                                       journey&,
                                                       date::sys_days const,
                                                       day_idx_t const);

template void reconstruct_journey<direction::kBackward>(timetable const&,
                                                        rt_timetable const*,
                                                        query const&,
                                                        raptor_state const&,
                                                        journey&,
                                                        date::sys_days const,
                                                        day_idx_t const);

template void reconstruct_journey_gpu<direction::kForward>(timetable const&,
                                                       rt_timetable const*,
                                                       query const&,
                                                       mem const&,
                                                       journey&,
                                                       date::sys_days const,
                                                       day_idx_t const);

template void reconstruct_journey_gpu<direction::kBackward>(timetable const&,
                                                        rt_timetable const*,
                                                        query const&,
                                                        mem const&,
                                                        journey&,
                                                        date::sys_days const,
                                                        day_idx_t const);

}  // namespace nigiri::routing
