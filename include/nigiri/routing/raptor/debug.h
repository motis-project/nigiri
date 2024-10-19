#pragma once

#include "fmt/core.h"

//#define NIGIRI_TRACING
#if defined(NIGIRI_TRACING)

//#define NIGIRI_RAPTOR_TRACING_ONLY_UPDATES
//#define NIGIRI_RAPTOR_INTERVAL_TRACING

#define trace_upd(...) fmt::print(__VA_ARGS__)

#ifdef NIGIRI_RAPTOR_TRACING_ONLY_UPDATES
#define trace(...)
#else
#define trace(...) fmt::print(__VA_ARGS__)
#endif

#define NIGIRI_TRACE_RECONSTRUCT
#ifdef NIGIRI_TRACE_RECONSTRUCT
#define trace_reconstruct(...) fmt::print(__VA_ARGS__)
#else
#define trace_reconstruct(...)
#endif

#define trace_print_state(...)         \
  fmt::print(__VA_ARGS__);             \
  state_.print(tt_, base(), kInvalid); \
  fmt::print("\n")

#define trace_print_state_after_round() \
  trace_print_state("STATE AFTER ROUND {}\n", k)

#define trace_print_init_state(...) trace_print_state("INIT\n")

#define trace_rc_find_start_footpath \
  trace_reconstruct("find_start_footpath()\n")

#define trace_rc_direct_start_found                                     \
  trace_reconstruct(                                                    \
      "  leg_start_location={} is a start, time matches ({}) - done\n", \
      location{tt, leg_start_location}, j.start_time_)

#define trace_rc_direct_start_excluded                                         \
  trace_reconstruct(                                                           \
      "  direct start excluded intermodal_start={}, is_journey_start({})={}, " \
      "leg_start_time={}, journey_start_time={}\n",                            \
      q.start_match_mode_ == location_match_mode::kIntermodal,                 \
      location{tt, leg_start_location},                                        \
      is_journey_start(tt, q, leg_start_location), leg_start_time,             \
      j.start_time_)

#define trace_rc_checking_start_fp                                           \
  trace_reconstruct(                                                         \
      "  j_start={} is not a start meta={}, start={}, checking footpaths\n", \
      location{tt, leg_start_location},                                      \
      q.start_match_mode_ == location_match_mode::kEquivalent,               \
      is_journey_start(tt, q, leg_start_location))

#define trace_rc_intermodal_start_found                                   \
  trace_reconstruct(                                                      \
      "    --> start: INTERMODAL DIRECT START -> {}  leg_start_time={}, " \
      "j_start_time={}, offset={}\n",                                     \
      location{tt, o.target()}, leg_start_time, j.start_time_, o.duration())

#define trace_rc_intermodal_no_match                                    \
  trace_reconstruct(                                                    \
      "    no start: INTERMODAL DIRECT START -> {}  matches={}, "       \
      "leg_start_location={}, leg_start_time={}, j_start_time={}, "     \
      "offset={}\n",                                                    \
      location{tt, o.target()},                                         \
      matches(tt, q.start_match_mode_, o.target(), leg_start_location), \
      location{tt, leg_start_location}, leg_start_time, j.start_time_,  \
      o.duration())

#define trace_rc_intermodal_fp_start_found                                   \
  trace_reconstruct(                                                         \
      "    --> start: FP+INTERMODAL START -> {}  leg_start_time={}, "        \
      "j_start_time={}, offset={}, footpath=({}, {})\n",                     \
      location{tt, o.target()}, leg_start_time, j.start_time_, o.duration(), \
      fp.duration(), location{tt, fp.target()})

#define trace_rc_intermodal_fp_no_match                                        \
  trace_reconstruct(                                                           \
      "    no start: FP+INTERMODAL START -> {}  matches={}, "                  \
      "leg_start_location={}, leg_start_time={}, j_start_time={}, offset={}, " \
      "footpath=({}, {})\n",                                                   \
      location{tt, o.target()},                                                \
      matches(tt, q.start_match_mode_, o.target(), leg_start_location),        \
      location{tt, leg_start_location}, leg_start_time, j.start_time_,         \
      o.duration(), fp.duration(), location{tt, fp.target()})

#define trace_rc_fp_start_found                                            \
  trace_reconstruct(                                                       \
      "    --> from={}, j_start={}, journey_start={}, fp_target_time={}, " \
      "duration={}\n",                                                     \
      location{tt, fp.target()}, location{tt, leg_start_location},         \
      j.start_time_, fp_target_time, fp.duration())

#define trace_rc_fp_start_no_match                                        \
  trace_reconstruct(                                                      \
      "    no start: {} -> {}  is_journey_start(fp.target())={}, "        \
      "fp_start_time={}, j_start_time={}, fp_duration={}\n",              \
      location{tt, fp.target()}, location{tt, leg_start_location},        \
      is_journey_start(tt, q, fp.target()), fp_target_time, j_start_time, \
      fp.duration().count())

#define trace_rc_transport                                                    \
  trace_reconstruct(                                                          \
      "  CHECKING TRANSPORT name={}, dbg={}, stop={}, time={} (day={}, "      \
      "mam={}), traffic_day={}, event_mam={}\n",                              \
      tt.transport_name(t), tt.dbg(t),                                        \
      location{tt, stop{tt.route_location_seq_[r][stop_idx]}.location_idx()}, \
      delta_to_unix(base, time), day, mam,                                    \
      static_cast<int>(to_idx(day)) - event_mam.count() / 1440, event_mam)

#define trace_rc_transport_mam_mismatch                              \
  trace_reconstruct(                                                 \
      "    -> ev_mam mismatch: transport_ev={} vs footpath = {}\n ", \
      duration_t{event_mam.count()}, duration_t{mam})

#define trace_rc_transport_no_traffic \
  trace_reconstruct("    -> no traffic on day {}\n ", traffic_day)

#define trace_rc_transport_not_found \
  trace_reconstruct("    -> no entry found\n")

#define trace_rc_transport_entry_not_possible                              \
  trace_reconstruct(                                                       \
      "      ENTRY NOT POSSIBLE AT {}: k={} k-1={}, best_at_stop=min({}, " \
      "{})={}={} > event_time={}={}\n",                                    \
      location{tt, l}, k, k - 1, raptor_state.best_[to_idx(l)],            \
      raptor_state.round_times_[k - 1][to_idx(l)], best(k - 1, l),         \
      delta_to_unix(base, best(k - 1, l)), event_time,                     \
      fr[stop_idx].time(kFwd ? event_type::kDep : event_type::kArr))

#define trace_rc_transport_entry_not_possible_gpu                              \
  trace_reconstruct(                                                       \
      "      ENTRY NOT POSSIBLE AT {}: k={} k-1={}, best_at_stop=min({}, " \
      "{})={}={} > event_time={}={}\n",                                    \
      location{tt, l}, k, k - 1, state.host_.best_[to_idx(l)],            \
      state.host_.round_times_[(k - 1)* state.host_.column_count_round_times_ +to_idx(l)], best(k - 1, l),         \
      delta_to_unix(base, best(k - 1, l)), event_time,                     \
      fr[stop_idx].time(kFwd ? event_type::kDep : event_type::kArr))

#define trace_rc_transport_entry_found                                 \
  trace_reconstruct(                                                   \
      "      FOUND ENTRY AT name={}, dbg={}, location={}: {} <= {}\n", \
      fr.name(), fr.dbg(), location{tt, l},                            \
      delta_to_unix(base, best(k - 1, l)), delta_to_unix(base, event_time))

#define trace_rc_fp_intermodal_dest_mismatch                            \
  trace_reconstruct(                                                    \
      "  BAD intermodal+footpath dest offset: {}@{} --{}--> "           \
      "{}@{} --{}--> END@{} (type={})\n",                               \
      location{tt, fp.target()},                                        \
      raptor_state.round_times_[k][to_idx(fp.target())], fp.duration(), \
      location{tt, eq}, raptor_state.round_times_[k][to_idx(eq)],       \
      dest_offset.duration_, curr_time, dest_offset.type_)

#define trace_rc_fp_intermodal_dest_mismatch_gpu                        \
  trace_reconstruct(                                                    \
      "  BAD intermodal+footpath dest offset: {}@{} --{}--> "           \
      "{}@{} --{}--> END@{} (type={})\n",                               \
      location{tt, fp.target()},                                        \
      state.host_.round_times_[k* state.host_.column_count_round_times_ + to_idx(fp.target())], fp.duration(), \
      location{tt, eq}, state.host_.round_times_[k* state.host_.column_count_round_times_ + to_idx(eq)],       \
      dest_offset.duration_, curr_time, dest_offset.type_)

#define trace_rc_fp_intermodal_dest_match                        \
  trace_reconstruct(                                             \
      "  found intermodal+footpath dest offset END [{}] -> {}: " \
      "offset={}\n",                                             \
      curr_time, location{tt, fp.target()}, fp.duration())

#define trace_rc_intermodal_dest_mismatch                                 \
  trace_reconstruct("  BAD intermodal dest offset: END [{}] -> {}: {}\n", \
                    curr_time, location{tt, dest_offset.target_},         \
                    dest_offset.duration_)

#define trace_rc_intermodal_dest_match                  \
  trace_reconstruct(                                    \
      "  found intermodal dest offset END [{}] -> {}: " \
      "offset={}\n",                                    \
      curr_time, location{tt, dest_offset.target_}, dest_offset.duration_)

#define trace_rc_legs_found                                            \
  trace_reconstruct("found:\n");                                       \
  transport_leg->print(std::cout, tt, rtt, 1, true);                   \
  trace_reconstruct(" fp leg: {} {} --{}--> {} {}\n", location{tt, l}, \
                    delta_to_unix(base, fp_start), fp.duration(),      \
                    location{tt, fp.target()}, delta_to_unix(base, curr_time))

#define trace_rc_check_fp                                                   \
  trace_reconstruct(                                                        \
      "round {}: searching for transports at {} with curr_time={} --{}--> " \
      "fp_start={}\n ",                                                     \
      k, location{tt, fp.target()}, delta_to_unix(base, curr_time),         \
      fp.duration(), delta_to_unix(base, fp_start))

#else
#define trace_reconstruct(...)
#define trace_rc_find_start_footpath
#define trace_rc_direct_start_found
#define trace_rc_direct_start_excluded
#define trace_rc_checking_start_fp
#define trace_rc_intermodal_start_found
#define trace_rc_intermodal_no_match
#define trace_rc_intermodal_fp_start_found
#define trace_rc_intermodal_fp_no_match
#define trace_rc_fp_start_found
#define trace_rc_fp_start_no_match
#define trace_rc_transport
#define trace_rc_transport_mam_mismatch
#define trace_rc_transport_no_traffic
#define trace_rc_transport_not_found
#define trace_rc_transport_entry_not_possible
#define trace_rc_transport_entry_not_possible_gpu
#define trace_rc_transport_entry_found
#define trace_rc_fp_intermodal_dest_mismatch
#define trace_rc_fp_intermodal_dest_mismatch_gpu
#define trace_rc_fp_intermodal_dest_match
#define trace_rc_intermodal_dest_mismatch
#define trace_rc_intermodal_dest_match
#define trace_rc_legs_found
#define trace_rc_check_fp

#define trace_print_state_after_round()
#define trace_print_init_state()
#define trace_upd(...)
#define trace(...)
#endif