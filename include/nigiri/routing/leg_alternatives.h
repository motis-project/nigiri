#pragma once

#include <vector>

#include "nigiri/routing/journey.h"
#include "nigiri/routing/query.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing {

// Builds a synthetic query that searches for a direct connection (single
// transit leg sandwiched between footpaths) between `from` and `to`,
// inheriting the relevant access settings (profile, transfer time, mode and
// transport-property constraints) from `q`.
query make_alternative_query(timetable const&,
                             rt_timetable const*,
                             query const&,
                             location_idx_t from,
                             location_idx_t to);

// Returns up to `max_alternatives` alternatives
// [ingress footpath, transit, egress footpath] for the transit leg at
// `leg_idx` in `j`, bounded by the surrounding journey context:
//   - first transit leg with a successor: iterate backward from the next
//     leg's departure to collect the LATEST alternatives still arriving in
//     time (only those a passenger could realistically choose instead).
//   - intermediate transit leg: must depart after the predecessor's arrival
//     and arrive before the successor's departure.
//   - first/last transit leg without one of the bounds: the missing bound
//     defaults to the journey's start time / the original transit's
//     arrival, yielding "earlier alternatives".
// The original transport is filtered out of the result.
// `max_alternatives == 0` returns an empty vector without searching.
std::vector<journey> get_leg_alternatives(timetable const&,
                                          rt_timetable const*,
                                          query const&,
                                          journey const&,
                                          std::size_t leg_idx,
                                          std::size_t max_alternatives);

}  // namespace nigiri::routing
