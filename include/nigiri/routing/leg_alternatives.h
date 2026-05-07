#pragma once

#include <vector>

#include "nigiri/routing/journey.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
struct rt_timetable;
}  // namespace nigiri

namespace nigiri::routing {

// Returns all transports that cover the same leg as the given one:
// same boarding location, same alighting location, same departure time,
// same arrival time — but on a different transport (e.g. Flügelzug /
// coupled trains on a parallel route).
//
// The original transport (identified by `exclude`) is never included in the
// result. Results are stored directly in `leg.alternatives_` after routing
// (see reconstruct.cc) so that both rRAPTOR and PONG produce them consistently.
std::vector<journey::leg_alternative> find_equivalent_transports(
    timetable const&,
    rt_timetable const*,
    profile_idx_t prf_idx,
    location_idx_t boarding_loc,
    location_idx_t alighting_loc,
    unixtime_t dep_time,
    unixtime_t arr_time,
    transport_idx_t exclude);

}  // namespace nigiri::routing
