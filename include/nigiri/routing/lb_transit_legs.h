#pragma once

#include "nigiri/routing/query.h"
#include "nigiri/timetable.h"

#include "raptor/raptor_state.h"

namespace nigiri::routing {

// SearchDir refers to the direction of the main routing query
// fwd: finds the minimum number of transit legs backward from the destination
// bwd: finds the minimum number of transit legs forward from the destination
template <direction SearchDir>
void lb_transit_legs(timetable const&,
                     query const&,
                     raptor_state&,
                     std::vector<std::uint8_t>& lb);

}  // namespace nigiri::routing