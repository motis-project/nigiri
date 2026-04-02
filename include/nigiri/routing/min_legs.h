#pragma once

#include "nigiri/routing/query.h"
#include "nigiri/timetable.h"

#include "raptor/raptor_state.h"

namespace nigiri::routing {

// SearchDir refers to the direction of the main routing query
// forward: finds the minimum number of legs backward from the destination
// backward: finds the minimum number of legs forward from the destination
template<direction SearchDir>
void min_rounds(timetable const&, query const&, raptor_state&, std::vector<std::uint8_t>& min_legs);

} // namespace nigiri::routing