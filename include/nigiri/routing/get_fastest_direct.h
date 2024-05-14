#pragma once

#include "nigiri/routing/dijkstra.h"
#include "nigiri/routing/query.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing {

duration_t get_fastest_direct(
    timetable const&,
    query const&,
    direction const,
    label::dist_t const max_dist = std::numeric_limits<label::dist_t>::max());

}  // namespace nigiri::routing
