#pragma once

#include "nigiri/routing/dijkstra.h"
#include "nigiri/routing/query.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing {

constexpr auto const kUnreachableDirect =
    duration_t{std::numeric_limits<duration_t::rep>::max()};

duration_t get_fastest_direct(
    timetable const&,
    query const&,
    direction const,
    label::dist_t const max_dist = std::numeric_limits<label::dist_t>::max());

}  // namespace nigiri::routing
