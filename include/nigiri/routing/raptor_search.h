#pragma once

#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/search.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

template <typename AlgoState>
routing_result raptor_search(
    timetable const&,
    rt_timetable const*,
    search_state&,
    AlgoState&,
    query,
    direction search_dir,
    std::optional<std::chrono::seconds> timeout = std::nullopt);

// combined scheduled+rt search: journeys of both worlds in one sweep,
// tagged via journey::slot_ (0 = scheduled-only, 1 = rt);
// copy_on_diverge stores the rt world as a sparse overlay over the
// scheduled base plane
routing_result raptor_search_schedrt(
    timetable const&,
    rt_timetable const*,
    search_state&,
    raptor_state&,
    query,
    direction search_dir,
    std::optional<std::chrono::seconds> timeout = std::nullopt,
    bool copy_on_diverge = false);

}  // namespace nigiri::routing
