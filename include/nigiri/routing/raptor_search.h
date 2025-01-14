#pragma once

#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/search.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

template <typename AlgoState>
routing_result<raptor_stats> raptor_search(
    timetable const&,
    rt_timetable const*,
    search_state&,
    AlgoState&,
    query,
    direction search_dir,
    std::optional<std::chrono::seconds> timeout = std::nullopt);

}  // namespace nigiri::routing
