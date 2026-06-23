/*
 * This file is based on the mcraptor from PumarPap
 * https://github.com/motis-project/nigiri/pull/183
 */

#pragma once

#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/search.h"
#include "nigiri/timetable.h"

namespace nigiri::routing::da {

template <direction SearchDir>
routing_result mcraptor_search(
    timetable const& tt,
    rt_timetable const* rtt,
    search_state& s_state,
    raptor_state& r_state,
    query q,
    std::optional<std::chrono::seconds> timeout = std::nullopt,
    std::vector<std::vector<std::pair<int, double>>> arr_dist = {});

}  // namespace nigiri::routing::da
