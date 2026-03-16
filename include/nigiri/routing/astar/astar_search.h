#pragma once

#include "nigiri/routing/astar/astar_engine.h"
#include "nigiri/routing/search.h"
#include "nigiri/timetable.h"

namespace nigiri::routing::astar {

routing_result astar_search(timetable const& tt,
                            search_state& s_state,
                            astar_state& a_state,
                            query q,
                            std::uint32_t astar_transfer_penalty);

}  // namespace nigiri::routing::astar