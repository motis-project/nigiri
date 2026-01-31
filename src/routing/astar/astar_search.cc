#pragma once
#include "nigiri/routing/astar/astar_search.h"

namespace nigiri::routing::astar {

routing_result astar_search(timetable const& tt,
                            search_state& search_state,
                            astar_state& algo_state,
                            query q) {
  return routing::search<direction::kForward, astar::astar_engine<true>>{
      tt, nullptr, search_state, algo_state, std::move(q)}
      .execute();
}

}  // namespace nigiri::routing::astar