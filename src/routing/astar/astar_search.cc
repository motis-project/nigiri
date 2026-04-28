#include "nigiri/routing/astar/astar_search.h"

namespace nigiri::routing::astar {

routing_result astar_search(timetable const& tt,
                            search_state& search_state,
                            astar_state& algo_state,
                            query q,
                            std::uint32_t const astar_transfer_penalty) {
  algo_state.astar_transfer_penalty_ = astar_transfer_penalty;
  return routing::search<direction::kForward, astar::astar_engine<true>>{
      tt, nullptr, search_state, algo_state, std::move(q)}
      .execute();
}

}  // namespace nigiri::routing::astar