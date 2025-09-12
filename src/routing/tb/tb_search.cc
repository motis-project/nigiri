#include "nigiri/routing/tb/tb_search.h"

namespace nigiri::routing::tb {

routing_result tb_search(timetable const& tt,
                         search_state& search_state,
                         query_state& algo_state,
                         query q) {
  return routing::search<direction::kForward, tb::query_engine<true>>{
      tt, nullptr, search_state, algo_state, std::move(q)}
      .execute();
}

}  // namespace nigiri::routing::tb