#include "nigiri/routing/a_star/a_star_search.h"

namespace nigiri::routing {

routing_result a_star_search(timetable const& tt,
                             search_state& search_state,
                             a_star_state& algo_state,
                             query q) {
  return routing::search<direction::kForward, a_star<true>>{
      tt, nullptr, search_state, algo_state, std::move(q)}
      .execute();
}

}  // namespace nigiri::routing