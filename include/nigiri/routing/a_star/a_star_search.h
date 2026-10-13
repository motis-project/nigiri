#pragma once

#include "nigiri/routing/a_star/a_star.h"
#include "nigiri/routing/search.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

routing_result a_star_search(timetable const& tt,
                             search_state& search_state,
                             a_star_state& algo_state,
                             query q);

}  // namespace nigiri::routing