#pragma once

#include "nigiri/routing/search.h"
#include "nigiri/routing/tb/query_engine.h"
#include "nigiri/timetable.h"

namespace nigiri::routing::tb {

routing_result tb_search(timetable const& tt,
                         search_state& s_state,
                         query_state& r_state,
                         query q);

}  // namespace nigiri::routing::tb