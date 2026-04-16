#pragma once

#include <vector>

#include "nigiri/common/interval.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/types.h"

namespace nigiri {

struct timetable;
struct rt_timetable;

namespace routing {

void get_direct(timetable const&,
                rt_timetable const*,
                location_idx_t const from,
                location_idx_t const to,
                routing::query const&,
                interval<unixtime_t>,
                direction,
                hash_set<std::pair<location_idx_t, location_idx_t>>& done,
                std::vector<journey>& direct);

void enrich_with_slow_direct(timetable const&,
                             rt_timetable const*,
                             query const&,
                             interval<unixtime_t>,
                             direction,
                             pareto_set<journey>& results);
}  // namespace routing

}  // namespace nigiri