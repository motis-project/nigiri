#pragma once

#include <vector>

#include "utl/generator.h"

#include "nigiri/common/interval.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/query.h"
#include "nigiri/types.h"

namespace nigiri {

struct timetable;
struct rt_timetable;

namespace routing {

template <direction Dir>
utl::generator<std::vector<journey::leg>> get_direct_journeys(
    timetable const&, rt_timetable const*, query const&, unixtime_t time);

template <direction Dir>
void enrich_with_slow_direct(timetable const&,
                             rt_timetable const*,
                             query const&,
                             interval<unixtime_t> const&,
                             pareto_set<journey>& results);

}  // namespace routing

}  // namespace nigiri