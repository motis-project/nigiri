#pragma once

#include <optional>
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

enum class side { kBoarding, kAlighting };

std::optional<journey::leg> lookup_offset(location_idx_t,
                                          unixtime_t,
                                          side,
                                          std::vector<offset> const&,
                                          td_offsets_t const&);

std::optional<journey::leg> lookup_footpath(location_idx_t,
                                            unixtime_t,
                                            side,
                                            timetable const&,
                                            rt_timetable const*,
                                            query const&,
                                            std::vector<offset> const& offs,
                                            location_match_mode,
                                            bool use_footpaths);

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