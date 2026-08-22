#pragma once

#include <optional>
#include <vector>

#include "nigiri/routing/journey.h"
#include "nigiri/routing/query.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing {

query make_alternative_query(timetable const&,
                             rt_timetable const*,
                             query const&,
                             location_idx_t from,
                             location_idx_t to);

std::vector<journey> get_leg_alternatives(
    timetable const&,
    rt_timetable const*,
    query const& direct_query,
    direction search_dir,
    unixtime_t anchor_time,
    std::optional<unixtime_t> max_arrival,
    journey::run_enter_exit const& original,
    std::size_t max_alternatives);

std::vector<journey> get_leg_alternatives(timetable const&,
                                          rt_timetable const*,
                                          query const&,
                                          journey const&,
                                          std::size_t leg_idx,
                                          std::size_t max_alternatives);

}  // namespace nigiri::routing
