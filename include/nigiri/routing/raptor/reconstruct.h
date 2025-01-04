#pragma once

#include "nigiri/routing/raptor/raptor_variant.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
struct rt_timetable;
}  // namespace nigiri

namespace nigiri::routing {

struct query;
struct search_state;
struct raptor_state;
struct journey;

template <direction SearchDir>
void reconstruct_journey(timetable const&,
                         rt_timetable const*,
                         query const&,
                         raptor_state const&,
                         journey&,
                         date::sys_days const base,
                         day_idx_t base_day_idx,
                         raptor_variant);

template <direction SearchDir>
void optimize_footpaths(timetable const&,
                        rt_timetable const*,
                        query const&,
                        journey&);

}  // namespace nigiri::routing
