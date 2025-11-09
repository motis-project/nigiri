#pragma once

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
                         day_idx_t const base_day_idx);

template <direction SearchDir>
void optimize_footpaths(timetable const&,
                        rt_timetable const*,
                        query const&,
                        journey&);

template <direction SearchDir>
void specify_td_offsets(query const&, journey&);

}  // namespace nigiri::routing
