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

// Outside of the routing a real-time virtual location is its platform: maps
// the locations of every leg and the destination of a reconstructed journey.
void to_platforms(rt_timetable const*, journey&);

template <direction SearchDir>
void reconstruct_journey(timetable const&,
                         rt_timetable const*,
                         query const&,
                         raptor_state const&,
                         journey&,
                         date::sys_days const base,
                         day_idx_t const base_day_idx);

void optimize_footpaths(timetable const&,
                        rt_timetable const*,
                        query const&,
                        journey&);

void specify_td_offsets(query const&, journey&);

}  // namespace nigiri::routing
