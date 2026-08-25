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

// combined scheduled+rt search: journeys carry their world in journey::slot_,
// slot 0 reconstructs against the static timetable (rtt ignored), slot 1
// against the rt timetable
template <direction SearchDir>
void reconstruct_journey_schedrt(timetable const&,
                                 rt_timetable const*,
                                 query const&,
                                 raptor_state const&,
                                 journey&,
                                 date::sys_days const base,
                                 day_idx_t const base_day_idx);

// copy-on-diverge layout, rt world (journey::slot_ == 1); the scheduled
// world reconstructs through the plain path on the base plane
template <direction SearchDir>
void reconstruct_journey_schedrt_cod(timetable const&,
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
