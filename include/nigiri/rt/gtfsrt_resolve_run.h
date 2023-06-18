#pragma once

#include "nigiri/rt/rt_timetable.h"
#include "nigiri/rt/run.h"
#include "nigiri/timetable.h"

#include "gtfsrt/gtfs-realtime.pb.h"

namespace nigiri::rt {

run gtfsrt_resolve_run(date::sys_days const today,
                       timetable const&,
                       rt_timetable&,
                       source_idx_t,
                       transit_realtime::TripDescriptor const&);

}  // namespace nigiri::rt