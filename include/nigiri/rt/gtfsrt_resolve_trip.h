#include "gtfs-realtime.pb.h"

#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

#include "gtfs-realtime.pb.h"

namespace nigiri::rt {

struct trip {};

std::optional<transport> gtfsrt_resolve_trip(
    date::sys_days const today,
    timetable const&,
    rt_timetable&,
    source_idx_t,
    transit_realtime::TripDescriptor const&);

}  // namespace nigiri::rt