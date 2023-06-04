#include "gtfs-realtime.pb.h"

#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

#include "gtfs-realtime.pb.h"

namespace nigiri::rt {

struct trip {};

trip gtfsrt_resolve_trip(timetable const& tt,
                         rt_timetable& rtt,
                         source_idx_t const src,
                         transit_realtime::TripDescriptor const& td);

}  // namespace nigiri::rt