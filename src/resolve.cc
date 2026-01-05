#include "nigiri/resolve.h"

#include "nigiri/rt/gtfsrt_resolve_run.h"

using namespace nigiri::rt;

namespace nigiri {

frun resolve(timetable const& tt,
             rt_timetable const* rtt,
             std::string_view trip_id,
             std::string_view start_date,
             std::string_view start_time) {
  auto td = transit_realtime::TripDescriptor{};
  td.set_trip_id(trip_id);
  if (!start_date.empty()) {
    td.set_start_date(start_date);
  }
  if (!start_time.empty()) {
    td.set_start_time(start_time);
  }
  auto const [r, _] = gtfsrt_resolve_run(date::sys_days{}, tt, rtt, {}, td);
  utl::verify(r.valid(),
              "trip not found: trip_id={}, start_date={}, start_time={}",
              trip_id, start_date, start_time);
  return frun{tt, rtt, r};
}

}  // namespace nigiri