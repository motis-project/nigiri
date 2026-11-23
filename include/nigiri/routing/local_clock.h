#pragma once

#include <vector>

#include "nigiri/types.h"

namespace nigiri {
struct timetable;
struct rt_timetable;
}  // namespace nigiri

namespace nigiri::routing {

// The minute of the local day of an event at a stop, in the stop's timezone,
// else its trip's agency's (as rt::run_stop::get_tz), else UTC. A timezone's
// offset is cached for the UTC range it holds.
struct local_clock {
  local_clock(timetable const&, rt_timetable const*);

  int mam(unixtime_t, location_idx_t, timezone_idx_t trip_tz);
  timezone_idx_t trip_tz(transport_idx_t) const;
  timezone_idx_t trip_tz(rt_transport_idx_t) const;

private:
  struct offset {
    unixtime_t from_{}, to_{};
    duration_t offset_{};
  };

  timetable const& tt_;
  rt_timetable const* rtt_;
  std::vector<offset> offsets_;
};

}  // namespace nigiri::routing
