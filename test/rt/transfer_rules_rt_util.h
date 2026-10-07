#pragma once

#include <chrono>
#include <optional>
#include <string>
#include <vector>

#include "fmt/format.h"

#include "gtfsrt/gtfs-realtime.pb.h"

#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

#include "../transfer_rules_util.h"
#include "./util.h"

// GTFS-RT messages for the transfers.txt rule tests: trip updates of kDay,
// source 0.

namespace nigiri::test {

// The run of trip_id on kDay (source 0) and its trip; the run is invalid if it
// does not run.
inline std::pair<rt::run, trip_idx_t> resolve(timetable const& tt,
                                              rt_timetable const& rtt,
                                              std::string const& trip_id) {
  auto td = transit_realtime::TripDescriptor{};
  td.set_trip_id(trip_id);
  td.set_start_date("20190501");
  return rt::gtfsrt_resolve_run(kDay, tt, &rtt, source_idx_t{0}, td);
}

// Applies the updates; the lower bounds follow, as motis does after every
// update.
inline void update(timetable const& tt,
                   rt_timetable& rtt,
                   std::vector<trip_update> const& updates) {
  rt::gtfsrt_update_msg(
      tt, rtt, source_idx_t{0}, "",
      to_msg(updates, kDay + std::chrono::hours{8}, "20190501"));
  rtt.update_lbs(tt);
}

}  // namespace nigiri::test
