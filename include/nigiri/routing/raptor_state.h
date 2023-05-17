#pragma once

#include <set>
#include <vector>

#include "cista/containers/flat_matrix.h"

#include "nigiri/routing/routing_time.h"
#include "nigiri/routing/start_times.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::routing {

struct raptor_state {
  void reset(timetable const& tt, routing_time init);

  std::vector<routing_time> best_;
  cista::raw::flat_matrix<routing_time> round_times_;
  std::vector<bool> station_mark_;
  std::vector<bool> transport_station_mark_;
  std::vector<bool> prev_station_mark_;
  std::vector<bool> route_mark_;
};

}  // namespace nigiri::routing
