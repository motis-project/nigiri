#pragma once

#include <vector>

#include "cista/containers/matrix.h"

#include "nigiri/routing/journey.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/routing_time.h"
#include "nigiri/routing/start_times.h"

namespace nigiri::routing {

struct search_state {
  void reset(timetable const& tt, routing_time init);

  std::vector<routing_time> best_;
  cista::raw::matrix<routing_time> round_times_;
  std::vector<bool> station_mark_;
  std::vector<bool> route_mark_;
  std::vector<start> starts_;
  pareto_set<journey> results_;
  interval<unixtime_t> search_interval_;
};

}  // namespace nigiri::routing
