#pragma once

#include "nigiri/routing/journey.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::routing {

struct reach_info {
  bool valid() const noexcept {
    return start_end_ != location_idx_t::invalid();
  }

  double reach_{-1.0};
  routing::journey j_;
  location_idx_t start_end_{location_idx_t::invalid()};
  location_idx_t stop_in_route_{location_idx_t::invalid()};
};

vector_map<route_idx_t, reach_info> get_reach_values(
    timetable const& tt,
    std::vector<location_idx_t> const& source_locations,
    interval<unixtime_t>);

}  // namespace nigiri::routing