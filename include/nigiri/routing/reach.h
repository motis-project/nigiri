#pragma once

#include <mutex>

#include "nigiri/routing/journey.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::routing {

struct reach_info {
  reach_info();

  bool valid() const noexcept {
    return start_end_ != location_idx_t::invalid();
  }

  void update(double const new_reach,
              routing::journey const&,
              location_idx_t const start_end,
              location_idx_t const stop_in_route);

  std::mutex mutex_;
  double reach_{-1.0};
  routing::journey j_;
  location_idx_t start_end_{location_idx_t::invalid()};
  location_idx_t stop_in_route_{location_idx_t::invalid()};
};

std::vector<reach_info> get_reach_values(
    timetable const& tt,
    std::vector<location_idx_t> const& source_locations,
    interval<date::sys_days> const search_interval);

}  // namespace nigiri::routing