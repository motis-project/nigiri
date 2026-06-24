#pragma once

#include <chrono>
#include <cstdint>
#include <vector>

#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/routing/limits.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri {
struct rt_timetable;
}

namespace nigiri::routing {

struct n_to_all_cell {
  location_idx_t owner_{location_idx_t::invalid()};
  duration_t travel_time_{std::numeric_limits<duration_t::rep>::max()};
  std::uint8_t transfers_{std::numeric_limits<std::uint8_t>::max()};
  bool reached_{false};
};

struct n_to_all_result {
  n_to_all_cell const& cell(location_idx_t const l) const {
    return cells_[to_idx(l)];
  }

  std::vector<n_to_all_cell> cells_;
  raptor_state state_;
};

n_to_all_result raptor_n_to_all(
    timetable const& tt,
    rt_timetable const* rtt,
    std::vector<location_idx_t> const& origins,
    unixtime_t start_time,
    std::uint8_t max_transfers = kMaxTransfers,
    duration_t max_travel_time = kMaxTravelTime);

}  // namespace nigiri::routing
