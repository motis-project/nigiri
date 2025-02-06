#pragma once

#include "nigiri/routing/query.h"
#include "nigiri/timetable.h"

namespace nigiri {
unixtime_t next_day(unixtime_t const time) {
  return floor<date::days>(time) + date::days{1};
};

hash_map<location_idx_t, std::vector<routing::td_offset>> create_td_offsets(
    timetable& tt,
    uint32_t location_idx,
    routing::start_time_t const& start_time,
    std::function<duration_t(geo::latlng, geo::latlng)> get_duration,
    uint8_t extra_days = 0,
    direction const search_dir = direction::kForward);
}  // namespace nigiri
