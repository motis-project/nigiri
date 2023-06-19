#pragma once

#include <chrono>
#include <set>
#include <vector>

#include "cista/reflection/comparable.h"

#include "nigiri/routing/query.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing {

struct start {
  CISTA_FRIEND_COMPARABLE(start)
  unixtime_t time_at_start_;
  unixtime_t time_at_stop_;
  location_idx_t stop_;
};

void get_starts(direction,
                timetable const&,
                start_time_t const& start_time,
                std::vector<offset> const& station_offsets,
                location_match_mode,
                bool use_start_footpaths,
                std::vector<start>&,
                bool add_ontrip,
                int const& profile);

void collect_destinations(timetable const&,
                          std::vector<offset> const& destinations,
                          location_match_mode const,
                          std::vector<bool>& is_destination,
                          std::vector<std::uint16_t>& dist_to_dest);

}  // namespace nigiri::routing
