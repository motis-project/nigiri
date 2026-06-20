#pragma once

#include <chrono>
#include <set>
#include <vector>

#include "cista/reflection/comparable.h"

#include "nigiri/routing/query.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
struct rt_timetable;
}  // namespace nigiri

namespace nigiri::routing {

struct start {
  CISTA_FRIEND_COMPARABLE(start)
  unixtime_t time_at_start_;
  unixtime_t time_at_stop_;
  location_idx_t stop_;
};

void get_starts(
    direction,
    timetable const&,
    rt_timetable const*,
    start_time_t const& start_time,
    std::vector<offset> const& start_offsets,
    hash_map<location_idx_t, std::vector<td_offset>> const& start_td_offsets,
    std::vector<via_stop> const& via_stops,
    duration_t const max_start_offset,
    location_match_mode,
    bool use_start_footpaths,
    std::vector<start>&,
    bool add_ontrip,
    profile_idx_t,
    transfer_time_settings const&);

void collect_destinations(timetable const&,
                          std::vector<offset> const& destinations,
                          location_match_mode const,
                          bitvec& is_destination,
                          std::vector<std::uint16_t>& dist_to_dest);

void collect_via_destinations(timetable const&,
                              location_idx_t via,
                              bitvec& is_destination);

}  // namespace nigiri::routing
