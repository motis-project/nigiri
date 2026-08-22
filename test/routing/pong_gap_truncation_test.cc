#include "gtest/gtest.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"

#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "../raptor_search.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace std::chrono_literals;

namespace {

// A -> B, three ways:
//   PH: A 09:59 -> B 11:00, direct (departs 1min BEFORE the interval)
//   E:  A 10:05 -> C 10:20, C 10:30 -> B 11:00 (1 transfer)
//   L:  A 15:00 -> B 15:30, direct (after a service gap)
//
// E must have a transfer: the phantom is only kept as a backward-pass
// pareto byproduct because it beats the matched journey on transfer count.
mem_dir test_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Etc/UTC

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
C,C,,1.0,2.0,,
B,B,,2.0,4.0,,

# calendar_dates.txt
service_id,date,exception_type
S,20240619,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
RPH,DB,PH,,,2
REA,DB,EA,,,2
REB,DB,EB,,,2
RL,DB,L,,,2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
RPH,S,PH,,
REA,S,EA,,
REB,S,EB,,
RL,S,L,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
PH,09:59:00,09:59:00,A,1,0,0
PH,11:00:00,11:00:00,B,2,0,0
EA,10:05:00,10:05:00,A,1,0,0
EA,10:20:00,10:20:00,C,2,0,0
EB,10:30:00,10:30:00,C,1,0,0
EB,11:00:00,11:00:00,B,2,0,0
L,15:00:00,15:00:00,A,1,0,0
L,15:30:00,15:30:00,B,2,0,0
)");
}

}  // namespace

// The backward validation runs bounded at the ping's start time. If that
// bound admits departures BEFORE the search interval (historically via the
// exclusive-bound "-1" compensation that the is_better_loose switch turned
// inclusive), the backward pass surfaces PH (dep 09:59, 0 transfers) as a
// pareto byproduct: it can never be produced by any forward ping (all pings
// start >= 10:00), stays unreconstructed, but counts towards
// 2 x min_connection_count. With min_connection_count = 2, the sweep then
// stops at cursor 10:06 (phantom + E validated = budget full) inside the
// 10:05 -> 15:00 service gap, and L - already found and reconstructed by
// the very first pong - is erased as not-validated: only 1 of 2 requested
// connections is delivered.
TEST(routing, pong_gap_truncation) {
  auto tt = timetable{};
  tt.date_range_ = {sys_days{2024_y / June / 18}, sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  auto const A = tt.find(location_id{"A", source_idx_t{0}}).value();
  auto const B = tt.find(location_id{"B", source_idx_t{0}}).value();

  auto const day = sys_days{2024_y / June / 19};
  auto q = routing::query{
      .start_time_ = interval<unixtime_t>{day + 10h, day + 10h + 3min},
      .start_match_mode_ = routing::location_match_mode::kEquivalent,
      .dest_match_mode_ = routing::location_match_mode::kEquivalent,
      .use_start_footpaths_ = false,
      .start_ = {{A, 0min, 0U}},
      .destination_ = {{B, 0min, 0U}},
      .min_connection_count_ = 2U,
      .extend_interval_later_ = true};

  auto const results =
      test::raptor_search(tt, nullptr, std::move(q), direction::kForward);

  auto deps = std::vector<unixtime_t>{};
  for (auto const& j : results) {
    // no journey may depart before the search interval
    EXPECT_GE(j.start_time_, day + 10h) << "phantom journey in results";
    deps.emplace_back(j.start_time_);
  }
  std::sort(begin(deps), end(deps));

  auto const expected = std::vector<unixtime_t>{day + 10h + 5min, day + 15h};
  EXPECT_EQ(expected, deps)
      << "sweep terminated inside the service gap - did an out-of-interval "
         "phantom eat the connection budget?";
}
