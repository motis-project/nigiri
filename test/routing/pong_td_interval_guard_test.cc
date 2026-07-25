#include "gtest/gtest.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"

#include "nigiri/routing/raptor/pong.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace std::chrono_literals;

namespace {

// A --R1--> B --R2--> C, hourly, always 1 transfer at B.
mem_dir test_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Etc/UTC

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,
C,C,,4.0,5.0,,

# calendar_dates.txt
service_id,date,exception_type
S,20240619,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,RE 1,,,2
R2,DB,RE 2,,,2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S,T1_10,RE 1,
R1,S,T1_11,RE 1,
R1,S,T1_12,RE 1,
R2,S,T2_11,RE 2,
R2,S,T2_12,RE 2,
R2,S,T2_13,RE 2,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1_10,10:30:00,10:30:00,A,1,0,0
T1_10,11:00:00,11:00:00,B,2,0,0
T1_11,11:30:00,11:30:00,A,1,0,0
T1_11,12:00:00,12:00:00,B,2,0,0
T1_12,12:30:00,12:30:00,A,1,0,0
T1_12,13:00:00,13:00:00,B,2,0,0
T2_11,11:10:00,11:10:00,B,1,0,0
T2_11,11:25:00,11:25:00,C,2,0,0
T2_12,12:10:00,12:10:00,B,1,0,0
T2_12,12:25:00,12:25:00,C,2,0,0
T2_13,13:10:00,13:10:00,B,1,0,0
T2_13,13:25:00,13:25:00,C,2,0,0
)");
}

}  // namespace

// A td/flex access to B that is only valid BEFORE the search interval
// (window closes 05:00, interval starts 10:00). The forward ping can never
// use it. A backward pong, however, traverses it backwards from ~11:10 to
// the window and - without the time_at_dest guard on the td egress write -
// records journeys departing ~04:50, far before the interval. These
// phantoms are non-dominated (0 transfers vs 1 for the real connections),
// count towards 2 x min_connection_count, terminate the sweep early, and
// real later departures are never found.
TEST(routing, pong_td_egress_respects_worst_time_at_dest) {
  timetable tt;
  tt.date_range_ = {sys_days{2024_y / June / 18}, sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  auto const A = tt.find(location_id{"A", source_idx_t{0}}).value();
  auto const B = tt.find(location_id{"B", source_idx_t{0}}).value();
  auto const C = tt.find(location_id{"C", source_idx_t{0}}).value();

  auto const day = sys_days{2024_y / June / 19};
  auto q = routing::query{
      .start_time_ = interval<unixtime_t>{day + 10h, day + 11h},
      .start_match_mode_ = routing::location_match_mode::kIntermodal,
      .dest_match_mode_ = routing::location_match_mode::kIntermodal,
      .use_start_footpaths_ = false,
      .start_ = {{A, 5min, 0U}},
      .destination_ = {{C, 0min, 0U}},
      .td_start_ = {{B,
                     {{.valid_from_ = day + 0h,
                       .duration_ = 10min,
                       .transport_mode_id_ = 5},
                      {.valid_from_ = day + 5h,
                       .duration_ = footpath::kMaxDuration,
                       .transport_mode_id_ = 5}}}},
      .min_connection_count_ = 3U,
      .extend_interval_later_ = true};

  auto search_state = routing::search_state{};
  auto raptor_state = routing::raptor_state{};
  auto const result =
      routing::pong_search(tt, nullptr, search_state, raptor_state,
                           std::move(q), direction::kForward);

  auto deps = std::vector<unixtime_t>{};
  for (auto const& j : *result.journeys_) {
    // no journey may depart before the search interval
    EXPECT_GE(j.start_time_, day + 10h) << "phantom journey in results";
    deps.emplace_back(j.start_time_);
  }
  std::sort(begin(deps), end(deps));

  // 2 x min_connection_count = 6 -> the sweep must run until the third
  // real departure. With the unguarded td egress write, phantom journeys
  // (dep ~04:50, 0 transfers) fill the budget and the sweep stops early.
  auto const expected = std::vector<unixtime_t>{
      day + 10h + 25min, day + 11h + 25min, day + 12h + 25min};
  EXPECT_EQ(expected, deps)
      << "sweep terminated early - td phantom ate the connection budget?";
}
