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

// Two ways from a coordinate that reaches both A and A2 in 5 min to C:
//   A2 -> C direct        (0 transfers, leaves the coordinate at 09:45)
//   A -> B -> C           (1 transfer,  leaves the coordinate at 09:55)
// Both arrive 11:00, so neither dominates the other: one departs later, the
// other changes less. Verifying the 1-transfer journey is what needs a loose
// comparison - see the test below.
mem_dir test_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Etc/UTC

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
A2,A2,,0.1,1.1,,
B,B,,2.0,3.0,,
C,C,,4.0,5.0,,

# calendar_dates.txt
service_id,date,exception_type
MON,20240617,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,RE 1,,,2
R2,DB,RE 2,,,2
R3,DB,RE 3,,,2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,MON,T1,RE 1,
R2,MON,T2,RE 2,
R3,MON,T4,RE 3,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,10:00:00,10:00:00,A,1,0,0
T1,10:30:00,10:30:00,B,2,0,0
T2,10:40:00,10:40:00,B,1,0,0
T2,11:00:00,11:00:00,C,2,0,0
T4,09:50:00,09:50:00,A2,1,0,0
T4,11:00:00,11:00:00,C,2,0,0
)");
}

}  // namespace

// An intermodal start makes the destination of pong's backward run the
// intermodal one, and its bound the departure of the journey it has to
// confirm. The search interval starts at that departure, so the forward search
// reports exactly the time the backward run then computes: the egress lands on
// the bound rather than beating it. Comparing that strictly drops the only
// label there is, the verification finds nothing, and the query is answered
// with nothing at all - "no pong for transfers=0 ... journeys=[]", the same
// way real queries failed.
//
// test::raptor_search runs both engines and compares them whenever the build
// has CUDA, so this test is what tells them apart: it fails on a build that
// compares strictly and passes on one that compares loosely, like the CPU.
// Without CUDA it still exercises the CPU path, which has always been loose.
TEST(routing, gpu_intermodal_egress_at_bound) {
  timetable tt;
  tt.date_range_ = {sys_days{2024_y / June / 16}, sys_days{2024_y / June / 19}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  auto const A = tt.find(location_id{"A", source_idx_t{0}}).value();
  auto const A2 = tt.find(location_id{"A2", source_idx_t{0}}).value();
  auto const C = tt.find(location_id{"C", source_idx_t{0}}).value();

  auto const monday = sys_days{2024_y / June / 17};
  auto q = routing::query{
      // anchored at the departure of the direct journey, so the forward
      // search reports exactly the time the backward run computes - that
      // equality is what the comparison has to accept
      .start_time_ =
          interval<unixtime_t>{monday + 9h + 45min, monday + 10h + 45min},
      .start_match_mode_ = routing::location_match_mode::kIntermodal,
      .dest_match_mode_ = routing::location_match_mode::kIntermodal,
      .use_start_footpaths_ = false,
      .start_ = {{A, 5min, 0U}, {A2, 5min, 0U}},
      .destination_ = {{C, 0min, 0U}},
      .min_connection_count_ = 1U};

  auto const results =
      test::raptor_search(tt, nullptr, std::move(q), direction::kForward);

  auto found = false;
  auto got = std::string{};
  for (auto const& j : results) {
    found |= j.start_time_ == monday + 9h + 55min &&
             j.dest_time_ == monday + 11h && j.transfers_ == 1U;
    got += fmt::format("({}, {}, {})\n", j.start_time_, j.dest_time_,
                       j.transfers_);
  }
  EXPECT_TRUE(found) << "intermodal journey lost, got:\n" << got;
}
