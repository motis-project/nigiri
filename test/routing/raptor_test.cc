#include "gtest/gtest.h"

#include "nigiri/loader/load.h"
#include "nigiri/special_stations.h"

#include "../loader/hrd/hrd_timetable.h"

#include "../raptor_search.h"
#include "results_to_string.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::test_data::hrd_timetable;
using nigiri::test::raptor_search;

constexpr auto const fwd_journeys = R"(
[2020-03-30 05:00, 2020-03-30 07:15]
TRANSFERS: 1
     FROM: (A, 0000001) [2020-03-30 05:00]
       TO: (C, 0000003) [2020-03-30 07:15]
leg 0: (A, 0000001) [2020-03-30 05:00] -> (B, 0000002) [2020-03-30 06:00]
   0: 0000001 A...............................................                               d: 30.03 05:00 [30.03 07:00]  [{name=RE 1337, day=2020-03-30, id=1337/0000001/300/0000002/360/, src=0}]
   1: 0000002 B............................................... a: 30.03 06:00 [30.03 08:00]
leg 1: (B, 0000002) [2020-03-30 06:00] -> (B, 0000002) [2020-03-30 06:02]
  FOOTPATH (duration=2)
leg 2: (B, 0000002) [2020-03-30 06:15] -> (C, 0000003) [2020-03-30 07:15]
   0: 0000002 B...............................................                               d: 30.03 06:15 [30.03 08:15]  [{name=RE 7331, day=2020-03-30, id=7331/0000002/375/0000003/435/, src=0}]
   1: 0000003 C............................................... a: 30.03 07:15 [30.03 09:15]

[2020-03-30 05:30, 2020-03-30 07:45]
TRANSFERS: 1
     FROM: (A, 0000001) [2020-03-30 05:30]
       TO: (C, 0000003) [2020-03-30 07:45]
leg 0: (A, 0000001) [2020-03-30 05:30] -> (B, 0000002) [2020-03-30 06:30]
   0: 0000001 A...............................................                               d: 30.03 05:30 [30.03 07:30]  [{name=RE 1337, day=2020-03-30, id=1337/0000001/330/0000002/390/, src=0}]
   1: 0000002 B............................................... a: 30.03 06:30 [30.03 08:30]
leg 1: (B, 0000002) [2020-03-30 06:30] -> (B, 0000002) [2020-03-30 06:32]
  FOOTPATH (duration=2)
leg 2: (B, 0000002) [2020-03-30 06:45] -> (C, 0000003) [2020-03-30 07:45]
   0: 0000002 B...............................................                               d: 30.03 06:45 [30.03 08:45]  [{name=RE 7331, day=2020-03-30, id=7331/0000002/405/0000003/465/, src=0}]
   1: 0000003 C............................................... a: 30.03 07:45 [30.03 09:45]

)";

TEST(routing, raptor_forward) {

  auto const tt = loader::load(files_abc(), full_period());

  auto const results = raptor_search(
      tt, nullptr, "0000001", "0000003",
      interval{unixtime_t{sys_days{2020_y / March / 30}} + 5_hours,
               unixtime_t{sys_days{2020_y / March / 30}} + 6_hours});

  EXPECT_EQ(std::string_view{fwd_journeys}, to_string(tt, results));
}

constexpr auto const bwd_journeys = R"(
[2020-03-30 03:00, 2020-03-30 05:15]
TRANSFERS: 1
     FROM: (A, 0000001) [2020-03-30 03:00]
       TO: (C, 0000003) [2020-03-30 05:15]
leg 0: (A, 0000001) [2020-03-30 03:00] -> (B, 0000002) [2020-03-30 04:00]
   0: 0000001 A...............................................                               d: 30.03 03:00 [30.03 05:00]  [{name=RE 1337, day=2020-03-30, id=1337/0000001/180/0000002/240/, src=0}]
   1: 0000002 B............................................... a: 30.03 04:00 [30.03 06:00]
leg 1: (B, 0000002) [2020-03-30 04:00] -> (B, 0000002) [2020-03-30 04:02]
  FOOTPATH (duration=2)
leg 2: (B, 0000002) [2020-03-30 04:15] -> (C, 0000003) [2020-03-30 05:15]
   0: 0000002 B...............................................                               d: 30.03 04:15 [30.03 06:15]  [{name=RE 7331, day=2020-03-30, id=7331/0000002/255/0000003/315/, src=0}]
   1: 0000003 C............................................... a: 30.03 05:15 [30.03 07:15]

[2020-03-30 03:30, 2020-03-30 05:45]
TRANSFERS: 1
     FROM: (A, 0000001) [2020-03-30 03:30]
       TO: (C, 0000003) [2020-03-30 05:45]
leg 0: (A, 0000001) [2020-03-30 03:30] -> (B, 0000002) [2020-03-30 04:30]
   0: 0000001 A...............................................                               d: 30.03 03:30 [30.03 05:30]  [{name=RE 1337, day=2020-03-30, id=1337/0000001/210/0000002/270/, src=0}]
   1: 0000002 B............................................... a: 30.03 04:30 [30.03 06:30]
leg 1: (B, 0000002) [2020-03-30 04:30] -> (B, 0000002) [2020-03-30 04:32]
  FOOTPATH (duration=2)
leg 2: (B, 0000002) [2020-03-30 04:45] -> (C, 0000003) [2020-03-30 05:45]
   0: 0000002 B...............................................                               d: 30.03 04:45 [30.03 06:45]  [{name=RE 7331, day=2020-03-30, id=7331/0000002/285/0000003/345/, src=0}]
   1: 0000003 C............................................... a: 30.03 05:45 [30.03 07:45]

)";

TEST(routing, raptor_backward) {
  using namespace date;

  auto const tt = loader::load(files_abc(), full_period());

  auto const results = raptor_search(
      tt, nullptr, "0000003", "0000001",
      interval{unixtime_t{sys_days{2020_y / March / 30}} + 5_hours,
               unixtime_t{sys_days{2020_y / March / 30}} + 6_hours},
      direction::kBackward);

  EXPECT_EQ(2U, results.size());

  EXPECT_EQ(std::string_view{bwd_journeys}, to_string(tt, results));
}

namespace {

// Range query A -> D, start times 10:30 and 10:00 (processed in this order).
// At 10:30, the loop trip T1 (A -> B -> A) leaves a round-1 label at A
// (14:00 + transfer time) before R2 sets the arrival at D to 11:00.
// round_times_ survive into the 10:00 iteration while best_ is cleared, and
// the start label lives in round_times_[0] only, so best_[A] at k = 1 is the
// stale 14:02. A route-level lower bound check that takes the boarding time
// from best_ (14:02 + lb(D) = 0 is not better than 11:00) skips R2 and loses
// the 10:00 -> 10:30 journey. Boarding uses round_times_[k - 1] (10:00).
mem_dir range_query_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,
D,D,,6.0,7.0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,1,,,3
R2,DB,2,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S1,T1,,
R2,S1,T2,,
R2,S1,T3,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T1,10:30:00,10:30:00,A,0
T1,10:45:00,10:45:00,B,1
T1,14:00:00,14:00:00,A,2
T2,10:00:00,10:00:00,A,0
T2,10:30:00,10:30:00,D,1
T3,10:30:00,10:30:00,A,0
T3,11:00:00,11:00:00,D,1

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)");
}

constexpr auto const expected_range_query = R"(
[2019-05-01 08:00, 2019-05-01 08:30]
TRANSFERS: 0
     FROM: (A, A) [2019-05-01 08:00]
       TO: (D, D) [2019-05-01 08:30]
leg 0: (A, A) [2019-05-01 08:00] -> (D, D) [2019-05-01 08:30]
   0: A       A...............................................                               d: 01.05 08:00 [01.05 10:00]  [{name=2, day=2019-05-01, id=T2, src=0}]
   1: D       D............................................... a: 01.05 08:30 [01.05 10:30]

[2019-05-01 08:30, 2019-05-01 09:00]
TRANSFERS: 0
     FROM: (A, A) [2019-05-01 08:30]
       TO: (D, D) [2019-05-01 09:00]
leg 0: (A, A) [2019-05-01 08:30] -> (D, D) [2019-05-01 09:00]
   0: A       A...............................................                               d: 01.05 08:30 [01.05 10:30]  [{name=2, day=2019-05-01, id=T3, src=0}]
   1: D       D............................................... a: 01.05 09:00 [01.05 11:00]

)";

}  // namespace

TEST(routing, route_lb_range_query_uses_round_times) {
  auto const tt =
      loader::load(range_query_files(), {2019_y / May / 1, 2019_y / May / 2});
  // 10:00 - 10:31 Europe/Berlin (CEST)
  auto const results = raptor_search(
      tt, nullptr, "A", "D",
      interval{unixtime_t{sys_days{2019_y / May / 1}} + 8_hours,
               unixtime_t{sys_days{2019_y / May / 1}} + 8_hours + 31_minutes});
  EXPECT_EQ(expected_range_query, to_string(tt, results));
}
