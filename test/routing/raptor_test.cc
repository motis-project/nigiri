#include "gtest/gtest.h"

#include "nigiri/loader/dir.h"
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

// Local times (Europe/Berlin, UTC+2). MO runs Mon 2024-06-17, WE Wed 06-19.
// A -> B Mon 22:00-23:00, B -> C only Wed 07:00-08:00: the earliest trip at B
// is two days later, past a day without the route.
// E -> F Mon 23:00-02:10, then the night trip H 23:30 -> F 02:40 -> G 03:00:
// Monday's service, boarded at F on Tuesday also in UTC (00:40), the
// timetable's day.
loader::mem_dir multi_day_files() {
  return loader::mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
MTA,MOTIS Transit Authority,https://motis-project.de/,Europe/Berlin

# calendar_dates.txt
service_id,date,exception_type
MO,20240617,1
WE,20240619,1

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,,,,,
B,B,,,,,,
C,C,,,,,,
E,E,,,,,,
F,F,,,,,,
G,G,,,,,,
H,H,,,,,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R,MTA,R,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R,MO,AB,,
R,WE,BC,,
R,MO,EF,,
R,MO,NIGHT,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
AB,22:00:00,22:00:00,A,0,0,0
AB,23:00:00,23:00:00,B,1,0,0
BC,07:00:00,07:00:00,B,0,0,0
BC,08:00:00,08:00:00,C,1,0,0
EF,23:00:00,23:00:00,E,0,0,0
EF,26:10:00,26:10:00,F,1,0,0
NIGHT,23:30:00,23:30:00,H,0,0,0
NIGHT,26:40:00,26:40:00,F,1,0,0
NIGHT,27:00:00,27:00:00,G,2,0,0
)");
}

}  // namespace

// The earliest trip lookup across days, on every engine (test::raptor_search
// compares pong, McRAPTOR and the GPU against plain RAPTOR).
TEST(routing, earliest_transport_across_days) {
  using namespace std::chrono_literals;
  auto const tt =
      loader::load(multi_day_files(), {2024_y / June / 16, 2024_y / June / 24});
  auto const utc = [](date::sys_days const d, std::chrono::minutes const m) {
    return unixtime_t{d} + m;
  };
  auto const mon = sys_days{2024_y / June / 17};
  auto const wed = sys_days{2024_y / June / 19};
  auto const check = [&](std::string_view const from, std::string_view const to,
                         interval<unixtime_t> const window, direction const dir,
                         unixtime_t const dep, unixtime_t const arr) {
    auto const results = raptor_search(tt, nullptr, from, to, window, dir);
    ASSERT_EQ(1U, results.size()) << from << " -> " << to;
    EXPECT_EQ(dep, begin(results)->departure_time()) << from << " -> " << to;
    EXPECT_EQ(arr, begin(results)->arrival_time()) << from << " -> " << to;
  };

  // A 20:00 UTC Mon -> C 06:00 UTC Wed
  check("A", "C", {utc(mon, 19h + 30min), utc(mon, 20h + 30min)},
        direction::kForward, utc(mon, 20h), utc(wed, 6h));
  check("C", "A", {utc(wed, 5h + 30min), utc(wed, 6h + 30min)},
        direction::kBackward, utc(mon, 20h), utc(wed, 6h));

  // E 21:00 UTC Mon -> G 01:00 UTC Tue
  auto const tue = sys_days{2024_y / June / 18};
  check("E", "G", {utc(mon, 20h + 30min), utc(mon, 21h + 30min)},
        direction::kForward, utc(mon, 21h), utc(tue, 1h));
  check("G", "E", {utc(tue, 0h + 30min), utc(tue, 1h + 30min)},
        direction::kBackward, utc(mon, 21h), utc(tue, 1h));
}
