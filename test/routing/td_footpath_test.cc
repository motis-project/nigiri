#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"

#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/rt_timetable.h"
#include "../raptor_search.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace std::chrono_literals;
using nigiri::test::raptor_search;

namespace {

// Interchange at B
// A  --> B --> C
// A  --> D --> C
//
// A->B->C
// T1: A->B 10:00-11:00
// T2: B->C 11:30-12:00
// T3: B->C 12:00-12:30
//
// A->D->C
// T4: A->D 10:00-12:00
// T5: D->C 13:00-15:00
//
// Scenario 1:
// Everything works
// A@10:00 --T1--> 11:00 @ B @ 11:30 --T2--> 12:00 @ C
//
// Scenario 2:
// Elevator at B blocked completely, journey via D
// A@10:00 --T4--> 12:00 @ D @ 13:00 --T5--> 15:00 @ C
//
// Scenario 3:
// Elevator at B blocked until 11:25, 10min footpath = 11:35 arrival at B2
// A@10:00 --T1--> 11:00 @ B1 @ 11:00
// wait for evelator to work 11:00 - 11:25
// use elevator +10min       11:25 - 11:35
// B2 @ 12:00 --T3--> 12:30 @ C
mem_dir test_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B1,B1,,2.0,3.0,,
B2,B2,,2.0,3.0,,
C,C,,4.0,5.0,,
D,D,,6.0,7.0,,

# calendar_dates.txt
service_id,date,exception_type
S,20240619,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,RE 1,,,2
R2,DB,RE 2,,,2
R3,DB,RE 1,,,2
R4,DB,RE 2,,,2
R5,DB,RE 1,,,2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S,T1,RE 1,
R2,S,T2,RE 2,
R3,S,T3,RE 3,
R4,S,T4,RE 4,
R5,S,T5,RE 5,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,10:00:00,10:00:00,A,1,0,0
T1,11:00:00,11:00:00,B1,2,0,0
T2,11:30:00,11:30:00,B2,1,0,0
T2,12:00:00,12:00:00,C,2,0,0
T3,12:00:00,12:00:00,B2,1,0,0
T3,12:30:00,12:30:00,C,2,0,0
T4,10:00:00,10:00:00,A,1,0,0
T4,12:00:00,12:00:00,D,2,0,0
T5,13:00:00,13:00:00,D,1,0,0
T5,15:00:00,15:00:00,C,2,0,0
)");
}

}  // namespace

std::string to_string(timetable const& tt,
                      pareto_set<routing::journey> const& results) {
  std::stringstream ss;
  ss << "\n";
  for (auto const& x : results) {
    x.print(ss, tt);
    ss << "\n";
  }
  std::cout << ss.str() << "\n";
  return ss.str();
}

constexpr auto const kEverythingWorks = R"(
[2024-06-19 08:00, 2024-06-19 10:00]
TRANSFERS: 1
     FROM: (A, A) [2024-06-19 08:00]
       TO: (C, C) [2024-06-19 10:00]
leg 0: (A, A) [2024-06-19 08:00] -> (B1, B1) [2024-06-19 09:00]
   0: A       A...............................................                               d: 19.06 08:00 [19.06 10:00]  [{name=RE 1, day=2024-06-19, id=T1, src=0}]
   1: B1      B1.............................................. a: 19.06 09:00 [19.06 11:00]
leg 1: (B1, B1) [2024-06-19 09:00] -> (B2, B2) [2024-06-19 09:20]
  FOOTPATH (duration=20)
leg 2: (B2, B2) [2024-06-19 09:30] -> (C, C) [2024-06-19 10:00]
   0: B2      B2..............................................                               d: 19.06 09:30 [19.06 11:30]  [{name=RE 2, day=2024-06-19, id=T2, src=0}]
   1: C       C............................................... a: 19.06 10:00 [19.06 12:00]

)";

constexpr auto const kElevatorOutOfOrder = R"(
[2024-06-19 08:00, 2024-06-19 13:00]
TRANSFERS: 1
     FROM: (A, A) [2024-06-19 08:00]
       TO: (C, C) [2024-06-19 13:00]
leg 0: (A, A) [2024-06-19 08:00] -> (D, D) [2024-06-19 10:00]
   0: A       A...............................................                               d: 19.06 08:00 [19.06 10:00]  [{name=RE 2, day=2024-06-19, id=T4, src=0}]
   1: D       D............................................... a: 19.06 10:00 [19.06 12:00]
leg 1: (D, D) [2024-06-19 10:00] -> (D, D) [2024-06-19 10:02]
  FOOTPATH (duration=2)
leg 2: (D, D) [2024-06-19 11:00] -> (C, C) [2024-06-19 13:00]
   0: D       D...............................................                               d: 19.06 11:00 [19.06 13:00]  [{name=RE 1, day=2024-06-19, id=T5, src=0}]
   1: C       C............................................... a: 19.06 13:00 [19.06 15:00]

)";

constexpr auto const kElevatorStartsWorkingAt1125 = R"(
[2024-06-19 08:00, 2024-06-19 10:30]
TRANSFERS: 1
     FROM: (A, A) [2024-06-19 08:00]
       TO: (C, C) [2024-06-19 10:30]
leg 0: (A, A) [2024-06-19 08:00] -> (B1, B1) [2024-06-19 09:00]
   0: A       A...............................................                               d: 19.06 08:00 [19.06 10:00]  [{name=RE 1, day=2024-06-19, id=T1, src=0}]
   1: B1      B1.............................................. a: 19.06 09:00 [19.06 11:00]
leg 1: (B1, B1) [2024-06-19 09:25] -> (B2, B2) [2024-06-19 09:35]
  FOOTPATH (duration=10)
leg 2: (B2, B2) [2024-06-19 10:00] -> (C, C) [2024-06-19 10:30]
   0: B2      B2..............................................                               d: 19.06 10:00 [19.06 12:00]  [{name=RE 1, day=2024-06-19, id=T3, src=0}]
   1: C       C............................................... a: 19.06 10:30 [19.06 12:30]

)";

TEST(routing, td_footpath) {
  constexpr auto const kProfile = profile_idx_t{2U};

  timetable tt;
  tt.date_range_ = {date::sys_days{2024_y / June / 18},
                    date::sys_days{2024_y / June / 20}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  auto const B1 = tt.locations_.get({"B1", {}}).l_;
  auto const B2 = tt.locations_.get({"B2", {}}).l_;

  tt.locations_.footpaths_out_[kProfile].resize(tt.n_locations());
  tt.locations_.footpaths_in_[kProfile].resize(tt.n_locations());
  tt.locations_.footpaths_out_[kProfile][B1].push_back(footpath{B2, 20min});
  tt.locations_.footpaths_in_[kProfile][B2].push_back(footpath{B1, 20min});

  auto rtt = rt::create_rt_timetable(tt, sys_days{2024_y / June / 19});

  auto const run_search = [&]() {
    return raptor_search(tt, &rtt, "A", "C",
                         unixtime_t{sys_days{2024_y / June / 19}} + 8h,
                         nigiri::direction::kForward,
                         routing::all_clasz_allowed(), false, kProfile);
  };

  // Base: elevator available, no real-time information.
  EXPECT_EQ(kEverythingWorks, to_string(tt, run_search()));

  // Switch to real-time footpaths but don't add any footpaths.
  // Represents "elevator broken forever".
  rtt.has_td_footpaths_[kProfile].set(B1, true);
  rtt.has_td_footpaths_[kProfile].set(B2, true);
  rtt.td_footpaths_out_[kProfile].resize(tt.n_locations());
  rtt.td_footpaths_in_[kProfile].resize(tt.n_locations());

  EXPECT_EQ(kElevatorOutOfOrder, to_string(tt, run_search()));

  // Add elevator available beginning with 11:25 with 10min footpath length.
  rtt.td_footpaths_out_[kProfile][B1].push_back(td_footpath{
      B2, unixtime_t{sys_days{2024_y / June / 19} + 9h + 25min}, 10min});
  rtt.td_footpaths_in_[kProfile][B2].push_back(td_footpath{
      B1, unixtime_t{sys_days{2024_y / June / 19} + 9h + 25min}, 10min});

  EXPECT_EQ(kElevatorStartsWorkingAt1125, to_string(tt, run_search()));
}