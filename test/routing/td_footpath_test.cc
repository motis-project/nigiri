#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"

#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/rt_timetable.h"
#include "../raptor_search.h"
#include "results_to_string.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace std::chrono_literals;
using nigiri::test::raptor_search;

namespace {

// time-dependent footpath at start: works [9:30-9:45], duration=10min
// time-dependent foopath at end: does not work [14:30-15:30], duration=10min
//
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
// A@10:00 --T1--> 11:00 @ B @ 11:30 --T2--> 12:00 @ C --> 12:10
//
// Scenario 2:
// Elevator at B blocked completely, journey via D
// A@10:00 --T4--> 12:00 @ D @ 13:00 --T5--> 15:00 @ C --wait--> 15:30 --> 15:40
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

// clang-format-off
constexpr auto const kEverythingWorks = R"(
[2024-06-19 07:00, 2024-06-19 10:10]
TRANSFERS: 1
     FROM: (START, START) [2024-06-19 07:44]
       TO: (END, END) [2024-06-19 10:10]
leg 0: (START, START) [2024-06-19 07:44] -> (A, A) [2024-06-19 07:54]
  MUMO (id=0, duration=10)
leg 1: (A, A) [2024-06-19 08:00] -> (B1, B1) [2024-06-19 09:00]
   0: A       A...............................................                               d: 19.06 08:00 [19.06 10:00]  [{name=RE 1, day=2024-06-19, id=T1, src=0}]
   1: B1      B1.............................................. a: 19.06 09:00 [19.06 11:00]
leg 2: (B1, B1) [2024-06-19 09:00] -> (B2, B2) [2024-06-19 09:20]
  FOOTPATH (duration=20)
leg 3: (B2, B2) [2024-06-19 09:30] -> (C, C) [2024-06-19 10:00]
   0: B2      B2..............................................                               d: 19.06 09:30 [19.06 11:30]  [{name=RE 2, day=2024-06-19, id=T2, src=0}]
   1: C       C............................................... a: 19.06 10:00 [19.06 12:00]
leg 4: (C, C) [2024-06-19 10:00] -> (END, END) [2024-06-19 10:10]
  MUMO (id=0, duration=10)

)";

constexpr auto const kElevatorOutOfOrder = R"(
[2024-06-19 07:00, 2024-06-19 13:40]
TRANSFERS: 1
     FROM: (START, START) [2024-06-19 07:44]
       TO: (END, END) [2024-06-19 13:40]
leg 0: (START, START) [2024-06-19 07:44] -> (A, A) [2024-06-19 07:54]
  MUMO (id=0, duration=10)
leg 1: (A, A) [2024-06-19 08:00] -> (D, D) [2024-06-19 10:00]
   0: A       A...............................................                               d: 19.06 08:00 [19.06 10:00]  [{name=RE 2, day=2024-06-19, id=T4, src=0}]
   1: D       D............................................... a: 19.06 10:00 [19.06 12:00]
leg 2: (D, D) [2024-06-19 10:00] -> (D, D) [2024-06-19 10:02]
  FOOTPATH (duration=2)
leg 3: (D, D) [2024-06-19 11:00] -> (C, C) [2024-06-19 13:00]
   0: D       D...............................................                               d: 19.06 11:00 [19.06 13:00]  [{name=RE 1, day=2024-06-19, id=T5, src=0}]
   1: C       C............................................... a: 19.06 13:00 [19.06 15:00]
leg 4: (C, C) [2024-06-19 13:30] -> (END, END) [2024-06-19 13:40]
  MUMO (id=0, duration=10)

)";

constexpr auto const kElevatorStartsWorkingAt1125 = R"(
[2024-06-19 07:00, 2024-06-19 10:40]
TRANSFERS: 1
     FROM: (START, START) [2024-06-19 07:44]
       TO: (END, END) [2024-06-19 10:40]
leg 0: (START, START) [2024-06-19 07:44] -> (A, A) [2024-06-19 07:54]
  MUMO (id=0, duration=10)
leg 1: (A, A) [2024-06-19 08:00] -> (B1, B1) [2024-06-19 09:00]
   0: A       A...............................................                               d: 19.06 08:00 [19.06 10:00]  [{name=RE 1, day=2024-06-19, id=T1, src=0}]
   1: B1      B1.............................................. a: 19.06 09:00 [19.06 11:00]
leg 2: (B1, B1) [2024-06-19 09:25] -> (B2, B2) [2024-06-19 09:35]
  FOOTPATH (duration=10)
leg 3: (B2, B2) [2024-06-19 10:00] -> (C, C) [2024-06-19 10:30]
   0: B2      B2..............................................                               d: 19.06 10:00 [19.06 12:00]  [{name=RE 1, day=2024-06-19, id=T3, src=0}]
   1: C       C............................................... a: 19.06 10:30 [19.06 12:30]
leg 4: (C, C) [2024-06-19 10:30] -> (END, END) [2024-06-19 10:40]
  MUMO (id=0, duration=10)

)";

TEST(routing, td_footpath) {
  constexpr auto const kProfile = profile_idx_t{2U};

  timetable tt;
  tt.date_range_ = {date::sys_days{2024_y / June / 18},
                    date::sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  tt.fwd_search_lb_graph_[kWheelchairProfile] =
      tt.fwd_search_lb_graph_[kDefaultProfile];
  tt.bwd_search_lb_graph_[kWheelchairProfile] =
      tt.bwd_search_lb_graph_[kDefaultProfile];

  auto const find_loc = [&](std::string_view id) {
    auto const idx = tt.find(location_id{id, source_idx_t{0U}});
    EXPECT_TRUE(idx.has_value()) << id;
    return idx.value_or(location_idx_t::invalid());
  };
  auto const A = find_loc("A");
  auto const C = find_loc("C");
  auto const B1 = find_loc("B1");
  auto const B2 = find_loc("B2");

  tt.locations_.footpaths_out_[kProfile].resize(tt.n_locations());
  tt.locations_.footpaths_in_[kProfile].resize(tt.n_locations());
  tt.locations_.footpaths_out_[kProfile][B1].push_back(footpath{B2, 20min});
  tt.locations_.footpaths_in_[kProfile][B2].push_back(footpath{B1, 20min});

  auto rtt = rt::create_rt_timetable(tt, sys_days{2024_y / June / 19});

  auto const run_search = [&]() {
    return raptor_search(
        tt, &rtt,
        routing::query{
            .start_time_ = unixtime_t{sys_days{2024_y / June / 19}} + 7h,
            .start_match_mode_ = routing::location_match_mode::kIntermodal,
            .dest_match_mode_ = routing::location_match_mode::kIntermodal,
            .use_start_footpaths_ = false,
            .td_start_ =
                {{{A,
                   {{.valid_from_ = sys_days{1970_y / January / 1},
                     .duration_ = footpath::kMaxDuration,
                     .transport_mode_id_ = 0},
                    {.valid_from_ = sys_days{2024_y / June / 19} + 7h + 30min,
                     .duration_ = 10min,
                     .transport_mode_id_ = 0},
                    {.valid_from_ = sys_days{2024_y / June / 19} + 7h + 45min,
                     .duration_ = footpath::kMaxDuration,
                     .transport_mode_id_ = 0}}}}},
            .td_dest_ =
                {{{C,
                   {{.valid_from_ = sys_days{1970_y / January / 1},
                     .duration_ = 10min,
                     .transport_mode_id_ = 0},
                    {.valid_from_ = sys_days{2024_y / June / 19} + 12h + 30min,
                     .duration_ = footpath::kMaxDuration,
                     .transport_mode_id_ = 0},
                    {.valid_from_ = sys_days{2024_y / June / 19} + 13h + 30min,
                     .duration_ = 10min,
                     .transport_mode_id_ = 0}}}}},
            .prf_idx_ = 2U},
        direction::kForward);
  };

  // Base: elevator available, no real-time information.
  EXPECT_EQ(kEverythingWorks, to_string(tt, run_search()));

  // Switch to real-time footpaths but don't add any footpaths.
  // Represents "elevator broken forever".
  rtt.has_td_footpaths_in_[kProfile].set(B1, true);
  rtt.has_td_footpaths_in_[kProfile].set(B2, true);
  rtt.has_td_footpaths_out_[kProfile].set(B1, true);
  rtt.has_td_footpaths_out_[kProfile].set(B2, true);
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
