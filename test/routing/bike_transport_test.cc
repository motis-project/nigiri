
#include "gtest/gtest.h"

#include "utl/erase_if.h"

#include "nigiri/loader/gtfs/agency.h"
#include "nigiri/loader/gtfs/calendar.h"
#include "nigiri/loader/gtfs/calendar_date.h"
#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/gtfs/local_to_utc.h"
#include "nigiri/loader/gtfs/noon_offsets.h"
#include "nigiri/loader/init_finish.h"

#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

#include "../raptor_search.h"
#include "../rt/util.h"

using namespace nigiri;
using namespace date;
using namespace std::chrono_literals;
using namespace std::string_view_literals;
using nigiri::test::raptor_search;

namespace {

// Timetable 1:
// Bikes allowed in T1, T3, T4, T6, not allowed in T0, T2, T5
// Connections with bikes allowed are slower
// A -> C: 10:00 - 10:20 | T0 (no bikes)
//         10:00 - 10:30 | T1 (bikes allowed)
//         11:00 - 11:30 | T2 (no bikes)
// C -> D: 10:35 - 10:40 | T3 (bikes allowed)
//         10:45 - 10:50 | T4 (bikes allowed)
// A -> D: 10:00 - 10:40 | T0, T3 (no bikes in T0)
//         10:00 - 10:40 | T1, T3 (bikes allowed)
// A -> F: 10:00 - 10:55 | T0, T3, T5 (no bikes in T0, T5)
//         10:00 - 11:05 | T1, T4, T6 (bikes allowed)
constexpr auto const test_files_1 = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,
C,C,,4.0,5.0,,
D,D,,6.0,7.0,,
E,E,,8.0,9.0,,
F,F,,10.0,11.0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DB,0,,,3
R1,DB,1,,,3
R2,DB,2,,,3
R3,DB,3,,,3
R4,DB,4,,,3
R5,DB,5,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id,bikes_allowed
R0,S1,T0,,,2
R1,S1,T1,,,1
R1,S1,T2,,,2
R2,S1,T3,,1,1
R2,S1,T4,,1,1
R3,S1,T5,,1,2
R3,S1,T6,,1,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T0,10:00:00,10:00:00,A,0
T0,10:10:00,10:10:00,B,1
T0,10:20:00,10:20:00,C,1
T1,10:00:00,10:00:00,A,0
T1,10:15:00,10:15:00,B,1
T1,10:30:00,10:30:00,C,1
T2,11:00:00,11:00:00,A,0
T2,11:15:00,11:15:00,B,1
T2,11:30:00,11:30:00,C,1
T3,10:35:00,10:35:00,C,0
T3,10:40:00,10:40:00,D,1
T4,10:45:00,10:45:00,C,0
T4,10:50:00,10:50:00,D,1
T5,10:42:00,10:42:00,D,0
T5,10:50:00,10:50:00,E,0
T5,10:55:00,10:55:00,F,0
T6,10:52:00,10:52:00,D,0
T6,11:00:00,11:00:00,E,0
T6,11:05:00,11:05:00,F,0

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)"sv;

constexpr auto const expected_A_C_no_bike =
    R"(
[2019-05-01 08:00, 2019-05-01 08:20]
TRANSFERS: 0
     FROM: (A, A) [2019-05-01 08:00]
       TO: (C, C) [2019-05-01 08:20]
leg 0: (A, A) [2019-05-01 08:00] -> (C, C) [2019-05-01 08:20]
   0: A       A...............................................                               d: 01.05 08:00 [01.05 10:00]  [{name=Bus 0, day=2019-05-01, id=T0, src=0}]
   1: B       B............................................... a: 01.05 08:10 [01.05 10:10]  d: 01.05 08:10 [01.05 10:10]  [{name=Bus 0, day=2019-05-01, id=T0, src=0}]
   2: C       C............................................... a: 01.05 08:20 [01.05 10:20]


[2019-05-01 09:00, 2019-05-01 09:30]
TRANSFERS: 0
     FROM: (A, A) [2019-05-01 09:00]
       TO: (C, C) [2019-05-01 09:30]
leg 0: (A, A) [2019-05-01 09:00] -> (C, C) [2019-05-01 09:30]
   0: A       A...............................................                               d: 01.05 09:00 [01.05 11:00]  [{name=Bus 1, day=2019-05-01, id=T2, src=0}]
   1: B       B............................................... a: 01.05 09:15 [01.05 11:15]  d: 01.05 09:15 [01.05 11:15]  [{name=Bus 1, day=2019-05-01, id=T2, src=0}]
   2: C       C............................................... a: 01.05 09:30 [01.05 11:30]


)"sv;

constexpr auto const expected_A_C_bike =
    R"(
[2019-05-01 08:00, 2019-05-01 08:30]
TRANSFERS: 0
     FROM: (A, A) [2019-05-01 08:00]
       TO: (C, C) [2019-05-01 08:30]
leg 0: (A, A) [2019-05-01 08:00] -> (C, C) [2019-05-01 08:30]
   0: A       A...............................................                               d: 01.05 08:00 [01.05 10:00]  [{name=Bus 1, day=2019-05-01, id=T1, src=0}]
   1: B       B............................................... a: 01.05 08:15 [01.05 10:15]  d: 01.05 08:15 [01.05 10:15]  [{name=Bus 1, day=2019-05-01, id=T1, src=0}]
   2: C       C............................................... a: 01.05 08:30 [01.05 10:30]


)"sv;

constexpr auto const expected_A_D_no_bike =
    R"(
[2019-05-01 08:00, 2019-05-01 08:40]
TRANSFERS: 1
     FROM: (A, A) [2019-05-01 08:00]
       TO: (D, D) [2019-05-01 08:40]
leg 0: (A, A) [2019-05-01 08:00] -> (C, C) [2019-05-01 08:20]
   0: A       A...............................................                               d: 01.05 08:00 [01.05 10:00]  [{name=Bus 0, day=2019-05-01, id=T0, src=0}]
   1: B       B............................................... a: 01.05 08:10 [01.05 10:10]  d: 01.05 08:10 [01.05 10:10]  [{name=Bus 0, day=2019-05-01, id=T0, src=0}]
   2: C       C............................................... a: 01.05 08:20 [01.05 10:20]
leg 1: (C, C) [2019-05-01 08:35] -> (D, D) [2019-05-01 08:40]
   0: C       C...............................................                               d: 01.05 08:35 [01.05 10:35]  [{name=Bus 2, day=2019-05-01, id=T3, src=0}]
   1: D       D............................................... a: 01.05 08:40 [01.05 10:40]


)"sv;

constexpr auto const expected_A_D_bike =
    R"(
[2019-05-01 08:00, 2019-05-01 08:40]
TRANSFERS: 1
     FROM: (A, A) [2019-05-01 08:00]
       TO: (D, D) [2019-05-01 08:40]
leg 0: (A, A) [2019-05-01 08:00] -> (C, C) [2019-05-01 08:30]
   0: A       A...............................................                               d: 01.05 08:00 [01.05 10:00]  [{name=Bus 1, day=2019-05-01, id=T1, src=0}]
   1: B       B............................................... a: 01.05 08:15 [01.05 10:15]  d: 01.05 08:15 [01.05 10:15]  [{name=Bus 1, day=2019-05-01, id=T1, src=0}]
   2: C       C............................................... a: 01.05 08:30 [01.05 10:30]
leg 1: (C, C) [2019-05-01 08:35] -> (D, D) [2019-05-01 08:40]
   0: C       C...............................................                               d: 01.05 08:35 [01.05 10:35]  [{name=Bus 2, day=2019-05-01, id=T3, src=0}]
   1: D       D............................................... a: 01.05 08:40 [01.05 10:40]


)"sv;

constexpr auto const expected_C_D =
    R"(
[2019-05-01 08:35, 2019-05-01 08:40]
TRANSFERS: 0
     FROM: (C, C) [2019-05-01 08:35]
       TO: (D, D) [2019-05-01 08:40]
leg 0: (C, C) [2019-05-01 08:35] -> (D, D) [2019-05-01 08:40]
   0: C       C...............................................                               d: 01.05 08:35 [01.05 10:35]  [{name=Bus 2, day=2019-05-01, id=T3, src=0}]
   1: D       D............................................... a: 01.05 08:40 [01.05 10:40]


[2019-05-01 08:45, 2019-05-01 08:50]
TRANSFERS: 0
     FROM: (C, C) [2019-05-01 08:45]
       TO: (D, D) [2019-05-01 08:50]
leg 0: (C, C) [2019-05-01 08:45] -> (D, D) [2019-05-01 08:50]
   0: C       C...............................................                               d: 01.05 08:45 [01.05 10:45]  [{name=Bus 2, day=2019-05-01, id=T4, src=0}]
   1: D       D............................................... a: 01.05 08:50 [01.05 10:50]


)"sv;

constexpr auto const expected_A_F_no_bike =
    R"(
[2019-05-01 08:00, 2019-05-01 08:55]
TRANSFERS: 1
     FROM: (A, A) [2019-05-01 08:00]
       TO: (F, F) [2019-05-01 08:55]
leg 0: (A, A) [2019-05-01 08:00] -> (C, C) [2019-05-01 08:20]
   0: A       A...............................................                               d: 01.05 08:00 [01.05 10:00]  [{name=Bus 0, day=2019-05-01, id=T0, src=0}]
   1: B       B............................................... a: 01.05 08:10 [01.05 10:10]  d: 01.05 08:10 [01.05 10:10]  [{name=Bus 0, day=2019-05-01, id=T0, src=0}]
   2: C       C............................................... a: 01.05 08:20 [01.05 10:20]
leg 1: (C, C) [2019-05-01 08:35] -> (F, F) [2019-05-01 08:55]
   0: C       C...............................................                               d: 01.05 08:35 [01.05 10:35]  [{name=Bus 2, day=2019-05-01, id=T3, src=0}]
   1: D       D............................................... a: 01.05 08:40 [01.05 10:40]  d: 01.05 08:42 [01.05 10:42]  [{name=Bus 3, day=2019-05-01, id=T5, src=0}]
   2: E       E............................................... a: 01.05 08:50 [01.05 10:50]  d: 01.05 08:50 [01.05 10:50]  [{name=Bus 3, day=2019-05-01, id=T5, src=0}]
   3: F       F............................................... a: 01.05 08:55 [01.05 10:55]


)"sv;

constexpr auto const expected_A_F_bike =
    R"(
[2019-05-01 08:00, 2019-05-01 09:05]
TRANSFERS: 1
     FROM: (A, A) [2019-05-01 08:00]
       TO: (F, F) [2019-05-01 09:05]
leg 0: (A, A) [2019-05-01 08:00] -> (C, C) [2019-05-01 08:30]
   0: A       A...............................................                               d: 01.05 08:00 [01.05 10:00]  [{name=Bus 1, day=2019-05-01, id=T1, src=0}]
   1: B       B............................................... a: 01.05 08:15 [01.05 10:15]  d: 01.05 08:15 [01.05 10:15]  [{name=Bus 1, day=2019-05-01, id=T1, src=0}]
   2: C       C............................................... a: 01.05 08:30 [01.05 10:30]
leg 1: (C, C) [2019-05-01 08:45] -> (F, F) [2019-05-01 09:05]
   0: C       C...............................................                               d: 01.05 08:45 [01.05 10:45]  [{name=Bus 2, day=2019-05-01, id=T4, src=0}]
   1: D       D............................................... a: 01.05 08:50 [01.05 10:50]  d: 01.05 08:52 [01.05 10:52]  [{name=Bus 3, day=2019-05-01, id=T6, src=0}]
   2: E       E............................................... a: 01.05 09:00 [01.05 11:00]  d: 01.05 09:00 [01.05 11:00]  [{name=Bus 3, day=2019-05-01, id=T6, src=0}]
   3: F       F............................................... a: 01.05 09:05 [01.05 11:05]


)"sv;

}  // namespace

std::string results_to_str(pareto_set<routing::journey> const& results,
                           direction const dir,
                           timetable const& tt,
                           rt_timetable const* rtt = nullptr) {
  std::stringstream ss;
  ss << "\n";
  for (auto x : results) {
    // to allow for easier comparison, remove footpaths and adjust
    // start + dest time, so that the output is the same for
    // forward and backward search
    utl::erase_if(x.legs_, [](routing::journey::leg const& l) {
      return std::holds_alternative<footpath>(l.uses_);
    });
    if (dir == direction::kBackward) {
      auto const start_time = x.start_time_;
      x.start_time_ = x.dest_time_;
      x.dest_time_ = start_time;
    }
    x.print(ss, tt, rtt);
    ss << "\n\n";
  }
  return ss.str();
}

pareto_set<routing::journey> search(timetable const& tt,
                                    rt_timetable const* rtt,
                                    std::string_view const from,
                                    std::string_view const to,
                                    routing::start_time_t const start_time,
                                    direction const dir,
                                    bool const require_bikes_allowed) {
  return raptor_search(tt, rtt, dir == direction::kForward ? from : to,
                       dir == direction::kForward ? to : from, start_time, dir,
                       routing::all_clasz_allowed(), require_bikes_allowed);
}

TEST(routing, bike_transport_test_1) {
  auto tt = timetable{};

  tt.date_range_ = {date::sys_days{2019_y / May / 1},
                    date::sys_days{2019_y / May / 2}};
  loader::register_special_stations(tt);
  loader::gtfs::load_timetable({}, source_idx_t{0},
                               loader::mem_dir::read(test_files_1), tt);
  loader::finalize(tt);

  for (auto const dir : {direction::kForward, direction::kBackward}) {

    {  // A->C, without bike
      auto const results =
          search(tt, nullptr, "A", "C", tt.date_range_, dir, false);
      EXPECT_EQ(expected_A_C_no_bike, results_to_str(results, dir, tt));
    }

    {  // A->C, with bike
      auto const results =
          search(tt, nullptr, "A", "C", tt.date_range_, dir, true);
      EXPECT_EQ(expected_A_C_bike, results_to_str(results, dir, tt));
    }

    {  // C->D, without bike
      auto const results =
          search(tt, nullptr, "C", "D", tt.date_range_, dir, false);
      EXPECT_EQ(expected_C_D, results_to_str(results, dir, tt));
    }

    {  // C->D, with bike
      auto const results =
          search(tt, nullptr, "C", "D", tt.date_range_, dir, true);
      EXPECT_EQ(expected_C_D, results_to_str(results, dir, tt));
    }

    {  // A->D, without bike
      auto const results =
          search(tt, nullptr, "A", "D", tt.date_range_, dir, false);
      EXPECT_EQ(expected_A_D_no_bike, results_to_str(results, dir, tt));
    }

    {  // A->D, with bike
      auto const results =
          search(tt, nullptr, "A", "D", tt.date_range_, dir, true);
      EXPECT_EQ(expected_A_D_bike, results_to_str(results, dir, tt));
    }

    {  // A->F, without bike
      auto const results =
          search(tt, nullptr, "A", "F", tt.date_range_, dir, false);
      EXPECT_EQ(expected_A_F_no_bike, results_to_str(results, dir, tt));
    }

    {  // A->F, with bike
      auto const results =
          search(tt, nullptr, "A", "F", tt.date_range_, dir, true);
      EXPECT_EQ(expected_A_F_bike, results_to_str(results, dir, tt));
    }
  }
}

// Timetable 2:
// Bikes allowed in all trips except T1, blocks T1>T2 and T3>T4
// A -> D: 10:00 - 10:25 | T1, T2 (no bikes in T1)
//         10:00 - 10:35 | T3, T4 (bikes allowed)
constexpr auto const test_files2 = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,
C,C,,4.0,5.0,,
D,D,,6.0,7.0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,1,,,3
R2,DB,2,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id,bikes_allowed
R1,S1,T1,,1,2
R1,S1,T3,,2,1
R2,S1,T2,,1,1
R2,S1,T4,,2,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T1,10:00:00,10:00:00,A,0
T1,10:10:00,10:10:00,B,1
T2,10:15:00,10:15:00,B,0
T2,10:20:00,10:20:00,C,1
T2,10:25:00,10:25:00,D,2
T3,10:00:00,10:00:00,A,0
T3,10:20:00,10:20:00,B,1
T4,10:25:00,10:25:00,B,0
T4,10:30:00,10:30:00,C,1
T4,10:35:00,10:35:00,D,2

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)"sv;

constexpr auto const expected2_A_D_no_bike =
    R"(
[2019-05-01 08:00, 2019-05-01 08:25]
TRANSFERS: 0
     FROM: (A, A) [2019-05-01 08:00]
       TO: (D, D) [2019-05-01 08:25]
leg 0: (A, A) [2019-05-01 08:00] -> (D, D) [2019-05-01 08:25]
   0: A       A...............................................                               d: 01.05 08:00 [01.05 10:00]  [{name=Bus 1, day=2019-05-01, id=T1, src=0}]
   1: B       B............................................... a: 01.05 08:10 [01.05 10:10]  d: 01.05 08:15 [01.05 10:15]  [{name=Bus 2, day=2019-05-01, id=T2, src=0}]
   2: C       C............................................... a: 01.05 08:20 [01.05 10:20]  d: 01.05 08:20 [01.05 10:20]  [{name=Bus 2, day=2019-05-01, id=T2, src=0}]
   3: D       D............................................... a: 01.05 08:25 [01.05 10:25]


)"sv;

constexpr auto const expected2_A_D_bike =
    R"(
[2019-05-01 08:00, 2019-05-01 08:35]
TRANSFERS: 0
     FROM: (A, A) [2019-05-01 08:00]
       TO: (D, D) [2019-05-01 08:35]
leg 0: (A, A) [2019-05-01 08:00] -> (D, D) [2019-05-01 08:35]
   0: A       A...............................................                               d: 01.05 08:00 [01.05 10:00]  [{name=Bus 1, day=2019-05-01, id=T3, src=0}]
   1: B       B............................................... a: 01.05 08:20 [01.05 10:20]  d: 01.05 08:25 [01.05 10:25]  [{name=Bus 2, day=2019-05-01, id=T4, src=0}]
   2: C       C............................................... a: 01.05 08:30 [01.05 10:30]  d: 01.05 08:30 [01.05 10:30]  [{name=Bus 2, day=2019-05-01, id=T4, src=0}]
   3: D       D............................................... a: 01.05 08:35 [01.05 10:35]


)"sv;

constexpr auto const expected2_A_D_no_bike_rt =
    R"(
[2019-05-01 08:00, 2019-05-01 08:30]
TRANSFERS: 0
     FROM: (A, A) [2019-05-01 08:00]
       TO: (D, D) [2019-05-01 08:30]
leg 0: (A, A) [2019-05-01 08:00] -> (D, D) [2019-05-01 08:30]
   0: A       A...............................................                                                             d: 01.05 08:00 [01.05 10:00]  RT 01.05 08:00 [01.05 10:00]  [{name=Bus 1, day=2019-05-01, id=T1, src=0}]
   1: B       B............................................... a: 01.05 08:10 [01.05 10:10]  RT 01.05 08:10 [01.05 10:10]  d: 01.05 08:15 [01.05 10:15]  RT 01.05 08:20 [01.05 10:20]  [{name=Bus 2, day=2019-05-01, id=T2, src=0}]
   2: C       C............................................... a: 01.05 08:20 [01.05 10:20]  RT 01.05 08:25 [01.05 10:25]  d: 01.05 08:20 [01.05 10:20]  RT 01.05 08:25 [01.05 10:25]  [{name=Bus 2, day=2019-05-01, id=T2, src=0}]
   3: D       D............................................... a: 01.05 08:25 [01.05 10:25]  RT 01.05 08:30 [01.05 10:30]


)"sv;

constexpr auto const expected2_A_D_bike_rt =
    R"(
[2019-05-01 08:00, 2019-05-01 08:37]
TRANSFERS: 0
     FROM: (A, A) [2019-05-01 08:00]
       TO: (D, D) [2019-05-01 08:37]
leg 0: (A, A) [2019-05-01 08:00] -> (D, D) [2019-05-01 08:37]
   0: A       A...............................................                                                             d: 01.05 08:00 [01.05 10:00]  RT 01.05 08:00 [01.05 10:00]  [{name=Bus 1, day=2019-05-01, id=T3, src=0}]
   1: B       B............................................... a: 01.05 08:20 [01.05 10:20]  RT 01.05 08:20 [01.05 10:20]  d: 01.05 08:25 [01.05 10:25]  RT 01.05 08:27 [01.05 10:27]  [{name=Bus 2, day=2019-05-01, id=T4, src=0}]
   2: C       C............................................... a: 01.05 08:30 [01.05 10:30]  RT 01.05 08:32 [01.05 10:32]  d: 01.05 08:30 [01.05 10:30]  RT 01.05 08:32 [01.05 10:32]  [{name=Bus 2, day=2019-05-01, id=T4, src=0}]
   3: D       D............................................... a: 01.05 08:35 [01.05 10:35]  RT 01.05 08:37 [01.05 10:37]


)"sv;

TEST(routing, bike_transport_test_2) {
  auto tt = timetable{};

  tt.date_range_ = {date::sys_days{2019_y / May / 1},
                    date::sys_days{2019_y / May / 2}};
  loader::register_special_stations(tt);
  loader::gtfs::load_timetable({}, source_idx_t{0},
                               loader::mem_dir::read(test_files2), tt);
  loader::finalize(tt);

  for (auto const dir : {direction::kForward, direction::kBackward}) {
    {  // A->D, without bike
      auto const results =
          search(tt, nullptr, "A", "D", tt.date_range_, dir, false);
      EXPECT_EQ(expected2_A_D_no_bike, results_to_str(results, dir, tt));
    }

    {  // A->D, with bike
      auto const results =
          search(tt, nullptr, "A", "D", tt.date_range_, dir, true);
      EXPECT_EQ(expected2_A_D_bike, results_to_str(results, dir, tt));
    }
  }

  // RT Update
  auto rtt = rt::create_rt_timetable(tt, sys_days{2019_y / May / 1});
  auto const msg =
      test::to_feed_msg({test::trip{.trip_id_ = "T2",
                                    .delays_ = {{.stop_id_ = "B",
                                                 .ev_type_ = event_type::kDep,
                                                 .delay_minutes_ = 5}}},
                         test::trip{.trip_id_ = "T4",
                                    .delays_ = {{.stop_id_ = "B",
                                                 .ev_type_ = event_type::kDep,
                                                 .delay_minutes_ = 2}}}},
                        date::sys_days{2019_y / May / 1} + 1h);
  rt::gtfsrt_update_msg(tt, rtt, source_idx_t{0}, "", msg);

  for (auto const dir : {direction::kForward, direction::kBackward}) {
    {  // A->D, without bike
      auto const results =
          search(tt, &rtt, "A", "D", tt.date_range_, dir, false);
      EXPECT_EQ(expected2_A_D_no_bike_rt,
                results_to_str(results, dir, tt, &rtt));
    }

    {  // A->D, with bike
      auto const results =
          search(tt, &rtt, "A", "D", tt.date_range_, dir, true);
      EXPECT_EQ(expected2_A_D_bike_rt, results_to_str(results, dir, tt, &rtt));
    }
  }
}

// Timetable 3:
// Bikes allowed in all trips except T2, blocks T1>T2>T3 and T4>T5>T6
// A -> D: 10:00 - 10:30 | T1, T2, T3 (no bikes in T2)
//         10:00 - 10:40 | T4, T5, T6 (bikes allowed)
constexpr auto const test_files3 = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,
C,C,,4.0,5.0,,
D,D,,6.0,7.0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,1,,,3
R2,DB,2,,,3
R3,DB,3,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id,bikes_allowed
R1,S1,T1,,1,1
R1,S1,T4,,2,1
R2,S1,T2,,1,2
R2,S1,T5,,2,1
R3,S1,T3,,1,1
R3,S1,T6,,2,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T1,10:00:00,10:00:00,A,0
T1,10:10:00,10:10:00,B,1
T2,10:15:00,10:15:00,B,0
T2,10:20:00,10:20:00,C,1
T3,10:25:00,10:25:00,C,0
T3,10:30:00,10:30:00,D,1
T4,10:00:00,10:00:00,A,0
T4,10:20:00,10:20:00,B,1
T5,10:25:00,10:25:00,B,0
T5,10:30:00,10:30:00,C,1
T6,10:35:00,10:35:00,C,0
T6,10:40:00,10:40:00,D,1

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)"sv;

constexpr auto const expected3_A_D_no_bike =
    R"(
[2019-05-01 08:00, 2019-05-01 08:30]
TRANSFERS: 0
     FROM: (A, A) [2019-05-01 08:00]
       TO: (D, D) [2019-05-01 08:30]
leg 0: (A, A) [2019-05-01 08:00] -> (D, D) [2019-05-01 08:30]
   0: A       A...............................................                               d: 01.05 08:00 [01.05 10:00]  [{name=Bus 1, day=2019-05-01, id=T1, src=0}]
   1: B       B............................................... a: 01.05 08:10 [01.05 10:10]  d: 01.05 08:15 [01.05 10:15]  [{name=Bus 2, day=2019-05-01, id=T2, src=0}]
   2: C       C............................................... a: 01.05 08:20 [01.05 10:20]  d: 01.05 08:25 [01.05 10:25]  [{name=Bus 3, day=2019-05-01, id=T3, src=0}]
   3: D       D............................................... a: 01.05 08:30 [01.05 10:30]


)"sv;

constexpr auto const expected3_A_D_bike =
    R"(
[2019-05-01 08:00, 2019-05-01 08:40]
TRANSFERS: 0
     FROM: (A, A) [2019-05-01 08:00]
       TO: (D, D) [2019-05-01 08:40]
leg 0: (A, A) [2019-05-01 08:00] -> (D, D) [2019-05-01 08:40]
   0: A       A...............................................                               d: 01.05 08:00 [01.05 10:00]  [{name=Bus 1, day=2019-05-01, id=T4, src=0}]
   1: B       B............................................... a: 01.05 08:20 [01.05 10:20]  d: 01.05 08:25 [01.05 10:25]  [{name=Bus 2, day=2019-05-01, id=T5, src=0}]
   2: C       C............................................... a: 01.05 08:30 [01.05 10:30]  d: 01.05 08:35 [01.05 10:35]  [{name=Bus 3, day=2019-05-01, id=T6, src=0}]
   3: D       D............................................... a: 01.05 08:40 [01.05 10:40]


)"sv;

TEST(routing, bike_transport_test_3) {
  auto tt = timetable{};

  tt.date_range_ = {date::sys_days{2019_y / May / 1},
                    date::sys_days{2019_y / May / 2}};
  loader::register_special_stations(tt);
  loader::gtfs::load_timetable({}, source_idx_t{0},
                               loader::mem_dir::read(test_files3), tt);
  loader::finalize(tt);

  for (auto const dir : {direction::kForward, direction::kBackward}) {
    {  // A->D, without bike
      auto const results =
          search(tt, nullptr, "A", "D", tt.date_range_, dir, false);
      EXPECT_EQ(expected3_A_D_no_bike, results_to_str(results, dir, tt));
    }

    {  // A->D, with bike
      auto const results =
          search(tt, nullptr, "A", "D", tt.date_range_, dir, true);
      EXPECT_EQ(expected3_A_D_bike, results_to_str(results, dir, tt));
    }
  }
}
