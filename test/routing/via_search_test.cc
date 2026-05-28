#include "gtest/gtest.h"

#include <algorithm>
#include <regex>

#include "utl/erase_if.h"

#include "nigiri/common/parse_time.h"

#include "nigiri/loader/gtfs/agency.h"
#include "nigiri/loader/gtfs/calendar.h"
#include "nigiri/loader/gtfs/calendar_date.h"
#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/gtfs/local_to_utc.h"
#include "nigiri/loader/gtfs/noon_offsets.h"
#include "nigiri/loader/init_finish.h"

#include "nigiri/routing/raptor/pong.h"
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

std::string print_results(timetable const& tt,
                          pareto_set<nigiri::routing::journey> const& results) {
  std::stringstream ss;
  ss << "\n";
  for (auto const& x : results) {
    x.print(ss, tt);
    ss << "\n\n";
  }
  return ss.str();
}

constexpr auto const test_files_1 = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,
C,C,,4.0,5.0,,
D,D,,6.0,7.0,,
E,E,,8.0,9.0,,
F,F,,10.0,11.0,,
G,G,,12.0,13.0,,
H,H,,14.0,15.0,,
I,I,,16.0,17.0,1,
I1,I1,,16.0,17.0,,I
I2,I2,,16.0,17.0,,I
J,J,,18.0,19.0,,
K,K,,20.0,21.0,1,
K1,K1,,20.0,21.0,,K
K2,K2,,20.0,21.0,,K
L,L,,22.0,23.0,1,
M,M,,24.0,25.0,,
N,N,,26.0,27.0,,
O,O,,28.0,29.0,,
P,P,,30.0,31.0,,
Q,Q,,32.0,33.0,,
R,R,,34.0,35.0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DB,0,,,3
R1,DB,1,,,3
R2,DB,2,,,3
R3,DB,3,,,3
R4,DB,4,,,3
R5,DB,5,,,3
R6,DB,6,,,3
R7,DB,7,,,3
R8,DB,8,,,3
R9,DB,9,,,3
R10,DB,10,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,S1,T0,,
R1,S1,T1,,
R1,S1,T2,,
R2,S1,T3,,
R2,S1,T4,,
R3,S1,T5,,
R5,S1,T6,,
R4,S1,T7,,
R6,S1,T8,,
R7,S1,T9,,
R7,S1,T10,,
R8,S1,T11,,
R9,S1,T12,,
R2,S1,T13,,
R10,S1,T14,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T0,10:00:00,10:00:00,A,0
T0,10:10:00,10:10:00,B,1
T1,10:15:00,10:15:00,B,0
T1,10:30:00,10:30:00,D,1
T2,10:30:00,10:30:00,B,0
T2,10:45:00,10:45:00,D,1
T3,10:00:00,10:00:00,A,0
T3,10:20:00,10:22:00,C,1
T3,10:40:00,10:40:00,D,2
T4,10:15:00,10:15:00,A,0
T4,10:35:00,10:37:00,C,1
T4,10:55:00,10:55:00,D,2
T5,15:00:00,15:00:00,A,0
T5,15:15:00,15:17:00,I1,1
T5,15:30:00,15:30:00,J,2
T6,16:00:00,16:00:00,A,0
T6,16:15:00,16:17:00,I1,1
T7,16:20:00,16:20:00,I2,0
T7,16:40:00,16:40:00,J,1
T8,15:30:00,15:30:00,K1,0
T8,16:00:00,16:00:00,J,1
T9,11:00:00,11:00:00,M,0
T9,11:13:00,11:15:00,N,1
T9,11:28:00,11:30:00,O,2
T9,11:43:00,11:45:00,P,3
T9,12:00:00,12:00:00,Q,4
T10,12:00:00,12:00:00,M,0
T10,12:13:00,12:15:00,N,1
T10,12:28:00,12:30:00,O,2
T10,12:43:00,12:45:00,P,3
T10,13:00:00,13:00:00,Q,4
T11,11:00:00,11:00:00,H,0
T11,11:42:00,11:42:00,M,1
T12,11:00:00,11:00:00,H,0
T12,11:20:00,11:20:00,O,1
T13,10:30:00,10:30:00,A,0
T13,10:45:00,10:55:00,C,1
T13,11:30:00,11:30:00,D,2
T14,10:15:00,10:15:00,F,0
T14,10:42:00,10:45:00,G,1
T14,11:02:00,11:05:00,R,2
T14,12:00:00,12:02:00,L,3
T14,12:22:00,12:25:00,O,4
T14,12:40:00,12:40:00,P,5

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
E,A,2,300
D,H,2,180
I1,I2,2,240
I2,I1,2,240
I,K,2,300
K,I,2,300
C,G,2,300
D,R,2,900
)"sv;

std::string results_to_str(pareto_set<routing::journey> const& results,
                           timetable const& tt,
                           rt_timetable const* rtt = nullptr) {
  std::stringstream ss;
  ss << "\n";
  for (auto const& j : results) {
    j.print(ss, tt, rtt);
    ss << "\n\n";
  }
  return ss.str();
}

location_idx_t loc_idx(timetable const& tt, std::string_view const id) {
  return tt.find(location_id{id, source_idx_t{0}}).value();
}

interval<unixtime_t> iv(std::string_view const from, std::string_view to) {
  auto const time = [](std::string_view const x) {
    return parse_time_tz(x, "%Y-%m-%d %H:%M %Z");
  };
  return {time(from), time(to)};
}

timetable load_timetable(std::string_view s) {
  auto tt = timetable{};

  tt.date_range_ = {date::sys_days{2019_y / May / 1},
                    date::sys_days{2019_y / May / 2}};
  loader::register_special_stations(tt);
  loader::gtfs::load_timetable({}, source_idx_t{0}, loader::mem_dir::read(s),
                               tt);
  loader::finalize(tt);
  return tt;
}

pareto_set<routing::journey> search(timetable const& tt,
                                    rt_timetable const* rtt,
                                    routing::query q,
                                    direction const dir) {
  if (dir == direction::kBackward) {
    std::swap(q.start_, q.destination_);
    std::swap(q.start_match_mode_, q.dest_match_mode_);
    std::reverse(begin(q.via_stops_), end(q.via_stops_));
  }

  auto const rraptor_results = raptor_search(tt, rtt, q, dir);
  auto const rraptor_results_str = print_results(tt, rraptor_results);

  auto pong_results_str = std::string{};
  {
    auto search_state = routing::search_state{};
    auto raptor_state = routing::raptor_state{};
    auto const pong_results =
        routing::pong_search(tt, rtt, search_state, raptor_state, q, dir);
    pong_results_str = print_results(tt, *pong_results.journeys_);
  }

  EXPECT_EQ(rraptor_results_str, pong_results_str)
      << "dir=" << to_str(dir)  //
      << ", from=" << loc{tt, q.start_[0].target()}  //
      << ", to=" << loc{tt, q.destination_[0].target()}
      << ", interval=" << std::get<interval<unixtime_t>>(q.start_time_);

  return rraptor_results;
}

}  // namespace

TEST(routing, via_test_1_A_D_no_via) {
  // A -> D, no via
  auto const tt = load_timetable(test_files_1);

  constexpr auto const expected_A_D_no_via =
      R"(
[2019-05-01 08:00, 2019-05-01 08:40]
TRANSFERS: 0
     FROM: (A, A) [2019-05-01 08:00]
       TO: (D, D) [2019-05-01 08:40]
leg 0: (A, A) [2019-05-01 08:00] -> (D, D) [2019-05-01 08:40]
   0: A       A...............................................                               d: 01.05 08:00 [01.05 10:00]  [{name=2, day=2019-05-01, id=T3, src=0}]
   1: C       C............................................... a: 01.05 08:20 [01.05 10:20]  d: 01.05 08:22 [01.05 10:22]  [{name=2, day=2019-05-01, id=T3, src=0}]
   2: D       D............................................... a: 01.05 08:40 [01.05 10:40]


[2019-05-01 08:00, 2019-05-01 08:30]
TRANSFERS: 1
     FROM: (A, A) [2019-05-01 08:00]
       TO: (D, D) [2019-05-01 08:30]
leg 0: (A, A) [2019-05-01 08:00] -> (B, B) [2019-05-01 08:10]
   0: A       A...............................................                               d: 01.05 08:00 [01.05 10:00]  [{name=0, day=2019-05-01, id=T0, src=0}]
   1: B       B............................................... a: 01.05 08:10 [01.05 10:10]
leg 1: (B, B) [2019-05-01 08:10] -> (B, B) [2019-05-01 08:12]
  FOOTPATH (duration=2)
leg 2: (B, B) [2019-05-01 08:15] -> (D, D) [2019-05-01 08:30]
   0: B       B...............................................                               d: 01.05 08:15 [01.05 10:15]  [{name=1, day=2019-05-01, id=T1, src=0}]
   1: D       D............................................... a: 01.05 08:30 [01.05 10:30]


)"sv;

  auto const results =
      search(tt, nullptr,
             routing::query{.start_time_ = iv("2019-05-01 10:00 Europe/Berlin",
                                              "2019-05-01 10:01 Europe/Berlin"),
                            .start_ = {{loc_idx(tt, "A"), 0_minutes, 0U}},
                            .destination_ = {{loc_idx(tt, "D"), 0_minutes, 0U}},
                            .via_stops_ = {}},
             direction::kForward);

  EXPECT_EQ(expected_A_D_no_via, results_to_str(results, tt));
}

TEST(routing, via_test_2_A_D_via_B_0m) {
  // A -> D, via B (0 min)
  auto const tt = load_timetable(test_files_1);

  constexpr auto const expected_A_D_via_B_0min =
      R"(
[2019-05-01 08:00, 2019-05-01 08:30]
TRANSFERS: 1
     FROM: (A, A) [2019-05-01 08:00]
       TO: (D, D) [2019-05-01 08:30]
leg 0: (A, A) [2019-05-01 08:00] -> (B, B) [2019-05-01 08:10]
   0: A       A...............................................                               d: 01.05 08:00 [01.05 10:00]  [{name=0, day=2019-05-01, id=T0, src=0}]
   1: B       B............................................... a: 01.05 08:10 [01.05 10:10]
leg 1: (B, B) [2019-05-01 08:10] -> (B, B) [2019-05-01 08:12]
  FOOTPATH (duration=2)
leg 2: (B, B) [2019-05-01 08:15] -> (D, D) [2019-05-01 08:30]
   0: B       B...............................................                               d: 01.05 08:15 [01.05 10:15]  [{name=1, day=2019-05-01, id=T1, src=0}]
   1: D       D............................................... a: 01.05 08:30 [01.05 10:30]


)"sv;

  test::with_rt_trips(
      tt, sys_days{2019_y / May / 1}, {"T0", "T1"},
      [&](rt_timetable const* rtt, std::string_view rt_trips) {
        for (auto const& [dir, start_time] :
             {std::pair{direction::kForward,
                        iv("2019-05-01 10:00 Europe/Berlin",
                           "2019-05-01 10:01 Europe/Berlin")},
              std::pair{direction::kBackward,
                        iv("2019-05-01 10:30 Europe/Berlin",
                           "2019-05-01 10:31 Europe/Berlin")}}) {
          auto const results =
              search(tt, rtt,
                     routing::query{
                         .start_time_ = start_time,
                         .start_ = {{loc_idx(tt, "A"), 0_minutes, 0U}},
                         .destination_ = {{loc_idx(tt, "D"), 0_minutes, 0U}},
                         .via_stops_ = {{loc_idx(tt, "B"), 0_minutes}}},
                     dir);

          EXPECT_EQ(expected_A_D_via_B_0min, results_to_str(results, tt))
              << " dir: " << to_str(dir)  //
              << " start_time: " << start_time  //
              << " rt trips: " << rt_trips;
        }
      });
}

TEST(routing, via_test_3_A_D_via_B_3m) {
  // A -> D, via B (3 min)
  auto tt = load_timetable(test_files_1);

  constexpr auto const expected_A_D_via_B_3min =
      R"(
[2019-05-01 08:00, 2019-05-01 08:30]
TRANSFERS: 1
     FROM: (A, A) [2019-05-01 08:00]
       TO: (D, D) [2019-05-01 08:30]
leg 0: (A, A) [2019-05-01 08:00] -> (B, B) [2019-05-01 08:10]
   0: A       A...............................................                               d: 01.05 08:00 [01.05 10:00]  [{name=0, day=2019-05-01, id=T0, src=0}]
   1: B       B............................................... a: 01.05 08:10 [01.05 10:10]
leg 1: (B, B) [2019-05-01 08:13] -> (B, B) [2019-05-01 08:15]
  FOOTPATH (duration=2)
leg 2: (B, B) [2019-05-01 08:15] -> (D, D) [2019-05-01 08:30]
   0: B       B...............................................                               d: 01.05 08:15 [01.05 10:15]  [{name=1, day=2019-05-01, id=T1, src=0}]
   1: D       D............................................... a: 01.05 08:30 [01.05 10:30]


)"sv;

  for (auto const& [dir, start_time] :
       {std::pair{direction::kForward, iv("2019-05-01 10:00 Europe/Berlin",
                                          "2019-05-01 10:01 Europe/Berlin")},
        std::pair{direction::kBackward,
                  iv("2019-05-01 10:30 Europe/Berlin",
                     "2019-05-01 10:31 Europe/Berlin")}}) {
    auto const results = search(
        tt, nullptr,
        routing::query{.start_time_ = start_time,
                       .start_ = {{loc_idx(tt, "A"), 0_minutes, 0U}},
                       .destination_ = {{loc_idx(tt, "D"), 0_minutes, 0U}},
                       .via_stops_ = {{loc_idx(tt, "B"), 3_minutes}}},
        dir);

    EXPECT_EQ(expected_A_D_via_B_3min, results_to_str(results, tt));
  }
}

TEST(routing, via_test_4_A_D_via_B_10m) {
  // A -> D, via B (10 min)
  auto tt = load_timetable(test_files_1);

  constexpr auto const expected_A_D_via_B_10min =
      R"(
[2019-05-01 08:00, 2019-05-01 08:45]
TRANSFERS: 1
     FROM: (A, A) [2019-05-01 08:00]
       TO: (D, D) [2019-05-01 08:45]
leg 0: (A, A) [2019-05-01 08:00] -> (B, B) [2019-05-01 08:10]
   0: A       A...............................................                               d: 01.05 08:00 [01.05 10:00]  [{name=0, day=2019-05-01, id=T0, src=0}]
   1: B       B............................................... a: 01.05 08:10 [01.05 10:10]
leg 1: (B, B) [2019-05-01 08:20] -> (B, B) [2019-05-01 08:22]
  FOOTPATH (duration=2)
leg 2: (B, B) [2019-05-01 08:30] -> (D, D) [2019-05-01 08:45]
   0: B       B...............................................                               d: 01.05 08:30 [01.05 10:30]  [{name=1, day=2019-05-01, id=T2, src=0}]
   1: D       D............................................... a: 01.05 08:45 [01.05 10:45]


)"sv;

  test::with_rt_trips(
      tt, sys_days{2019_y / May / 1}, {"T0", "T2"},
      [&](rt_timetable const* rtt, std::string_view rt_trips) {
        for (auto const& [dir, start_time] :
             {std::pair{direction::kForward,
                        iv("2019-05-01 10:00 Europe/Berlin",
                           "2019-05-01 10:01 Europe/Berlin")},
              std::pair{direction::kBackward,
                        iv("2019-05-01 10:45 Europe/Berlin",
                           "2019-05-01 10:46 Europe/Berlin")}}) {
          auto const results =
              search(tt, rtt,
                     routing::query{
                         .start_time_ = start_time,
                         .start_ = {{loc_idx(tt, "A"), 0_minutes, 0U}},
                         .destination_ = {{loc_idx(tt, "D"), 0_minutes, 0U}},
                         .via_stops_ = {{loc_idx(tt, "B"), 10_minutes}}},
                     dir);

          EXPECT_EQ(expected_A_D_via_B_10min, results_to_str(results, tt))
              << " rt trips: " << rt_trips;
        }
      });
}

TEST(routing, via_test_5_A_D_via_C_0m) {
  // A -> D, via C (0 min)
  auto tt = load_timetable(test_files_1);

  constexpr auto const expected_A_D_via_C_0min =
      R"(
[2019-05-01 08:00, 2019-05-01 08:40]
TRANSFERS: 0
     FROM: (A, A) [2019-05-01 08:00]
       TO: (D, D) [2019-05-01 08:40]
leg 0: (A, A) [2019-05-01 08:00] -> (D, D) [2019-05-01 08:40]
   0: A       A...............................................                               d: 01.05 08:00 [01.05 10:00]  [{name=2, day=2019-05-01, id=T3, src=0}]
   1: C       C............................................... a: 01.05 08:20 [01.05 10:20]  d: 01.05 08:22 [01.05 10:22]  [{name=2, day=2019-05-01, id=T3, src=0}]
   2: D       D............................................... a: 01.05 08:40 [01.05 10:40]


)"sv;

  for (auto const& [dir, start_time] :
       {std::pair{direction::kForward, iv("2019-05-01 10:00 Europe/Berlin",
                                          "2019-05-01 10:01 Europe/Berlin")},
        std::pair{direction::kBackward,
                  iv("2019-05-01 10:40 Europe/Berlin",
                     "2019-05-01 10:41 Europe/Berlin")}}) {
    auto const results = search(
        tt, nullptr,
        routing::query{.start_time_ = start_time,
                       .start_ = {{loc_idx(tt, "A"), 0_minutes, 0U}},
                       .destination_ = {{loc_idx(tt, "D"), 0_minutes, 0U}},
                       .via_stops_ = {{loc_idx(tt, "C"), 0_minutes}}},
        dir);

    EXPECT_EQ(expected_A_D_via_C_0min, results_to_str(results, tt));
  }
}

TEST(routing, via_test_6_A_D_via_C_10m) {
  // A -> D, via C (10 min)
  auto tt = load_timetable(test_files_1);

  constexpr auto const expected_A_D_via_C_10min =
      R"(
[2019-05-01 08:00, 2019-05-01 08:55]
TRANSFERS: 1
     FROM: (A, A) [2019-05-01 08:00]
       TO: (D, D) [2019-05-01 08:55]
leg 0: (A, A) [2019-05-01 08:00] -> (C, C) [2019-05-01 08:20]
   0: A       A...............................................                               d: 01.05 08:00 [01.05 10:00]  [{name=2, day=2019-05-01, id=T3, src=0}]
   1: C       C............................................... a: 01.05 08:20 [01.05 10:20]
leg 1: (C, C) [2019-05-01 08:30] -> (C, C) [2019-05-01 08:32]
  FOOTPATH (duration=2)
leg 2: (C, C) [2019-05-01 08:37] -> (D, D) [2019-05-01 08:55]
   1: C       C...............................................                               d: 01.05 08:37 [01.05 10:37]  [{name=2, day=2019-05-01, id=T4, src=0}]
   2: D       D............................................... a: 01.05 08:55 [01.05 10:55]


)"sv;

  for (auto const& [dir, start_time] :
       {std::pair{direction::kForward, iv("2019-05-01 10:00 Europe/Berlin",
                                          "2019-05-01 10:01 Europe/Berlin")},
        std::pair{direction::kBackward,
                  iv("2019-05-01 10:55 Europe/Berlin",
                     "2019-05-01 10:56 Europe/Berlin")}}) {
    auto const results = search(
        tt, nullptr,
        routing::query{.start_time_ = start_time,
                       .start_ = {{loc_idx(tt, "A"), 0_minutes, 0U}},
                       .destination_ = {{loc_idx(tt, "D"), 0_minutes, 0U}},
                       .via_stops_ = {{loc_idx(tt, "C"), 10_minutes}}},
        dir);

    EXPECT_EQ(expected_A_D_via_C_10min, results_to_str(results, tt));
  }
}

namespace {
constexpr auto const expected_A_J_via_I_1500_0min =
    R"(
[2019-05-01 13:00, 2019-05-01 13:30]
TRANSFERS: 0
     FROM: (A, A) [2019-05-01 13:00]
       TO: (J, J) [2019-05-01 13:30]
leg 0: (A, A) [2019-05-01 13:00] -> (J, J) [2019-05-01 13:30]
   0: A       A...............................................                               d: 01.05 13:00 [01.05 15:00]  [{name=3, day=2019-05-01, id=T5, src=0}]
   1: I1      I............................................... a: 01.05 13:15 [01.05 15:15]  d: 01.05 13:17 [01.05 15:17]  [{name=3, day=2019-05-01, id=T5, src=0}]
   2: J       J............................................... a: 01.05 13:30 [01.05 15:30]


)"sv;
}  // namespace

TEST(routing, via_test_7_A_J_via_I_0m) {
  // A -> J, via I (0 min)
  auto tt = load_timetable(test_files_1);

  for (auto const& [dir, start_time] :
       {std::pair{direction::kForward, iv("2019-05-01 15:00 Europe/Berlin",
                                          "2019-05-01 15:01 Europe/Berlin")},
        std::pair{direction::kBackward,
                  iv("2019-05-01 15:30 Europe/Berlin",
                     "2019-05-01 15:31 Europe/Berlin")}}) {
    auto const results = search(
        tt, nullptr,
        routing::query{.start_time_ = start_time,
                       .start_ = {{loc_idx(tt, "A"), 0_minutes, 0U}},
                       .destination_ = {{loc_idx(tt, "J"), 0_minutes, 0U}},
                       .via_stops_ = {{loc_idx(tt, "I"), 0_minutes}}},
        dir);

    EXPECT_EQ(expected_A_J_via_I_1500_0min, results_to_str(results, tt));
  }
}

TEST(routing, via_test_8_A_J_via_I1_0m) {
  // A -> J, via I1 (0 min)
  auto tt = load_timetable(test_files_1);

  for (auto const& [dir, start_time] :
       {std::pair{direction::kForward, iv("2019-05-01 15:00 Europe/Berlin",
                                          "2019-05-01 15:01 Europe/Berlin")},
        std::pair{direction::kBackward,
                  iv("2019-05-01 15:30 Europe/Berlin",
                     "2019-05-01 15:31 Europe/Berlin")}}) {
    auto const results = search(
        tt, nullptr,
        routing::query{.start_time_ = start_time,
                       .start_ = {{loc_idx(tt, "A"), 0_minutes, 0U}},
                       .destination_ = {{loc_idx(tt, "J"), 0_minutes, 0U}},
                       .via_stops_ = {{loc_idx(tt, "I1"), 0_minutes}}},
        dir);

    EXPECT_EQ(expected_A_J_via_I_1500_0min, results_to_str(results, tt));
  }
}

TEST(routing, via_test_9_A_J_via_I2_0m) {
  // A -> J, via I2 (0 min)
  auto tt = load_timetable(test_files_1);

  for (auto const& [dir, start_time] :
       {std::pair{direction::kForward, iv("2019-05-01 15:00 Europe/Berlin",
                                          "2019-05-01 15:01 Europe/Berlin")},
        std::pair{direction::kBackward,
                  iv("2019-05-01 15:30 Europe/Berlin",
                     "2019-05-01 15:31 Europe/Berlin")}}) {
    auto const results = search(
        tt, nullptr,
        routing::query{.start_time_ = start_time,
                       .start_ = {{loc_idx(tt, "A"), 0_minutes, 0U}},
                       .destination_ = {{loc_idx(tt, "J"), 0_minutes, 0U}},
                       .via_stops_ = {{loc_idx(tt, "I2"), 0_minutes}}},
        dir);

    EXPECT_EQ(expected_A_J_via_I_1500_0min, results_to_str(results, tt));
  }
}

TEST(routing, via_test_10_A_J_via_I_0m) {
  // A -> J, via I (0 min)
  auto tt = load_timetable(test_files_1);

  constexpr auto const expected_A_J_via_I_1600_0min =
      R"(
[2019-05-01 13:00, 2019-05-01 13:30]
TRANSFERS: 0
     FROM: (A, A) [2019-05-01 13:00]
       TO: (J, J) [2019-05-01 13:30]
leg 0: (A, A) [2019-05-01 13:00] -> (J, J) [2019-05-01 13:30]
   0: A       A...............................................                               d: 01.05 13:00 [01.05 15:00]  [{name=3, day=2019-05-01, id=T5, src=0}]
   1: I1      I............................................... a: 01.05 13:15 [01.05 15:15]  d: 01.05 13:17 [01.05 15:17]  [{name=3, day=2019-05-01, id=T5, src=0}]
   2: J       J............................................... a: 01.05 13:30 [01.05 15:30]


[2019-05-01 14:00, 2019-05-01 14:40]
TRANSFERS: 1
     FROM: (A, A) [2019-05-01 14:00]
       TO: (J, J) [2019-05-01 14:40]
leg 0: (A, A) [2019-05-01 14:00] -> (I1, I1) [2019-05-01 14:15]
   0: A       A...............................................                               d: 01.05 14:00 [01.05 16:00]  [{name=5, day=2019-05-01, id=T6, src=0}]
   1: I1      I............................................... a: 01.05 14:15 [01.05 16:15]
leg 1: (I1, I1) [2019-05-01 14:15] -> (I2, I2) [2019-05-01 14:19]
  FOOTPATH (duration=4)
leg 2: (I2, I2) [2019-05-01 14:20] -> (J, J) [2019-05-01 14:40]
   0: I2      I...............................................                               d: 01.05 14:20 [01.05 16:20]  [{name=4, day=2019-05-01, id=T7, src=0}]
   1: J       J............................................... a: 01.05 14:40 [01.05 16:40]


)"sv;

  for (auto const& dir : {direction::kForward, direction::kBackward}) {
    auto const results = search(
        tt, nullptr,
        routing::query{.start_time_ = tt.date_range_,
                       .start_ = {{loc_idx(tt, "A"), 0_minutes, 0U}},
                       .destination_ = {{loc_idx(tt, "J"), 0_minutes, 0U}},
                       .via_stops_ = {{loc_idx(tt, "I"), 0_minutes}}},
        dir);

    EXPECT_EQ(expected_A_J_via_I_1600_0min, results_to_str(results, tt));
  }
}

TEST(routing, via_test_11_A_J_via_I_10m) {
  // A -> J, via I (10 min)
  auto tt = load_timetable(test_files_1);

  constexpr auto const expected_A_J_via_I_1500_10min =
      R"(
[2019-05-01 13:00, 2019-05-01 14:40]
TRANSFERS: 1
     FROM: (A, A) [2019-05-01 13:00]
       TO: (J, J) [2019-05-01 14:40]
leg 0: (A, A) [2019-05-01 13:00] -> (I1, I1) [2019-05-01 13:15]
   0: A       A...............................................                               d: 01.05 13:00 [01.05 15:00]  [{name=3, day=2019-05-01, id=T5, src=0}]
   1: I1      I............................................... a: 01.05 13:15 [01.05 15:15]
leg 1: (I1, I1) [2019-05-01 13:25] -> (I2, I2) [2019-05-01 13:29]
  FOOTPATH (duration=4)
leg 2: (I2, I2) [2019-05-01 14:20] -> (J, J) [2019-05-01 14:40]
   0: I2      I...............................................                               d: 01.05 14:20 [01.05 16:20]  [{name=4, day=2019-05-01, id=T7, src=0}]
   1: J       J............................................... a: 01.05 14:40 [01.05 16:40]


)"sv;

  for (auto const& [dir, start_time] :
       {std::pair{direction::kForward, iv("2019-05-01 15:00 Europe/Berlin",
                                          "2019-05-01 15:01 Europe/Berlin")},
        std::pair{direction::kBackward,
                  iv("2019-05-01 16:40 Europe/Berlin",
                     "2019-05-01 16:41 Europe/Berlin")}}) {
    auto const results = search(
        tt, nullptr,
        routing::query{.start_time_ = start_time,
                       .start_ = {{loc_idx(tt, "A"), 0_minutes, 0U}},
                       .destination_ = {{loc_idx(tt, "J"), 0_minutes, 0U}},
                       .via_stops_ = {{loc_idx(tt, "I"), 10_minutes}}},
        dir);

    EXPECT_EQ(expected_A_J_via_I_1500_10min, results_to_str(results, tt));
  }
}

namespace {
constexpr auto const expected_A_J_via_K_0min =
    R"(
[2019-05-01 13:00, 2019-05-01 14:00]
TRANSFERS: 1
     FROM: (A, A) [2019-05-01 13:00]
       TO: (J, J) [2019-05-01 14:00]
leg 0: (A, A) [2019-05-01 13:00] -> (I1, I1) [2019-05-01 13:15]
   0: A       A...............................................                               d: 01.05 13:00 [01.05 15:00]  [{name=3, day=2019-05-01, id=T5, src=0}]
   1: I1      I............................................... a: 01.05 13:15 [01.05 15:15]
leg 1: (I1, I1) [2019-05-01 13:15] -> (K1, K1) [2019-05-01 13:24]
  FOOTPATH (duration=9)
leg 2: (K1, K1) [2019-05-01 13:30] -> (J, J) [2019-05-01 14:00]
   0: K1      K...............................................                               d: 01.05 13:30 [01.05 15:30]  [{name=6, day=2019-05-01, id=T8, src=0}]
   1: J       J............................................... a: 01.05 14:00 [01.05 16:00]


)"sv;
}  // namespace

TEST(routing, via_test_12_A_J_via_K_0m) {
  // A -> J, via K (0 min)
  auto tt = load_timetable(test_files_1);

  for (auto const& [dir, start_time] :
       {std::pair{direction::kForward, iv("2019-05-01 15:00 Europe/Berlin",
                                          "2019-05-01 15:01 Europe/Berlin")},
        std::pair{direction::kBackward,
                  iv("2019-05-01 16:00 Europe/Berlin",
                     "2019-05-01 16:01 Europe/Berlin")}}) {
    auto const results = search(
        tt, nullptr,
        routing::query{.start_time_ = start_time,
                       .start_ = {{loc_idx(tt, "A"), 0_minutes, 0U}},
                       .destination_ = {{loc_idx(tt, "J"), 0_minutes, 0U}},
                       .via_stops_ = {{loc_idx(tt, "K"), 0_minutes}}},
        dir);

    EXPECT_EQ(expected_A_J_via_K_0min, results_to_str(results, tt));
  }
}

TEST(routing, via_test_13_A_J_via_K_5m) {
  // A -> J, via K (5 min)
  auto tt = load_timetable(test_files_1);

  for (auto const& [dir, start_time] :
       {std::pair{direction::kForward, iv("2019-05-01 15:00 Europe/Berlin",
                                          "2019-05-01 15:01 Europe/Berlin")},
        std::pair{direction::kBackward,
                  iv("2019-05-01 16:00 Europe/Berlin",
                     "2019-05-01 16:01 Europe/Berlin")}}) {
    auto const results = search(
        tt, nullptr,
        routing::query{.start_time_ = start_time,
                       .start_ = {{loc_idx(tt, "A"), 0_minutes, 0U}},
                       .destination_ = {{loc_idx(tt, "J"), 0_minutes, 0U}},
                       .via_stops_ = {{loc_idx(tt, "K"), 5_minutes}}},
        dir);

    EXPECT_EQ(expected_A_J_via_K_0min, results_to_str(results, tt));
  }
}

TEST(routing, via_test_14_A_J_via_I_0m_K_5m) {
  // A -> J, via I (0 min), K (5 min)
  auto tt = load_timetable(test_files_1);

  for (auto const& [dir, start_time] :
       {std::pair{direction::kForward, iv("2019-05-01 15:00 Europe/Berlin",
                                          "2019-05-01 15:01 Europe/Berlin")},
        std::pair{direction::kBackward,
                  iv("2019-05-01 16:00 Europe/Berlin",
                     "2019-05-01 16:01 Europe/Berlin")}}) {
    auto const results = search(
        tt, nullptr,
        routing::query{.start_time_ = start_time,
                       .start_ = {{loc_idx(tt, "A"), 0_minutes, 0U}},
                       .destination_ = {{loc_idx(tt, "J"), 0_minutes, 0U}},
                       .via_stops_ = {{loc_idx(tt, "I"), 0_minutes},
                                      {loc_idx(tt, "K"), 5_minutes}}},
        dir);

    EXPECT_EQ(expected_A_J_via_K_0min, results_to_str(results, tt));
  }
}

namespace {
constexpr auto const expected_H_Q_via_N_0min =
    R"(
[2019-05-01 09:00, 2019-05-01 11:00]
TRANSFERS: 1
     FROM: (H, H) [2019-05-01 09:00]
       TO: (Q, Q) [2019-05-01 11:00]
leg 0: (H, H) [2019-05-01 09:00] -> (M, M) [2019-05-01 09:42]
   0: H       H...............................................                               d: 01.05 09:00 [01.05 11:00]  [{name=8, day=2019-05-01, id=T11, src=0}]
   1: M       M............................................... a: 01.05 09:42 [01.05 11:42]
leg 1: (M, M) [2019-05-01 09:42] -> (M, M) [2019-05-01 09:44]
  FOOTPATH (duration=2)
leg 2: (M, M) [2019-05-01 10:00] -> (Q, Q) [2019-05-01 11:00]
   0: M       M...............................................                               d: 01.05 10:00 [01.05 12:00]  [{name=7, day=2019-05-01, id=T10, src=0}]
   1: N       N............................................... a: 01.05 10:13 [01.05 12:13]  d: 01.05 10:15 [01.05 12:15]  [{name=7, day=2019-05-01, id=T10, src=0}]
   2: O       O............................................... a: 01.05 10:28 [01.05 12:28]  d: 01.05 10:30 [01.05 12:30]  [{name=7, day=2019-05-01, id=T10, src=0}]
   3: P       P............................................... a: 01.05 10:43 [01.05 12:43]  d: 01.05 10:45 [01.05 12:45]  [{name=7, day=2019-05-01, id=T10, src=0}]
   4: Q       Q............................................... a: 01.05 11:00 [01.05 13:00]


)"sv;
}  // namespace

TEST(routing, via_test_15_H_Q_via_N_0m) {
  // H -> Q, via N (0 min)
  auto tt = load_timetable(test_files_1);

  for (auto const& [dir, start_time] :
       {std::pair{direction::kForward, iv("2019-05-01 11:00 Europe/Berlin",
                                          "2019-05-01 11:01 Europe/Berlin")},
        std::pair{direction::kBackward,
                  iv("2019-05-01 13:00 Europe/Berlin",
                     "2019-05-01 13:01 Europe/Berlin")}}) {
    auto const results = search(
        tt, nullptr,
        routing::query{.start_time_ = start_time,
                       .start_ = {{loc_idx(tt, "H"), 0_minutes, 0U}},
                       .destination_ = {{loc_idx(tt, "Q"), 0_minutes, 0U}},
                       .via_stops_ = {{loc_idx(tt, "N"), 0_minutes}}},
        dir);

    EXPECT_EQ(expected_H_Q_via_N_0min, results_to_str(results, tt));
  }
}

TEST(routing, via_test_16_A_D_via_C_5m) {
  // A -> D, via C (5 min)
  auto tt = load_timetable(test_files_1);

  constexpr auto const expected_A_D_via_C_5min =
      R"(
[2019-05-01 08:30, 2019-05-01 09:30]
TRANSFERS: 1
     FROM: (A, A) [2019-05-01 08:30]
       TO: (D, D) [2019-05-01 09:30]
leg 0: (A, A) [2019-05-01 08:30] -> (C, C) [2019-05-01 08:45]
   0: A       A...............................................                               d: 01.05 08:30 [01.05 10:30]  [{name=2, day=2019-05-01, id=T13, src=0}]
   1: C       C............................................... a: 01.05 08:45 [01.05 10:45]
leg 1: (C, C) [2019-05-01 08:50] -> (C, C) [2019-05-01 08:52]
  FOOTPATH (duration=2)
leg 2: (C, C) [2019-05-01 08:55] -> (D, D) [2019-05-01 09:30]
   1: C       C...............................................                               d: 01.05 08:55 [01.05 10:55]  [{name=2, day=2019-05-01, id=T13, src=0}]
   2: D       D............................................... a: 01.05 09:30 [01.05 11:30]


)"sv;

  for (auto const& [dir, start_time] :
       {/*std::pair{direction::kForward,
                  iv("2019-05-01 10:30 Europe/Berlin",
                     "2019-05-01 10:31 Europe/Berlin")}
           ,*/
        std::pair{direction::kBackward,
                  iv("2019-05-01 11:30 Europe/Berlin",
                     "2019-05-01 11:31 Europe/Berlin")}}) {
    auto const results = search(
        tt, nullptr,
        routing::query{.start_time_ = start_time,
                       .start_ = {{loc_idx(tt, "A"), 0_minutes, 0U}},
                       .destination_ = {{loc_idx(tt, "D"), 0_minutes, 0U}},
                       .via_stops_ = {{loc_idx(tt, "C"), 5_minutes}}},
        dir);

    EXPECT_EQ(expected_A_D_via_C_5min, results_to_str(results, tt));
  }
}

namespace {
constexpr auto const expected_intermodal_HO_Q_via_P_0min =
    R"(
[2019-05-01 09:00, 2019-05-01 10:00]
TRANSFERS: 1
     FROM: (START, START) [2019-05-01 09:00]
       TO: (Q, Q) [2019-05-01 10:00]
leg 0: (START, START) [2019-05-01 09:00] -> (H, H) [2019-05-01 09:00]
  MUMO (id=0, duration=0)
leg 1: (H, H) [2019-05-01 09:00] -> (O, O) [2019-05-01 09:20]
   0: H       H...............................................                               d: 01.05 09:00 [01.05 11:00]  [{name=9, day=2019-05-01, id=T12, src=0}]
   1: O       O............................................... a: 01.05 09:20 [01.05 11:20]
leg 2: (O, O) [2019-05-01 09:20] -> (O, O) [2019-05-01 09:22]
  FOOTPATH (duration=2)
leg 3: (O, O) [2019-05-01 09:30] -> (Q, Q) [2019-05-01 10:00]
   2: O       O...............................................                               d: 01.05 09:30 [01.05 11:30]  [{name=7, day=2019-05-01, id=T9, src=0}]
   3: P       P............................................... a: 01.05 09:43 [01.05 11:43]  d: 01.05 09:45 [01.05 11:45]  [{name=7, day=2019-05-01, id=T9, src=0}]
   4: Q       Q............................................... a: 01.05 10:00 [01.05 12:00]


[2019-05-01 09:50, 2019-05-01 11:00]
TRANSFERS: 0
     FROM: (START, START) [2019-05-01 09:50]
       TO: (Q, Q) [2019-05-01 11:00]
leg 0: (START, START) [2019-05-01 09:50] -> (O, O) [2019-05-01 10:30]
  MUMO (id=1, duration=40)
leg 1: (O, O) [2019-05-01 10:30] -> (Q, Q) [2019-05-01 11:00]
   2: O       O...............................................                               d: 01.05 10:30 [01.05 12:30]  [{name=7, day=2019-05-01, id=T10, src=0}]
   3: P       P............................................... a: 01.05 10:43 [01.05 12:43]  d: 01.05 10:45 [01.05 12:45]  [{name=7, day=2019-05-01, id=T10, src=0}]
   4: Q       Q............................................... a: 01.05 11:00 [01.05 13:00]


)"sv;
}  // namespace

TEST(routing, via_test_17_H_Q_via_N_0m_P_0m) {
  // H -> Q, via N (0 min), P (0 min)
  auto tt = load_timetable(test_files_1);

  test::with_rt_trips(
      tt, sys_days{2019_y / May / 1}, {"T11", "T10"},
      [&](rt_timetable const* rtt, std::string_view rt_trips) {
        for (auto const& [dir, start_time] :
             {std::pair{direction::kForward,
                        iv("2019-05-01 11:00 Europe/Berlin",
                           "2019-05-01 11:51 Europe/Berlin")},
              std::pair{direction::kBackward,
                        iv("2019-05-01 12:00 Europe/Berlin",
                           "2019-05-01 13:01 Europe/Berlin")}}) {
          auto const results =
              search(tt, rtt,
                     routing::query{
                         .start_time_ = start_time,
                         .start_ = {{loc_idx(tt, "H"), 0_minutes, 0U}},
                         .destination_ = {{loc_idx(tt, "Q"), 0_minutes, 0U}},
                         .via_stops_ = {{loc_idx(tt, "N"), 0_minutes},
                                        {loc_idx(tt, "P"), 0_minutes}}},
                     dir);

          EXPECT_EQ(expected_H_Q_via_N_0min, results_to_str(results, tt))
              << " rt trips: " << rt_trips;
        }
      });
}

TEST(routing, via_test_18_intermodal_HO_Q_via_P_0m) {
  // intermodal start: H / O -> Q, via P (0 min)
  auto tt = load_timetable(test_files_1);

  test::with_rt_trips(
      tt, sys_days{2019_y / May / 1}, {"T11", "T10"},
      [&](rt_timetable const* rtt, std::string_view rt_trips) {
        auto const results = search(
            tt, rtt,
            routing::query{
                .start_time_ = iv("2019-05-01 11:00 Europe/Berlin",
                                  "2019-05-01 11:51 Europe/Berlin"),
                .start_match_mode_ = routing::location_match_mode::kIntermodal,
                .start_ = {{loc_idx(tt, "H"), 0_minutes, 0U},
                           {loc_idx(tt, "O"), 40_minutes, 1U}},
                .destination_ = {{loc_idx(tt, "Q"), 0_minutes, 0U}},
                .via_stops_ = {{loc_idx(tt, "P"), 0_minutes}}},
            direction::kForward);

        EXPECT_EQ(expected_intermodal_HO_Q_via_P_0min,
                  results_to_str(results, tt))
            << " rt trips: " << rt_trips;
      });
}

TEST(routing, via_test_19_intermodal_HO_Q_via_O_0m_P_0m) {
  // intermodal start: H / O -> Q, via O (0 min), P (0 min)
  auto tt = load_timetable(test_files_1);

  test::with_rt_trips(
      tt, sys_days{2019_y / May / 1}, {"T11", "T10"},
      [&](rt_timetable const* rtt, std::string_view rt_trips) {
        auto const results = search(
            tt, rtt,
            routing::query{
                .start_time_ = iv("2019-05-01 11:00 Europe/Berlin",
                                  "2019-05-01 12:00 Europe/Berlin"),
                .start_match_mode_ = routing::location_match_mode::kIntermodal,
                .start_ = {{loc_idx(tt, "H"), 0_minutes, 0U},
                           {loc_idx(tt, "O"), 40_minutes, 1U}},
                .destination_ = {{loc_idx(tt, "Q"), 0_minutes, 0U}},
                .via_stops_ = {{loc_idx(tt, "O"), 0_minutes},
                               {loc_idx(tt, "P"), 0_minutes}}},
            direction::kForward);

        EXPECT_EQ(expected_intermodal_HO_Q_via_P_0min,
                  results_to_str(results, tt))
            << " rt trips: " << rt_trips;
      });
}

TEST(routing, via_test_20_N_intermodal_P_via_P_0m) {
  // intermodal dest: N -> P, via P (0 min)
  auto tt = load_timetable(test_files_1);

  constexpr auto const expected_intermodal_N_P_via_P_0min =
      R"(
[2019-05-01 09:15, 2019-05-01 09:53]
TRANSFERS: 0
     FROM: (N, N) [2019-05-01 09:15]
       TO: (END, END) [2019-05-01 09:53]
leg 0: (N, N) [2019-05-01 09:15] -> (P, P) [2019-05-01 09:43]
   1: N       N...............................................                               d: 01.05 09:15 [01.05 11:15]  [{name=7, day=2019-05-01, id=T9, src=0}]
   2: O       O............................................... a: 01.05 09:28 [01.05 11:28]  d: 01.05 09:30 [01.05 11:30]  [{name=7, day=2019-05-01, id=T9, src=0}]
   3: P       P............................................... a: 01.05 09:43 [01.05 11:43]
leg 1: (P, P) [2019-05-01 09:43] -> (END, END) [2019-05-01 09:53]
  MUMO (id=0, duration=10)


)"sv;

  test::with_rt_trips(
      tt, sys_days{2019_y / May / 1}, {"T11", "T10"},
      [&](rt_timetable const* rtt, std::string_view rt_trips) {
        for (auto const& [dir, start_time] :
             {std::pair{direction::kForward,
                        iv("2019-05-01 11:15 Europe/Berlin",
                           "2019-05-01 11:16 Europe/Berlin")},
              std::pair{direction::kBackward,
                        iv("2019-05-01 11:53 Europe/Berlin",
                           "2019-05-01 11:54 Europe/Berlin")}}) {
          auto const results = search(
              tt, rtt,
              routing::query{
                  .start_time_ = start_time,
                  .dest_match_mode_ = routing::location_match_mode::kIntermodal,
                  .start_ =
                      {
                          {loc_idx(tt, "N"), 0_minutes, 0U},
                      },
                  .destination_ = {{loc_idx(tt, "P"), 10_minutes, 0U}},
                  .via_stops_ = {{loc_idx(tt, "P"), 0_minutes}}},
              dir);

          auto results_str = results_to_str(results, tt);
          if (dir == direction::kBackward) {
            results_str =
                std::regex_replace(results_str, std::regex("START"), "END");
          }

          EXPECT_EQ(expected_intermodal_N_P_via_P_0min, results_str)
              << " rt trips: " << rt_trips;
        }
      });
}

TEST(routing, via_test_21_H_Q_via_H_0m_N_0m) {
  // H -> Q, via H (0 min), N (0 min)
  auto tt = load_timetable(test_files_1);

  for (auto const& [dir, start_time] :
       {std::pair{direction::kForward, iv("2019-05-01 11:00 Europe/Berlin",
                                          "2019-05-01 11:01 Europe/Berlin")},
        std::pair{direction::kBackward,
                  iv("2019-05-01 13:00 Europe/Berlin",
                     "2019-05-01 13:01 Europe/Berlin")}}) {
    auto const results = search(
        tt, nullptr,
        routing::query{.start_time_ = start_time,
                       .start_ = {{loc_idx(tt, "H"), 0_minutes, 0U}},
                       .destination_ = {{loc_idx(tt, "Q"), 0_minutes, 0U}},
                       .via_stops_ = {{loc_idx(tt, "H"), 0_minutes},
                                      {loc_idx(tt, "N"), 0_minutes}}},
        dir);

    EXPECT_EQ(expected_H_Q_via_N_0min, results_to_str(results, tt));
  }
}

/*
// this test needs kMaxVias = 3, currently disabled because of kMaxVias = 2
TEST(routing, via_test_22_H_Q_via_H_0m_N_0m_Q_0m) {
  // H -> Q, via H (0 min), N (0 min), Q (0 min)
  auto tt = load_timetable(test_files_1);

  for (auto const& [dir, start_time] :
       {std::pair{direction::kForward,
                  iv("2019-05-01 11:00 Europe/Berlin",
                     "2019-05-01 11:01 Europe/Berlin")},
        std::pair{direction::kBackward,
                  iv("2019-05-01 13:00 Europe/Berlin",
                     "2019-05-01 13:01 Europe/Berlin")}}) {
    auto const results = search(
        tt, nullptr,
        routing::query{.start_time_ = start_time,
                       .start_ = {{loc_idx(tt, "H"), 0_minutes, 0U}},
                       .destination_ = {{loc_idx(tt, "Q"), 0_minutes, 0U}},
                       .via_stops_ = {{loc_idx(tt, "H"), 0_minutes},
                                      {loc_idx(tt, "N"), 0_minutes},
                                      {loc_idx(tt, "Q"), 0_minutes}}},
        dir);

    EXPECT_EQ(expected_H_Q_via_N_0min, results_to_str(results, tt));
  }
}
*/

namespace {
constexpr auto const expected_M_Q_via_M_0min_O_0min =
    R"(
[2019-05-01 09:00, 2019-05-01 10:00]
TRANSFERS: 0
     FROM: (M, M) [2019-05-01 09:00]
       TO: (Q, Q) [2019-05-01 10:00]
leg 0: (M, M) [2019-05-01 09:00] -> (Q, Q) [2019-05-01 10:00]
   0: M       M...............................................                               d: 01.05 09:00 [01.05 11:00]  [{name=7, day=2019-05-01, id=T9, src=0}]
   1: N       N............................................... a: 01.05 09:13 [01.05 11:13]  d: 01.05 09:15 [01.05 11:15]  [{name=7, day=2019-05-01, id=T9, src=0}]
   2: O       O............................................... a: 01.05 09:28 [01.05 11:28]  d: 01.05 09:30 [01.05 11:30]  [{name=7, day=2019-05-01, id=T9, src=0}]
   3: P       P............................................... a: 01.05 09:43 [01.05 11:43]  d: 01.05 09:45 [01.05 11:45]  [{name=7, day=2019-05-01, id=T9, src=0}]
   4: Q       Q............................................... a: 01.05 10:00 [01.05 12:00]


)"sv;
}  // namespace

TEST(routing, via_test_23_M_Q_via_M_0m_O_0m) {
  // test: first via = start
  // M -> Q, via M (0 min), O (0 min)
  auto tt = load_timetable(test_files_1);

  for (auto const& [dir, start_time] :
       {std::pair{direction::kForward, iv("2019-05-01 11:00 Europe/Berlin",
                                          "2019-05-01 11:01 Europe/Berlin")},
        std::pair{direction::kBackward,
                  iv("2019-05-01 12:00 Europe/Berlin",
                     "2019-05-01 12:01 Europe/Berlin")}}) {
    auto const results = search(
        tt, nullptr,
        routing::query{.start_time_ = start_time,
                       .start_ = {{loc_idx(tt, "M"), 0_minutes, 0U}},
                       .destination_ = {{loc_idx(tt, "Q"), 0_minutes, 0U}},
                       .via_stops_ = {{loc_idx(tt, "M"), 0_minutes},
                                      {loc_idx(tt, "O"), 0_minutes}}},
        dir);

    EXPECT_EQ(expected_M_Q_via_M_0min_O_0min, results_to_str(results, tt));
  }
}

TEST(routing, via_test_24_M_Q_via_O_0m_Q_0m) {
  // test: last via = destination
  // M -> Q, via O (0 min), Q (0 min)
  auto tt = load_timetable(test_files_1);

  for (auto const& [dir, start_time] :
       {std::pair{direction::kForward, iv("2019-05-01 11:00 Europe/Berlin",
                                          "2019-05-01 11:01 Europe/Berlin")},
        std::pair{direction::kBackward,
                  iv("2019-05-01 12:00 Europe/Berlin",
                     "2019-05-01 12:01 Europe/Berlin")}}) {
    auto const results = search(
        tt, nullptr,
        routing::query{.start_time_ = start_time,
                       .start_ = {{loc_idx(tt, "M"), 0_minutes, 0U}},
                       .destination_ = {{loc_idx(tt, "Q"), 0_minutes, 0U}},
                       .via_stops_ = {{loc_idx(tt, "O"), 0_minutes},
                                      {loc_idx(tt, "Q"), 0_minutes}}},
        dir);

    EXPECT_EQ(expected_M_Q_via_M_0min_O_0min, results_to_str(results, tt));
  }
}

namespace {
constexpr auto const expected_A_L =
    R"(
[2019-05-01 08:15, 2019-05-01 10:00]
TRANSFERS: 1
     FROM: (A, A) [2019-05-01 08:15]
       TO: (L, L) [2019-05-01 10:00]
leg 0: (A, A) [2019-05-01 08:15] -> (C, C) [2019-05-01 08:35]
   0: A       A...............................................                               d: 01.05 08:15 [01.05 10:15]  [{name=2, day=2019-05-01, id=T4, src=0}]
   1: C       C............................................... a: 01.05 08:35 [01.05 10:35]
leg 1: (C, C) [2019-05-01 08:35] -> (G, G) [2019-05-01 08:40]
  FOOTPATH (duration=5)
leg 2: (G, G) [2019-05-01 08:45] -> (L, L) [2019-05-01 10:00]
   1: G       G...............................................                               d: 01.05 08:45 [01.05 10:45]  [{name=10, day=2019-05-01, id=T14, src=0}]
   2: R       R............................................... a: 01.05 09:02 [01.05 11:02]  d: 01.05 09:05 [01.05 11:05]  [{name=10, day=2019-05-01, id=T14, src=0}]
   3: L       L............................................... a: 01.05 10:00 [01.05 12:00]


)"sv;
}  // namespace

TEST(routing, via_test_25_A_L_no_via) {
  // test: shortest footpath between T3->T14 (5m D->R instead of 15m C->G)
  // A -> L
  auto tt = load_timetable(test_files_1);

  auto const results =
      search(tt, nullptr,
             routing::query{.start_time_ = iv("2019-05-01 10:00 Europe/Berlin",
                                              "2019-05-01 11:01 Europe/Berlin"),
                            .start_ = {{loc_idx(tt, "A"), 0_minutes, 0U}},
                            .destination_ = {{loc_idx(tt, "L"), 0_minutes, 0U}},
                            .via_stops_ = {}},
             direction::kForward);

  EXPECT_EQ(expected_A_L, results_to_str(results, tt));
}

TEST(routing, via_test_26_A_L_via_C_0m) {
  // test: shortest footpath between T3->T14 (5m D->R instead of 15m C->G)
  // A -> L, via C (0 min)
  auto tt = load_timetable(test_files_1);

  auto const results =
      search(tt, nullptr,
             routing::query{.start_time_ = iv("2019-05-01 10:00 Europe/Berlin",
                                              "2019-05-01 11:01 Europe/Berlin"),
                            .start_ = {{loc_idx(tt, "A"), 0_minutes, 0U}},
                            .destination_ = {{loc_idx(tt, "L"), 0_minutes, 0U}},
                            .via_stops_ = {{loc_idx(tt, "C"), 0_minutes}}},
             direction::kForward);

  EXPECT_EQ(expected_A_L, results_to_str(results, tt));
}

TEST(routing, via_test_27_A_L_via_R_0m) {
  // test: shortest footpath between T3->T14 (5m D->R instead of 15m C->G)
  // A -> L, via R (0 min)
  auto tt = load_timetable(test_files_1);

  auto const results =
      search(tt, nullptr,
             routing::query{.start_time_ = iv("2019-05-01 10:00 Europe/Berlin",
                                              "2019-05-01 11:01 Europe/Berlin"),
                            .start_ = {{loc_idx(tt, "A"), 0_minutes, 0U}},
                            .destination_ = {{loc_idx(tt, "L"), 0_minutes, 0U}},
                            .via_stops_ = {{loc_idx(tt, "R"), 0_minutes}}},
             direction::kForward);

  EXPECT_EQ(expected_A_L, results_to_str(results, tt));
}

TEST(routing, via_test_28_A_L_via_R_2m) {
  // test: keep the transfer with the longer footpath because of the
  // via with stay > 0 min
  // A -> L, via R (2 min)
  auto tt = load_timetable(test_files_1);

  constexpr auto const expected_A_L_via_R_2m =
      R"(
[2019-05-01 08:00, 2019-05-01 10:00]
TRANSFERS: 1
     FROM: (A, A) [2019-05-01 08:00]
       TO: (L, L) [2019-05-01 10:00]
leg 0: (A, A) [2019-05-01 08:00] -> (D, D) [2019-05-01 08:40]
   0: A       A...............................................                               d: 01.05 08:00 [01.05 10:00]  [{name=2, day=2019-05-01, id=T3, src=0}]
   1: C       C............................................... a: 01.05 08:20 [01.05 10:20]  d: 01.05 08:22 [01.05 10:22]  [{name=2, day=2019-05-01, id=T3, src=0}]
   2: D       D............................................... a: 01.05 08:40 [01.05 10:40]
leg 1: (D, D) [2019-05-01 08:40] -> (R, R) [2019-05-01 08:55]
  FOOTPATH (duration=15)
leg 2: (R, R) [2019-05-01 09:05] -> (L, L) [2019-05-01 10:00]
   2: R       R...............................................                               d: 01.05 09:05 [01.05 11:05]  [{name=10, day=2019-05-01, id=T14, src=0}]
   3: L       L............................................... a: 01.05 10:00 [01.05 12:00]


)"sv;

  auto const results =
      search(tt, nullptr,
             routing::query{.start_time_ = iv("2019-05-01 10:00 Europe/Berlin",
                                              "2019-05-01 10:01 Europe/Berlin"),
                            .start_ = {{loc_idx(tt, "A"), 0_minutes, 0U}},
                            .destination_ = {{loc_idx(tt, "L"), 0_minutes, 0U}},
                            .via_stops_ = {{loc_idx(tt, "R"), 2_minutes}}},
             direction::kForward);

  EXPECT_EQ(expected_A_L_via_R_2m, results_to_str(results, tt));
}

TEST(routing, via_test_29_A_L_via_D_2m) {
  // test: keep the transfer with the longer footpath because of the
  // via with stay > 0 min
  // A -> L, via R (2 min)
  auto tt = load_timetable(test_files_1);

  constexpr auto const expected_A_L_via_D_2m =
      R"(
[2019-05-01 08:00, 2019-05-01 10:00]
TRANSFERS: 1
     FROM: (A, A) [2019-05-01 08:00]
       TO: (L, L) [2019-05-01 10:00]
leg 0: (A, A) [2019-05-01 08:00] -> (D, D) [2019-05-01 08:40]
   0: A       A...............................................                               d: 01.05 08:00 [01.05 10:00]  [{name=2, day=2019-05-01, id=T3, src=0}]
   1: C       C............................................... a: 01.05 08:20 [01.05 10:20]  d: 01.05 08:22 [01.05 10:22]  [{name=2, day=2019-05-01, id=T3, src=0}]
   2: D       D............................................... a: 01.05 08:40 [01.05 10:40]
leg 1: (D, D) [2019-05-01 08:42] -> (R, R) [2019-05-01 08:57]
  FOOTPATH (duration=15)
leg 2: (R, R) [2019-05-01 09:05] -> (L, L) [2019-05-01 10:00]
   2: R       R...............................................                               d: 01.05 09:05 [01.05 11:05]  [{name=10, day=2019-05-01, id=T14, src=0}]
   3: L       L............................................... a: 01.05 10:00 [01.05 12:00]


)"sv;

  auto const results =
      search(tt, nullptr,
             routing::query{.start_time_ = iv("2019-05-01 10:00 Europe/Berlin",
                                              "2019-05-01 10:01 Europe/Berlin"),
                            .start_ = {{loc_idx(tt, "A"), 0_minutes, 0U}},
                            .destination_ = {{loc_idx(tt, "L"), 0_minutes, 0U}},
                            .via_stops_ = {{loc_idx(tt, "D"), 2_minutes}}},
             direction::kForward);

  EXPECT_EQ(expected_A_L_via_D_2m, results_to_str(results, tt));
}

TEST(routing, via_test_30_A_intermodal_LP_via_O_0m) {
  // test: keep the longer intermodal offset because of the via O
  // A -> intermodal L/P, via O (0 min)
  auto tt = load_timetable(test_files_1);

  constexpr auto const expected_A_intermodal_LP_via_O_0m =
      R"(
[2019-05-01 08:15, 2019-05-01 11:00]
TRANSFERS: 1
     FROM: (A, A) [2019-05-01 08:15]
       TO: (END, END) [2019-05-01 11:00]
leg 0: (A, A) [2019-05-01 08:15] -> (C, C) [2019-05-01 08:35]
   0: A       A...............................................                               d: 01.05 08:15 [01.05 10:15]  [{name=2, day=2019-05-01, id=T4, src=0}]
   1: C       C............................................... a: 01.05 08:35 [01.05 10:35]
leg 1: (C, C) [2019-05-01 08:35] -> (G, G) [2019-05-01 08:40]
  FOOTPATH (duration=5)
leg 2: (G, G) [2019-05-01 08:45] -> (P, P) [2019-05-01 10:40]
   1: G       G...............................................                               d: 01.05 08:45 [01.05 10:45]  [{name=10, day=2019-05-01, id=T14, src=0}]
   2: R       R............................................... a: 01.05 09:02 [01.05 11:02]  d: 01.05 09:05 [01.05 11:05]  [{name=10, day=2019-05-01, id=T14, src=0}]
   3: L       L............................................... a: 01.05 10:00 [01.05 12:00]  d: 01.05 10:02 [01.05 12:02]  [{name=10, day=2019-05-01, id=T14, src=0}]
   4: O       O............................................... a: 01.05 10:22 [01.05 12:22]  d: 01.05 10:25 [01.05 12:25]  [{name=10, day=2019-05-01, id=T14, src=0}]
   5: P       P............................................... a: 01.05 10:40 [01.05 12:40]
leg 3: (P, P) [2019-05-01 10:40] -> (END, END) [2019-05-01 11:00]
  MUMO (id=0, duration=20)


[2019-05-01 08:15, 2019-05-01 10:03]
TRANSFERS: 2
     FROM: (A, A) [2019-05-01 08:15]
       TO: (END, END) [2019-05-01 10:03]
leg 0: (A, A) [2019-05-01 08:15] -> (D, D) [2019-05-01 08:55]
   0: A       A...............................................                               d: 01.05 08:15 [01.05 10:15]  [{name=2, day=2019-05-01, id=T4, src=0}]
   1: C       C............................................... a: 01.05 08:35 [01.05 10:35]  d: 01.05 08:37 [01.05 10:37]  [{name=2, day=2019-05-01, id=T4, src=0}]
   2: D       D............................................... a: 01.05 08:55 [01.05 10:55]
leg 1: (D, D) [2019-05-01 08:55] -> (H, H) [2019-05-01 08:58]
  FOOTPATH (duration=3)
leg 2: (H, H) [2019-05-01 09:00] -> (O, O) [2019-05-01 09:20]
   0: H       H...............................................                               d: 01.05 09:00 [01.05 11:00]  [{name=9, day=2019-05-01, id=T12, src=0}]
   1: O       O............................................... a: 01.05 09:20 [01.05 11:20]
leg 3: (O, O) [2019-05-01 09:20] -> (O, O) [2019-05-01 09:22]
  FOOTPATH (duration=2)
leg 4: (O, O) [2019-05-01 09:30] -> (P, P) [2019-05-01 09:43]
   2: O       O...............................................                               d: 01.05 09:30 [01.05 11:30]  [{name=7, day=2019-05-01, id=T9, src=0}]
   3: P       P............................................... a: 01.05 09:43 [01.05 11:43]
leg 5: (P, P) [2019-05-01 09:43] -> (END, END) [2019-05-01 10:03]
  MUMO (id=0, duration=20)


)"sv;

  auto const results =
      search(tt, nullptr,
             routing::query{
                 .start_time_ = iv("2019-05-01 10:00 Europe/Berlin",
                                   "2019-05-01 11:01 Europe/Berlin"),
                 .dest_match_mode_ = routing::location_match_mode::kIntermodal,
                 .start_ = {{loc_idx(tt, "A"), 0_minutes, 0U}},
                 .destination_ = {{loc_idx(tt, "L"), 5_minutes, 0U},
                                  {loc_idx(tt, "P"), 20_minutes, 0U}},
                 .via_stops_ = {{loc_idx(tt, "O"), 0_minutes}}},
             direction::kForward);

  EXPECT_EQ(expected_A_intermodal_LP_via_O_0m, results_to_str(results, tt));
}

TEST(routing, via_test_31_M_Q_via_Q_10m) {
  // test: last via = destination, with stay duration (ignored)
  // M -> Q, via Q (10 min)
  auto tt = load_timetable(test_files_1);

  constexpr auto const expected_M_Q_via_O_10min =
      R"(
[2019-05-01 09:00, 2019-05-01 10:10]
TRANSFERS: 0
     FROM: (M, M) [2019-05-01 09:00]
       TO: (Q, Q) [2019-05-01 10:00]
leg 0: (M, M) [2019-05-01 09:00] -> (Q, Q) [2019-05-01 10:00]
   0: M       M...............................................                               d: 01.05 09:00 [01.05 11:00]  [{name=7, day=2019-05-01, id=T9, src=0}]
   1: N       N............................................... a: 01.05 09:13 [01.05 11:13]  d: 01.05 09:15 [01.05 11:15]  [{name=7, day=2019-05-01, id=T9, src=0}]
   2: O       O............................................... a: 01.05 09:28 [01.05 11:28]  d: 01.05 09:30 [01.05 11:30]  [{name=7, day=2019-05-01, id=T9, src=0}]
   3: P       P............................................... a: 01.05 09:43 [01.05 11:43]  d: 01.05 09:45 [01.05 11:45]  [{name=7, day=2019-05-01, id=T9, src=0}]
   4: Q       Q............................................... a: 01.05 10:00 [01.05 12:00]


)"sv;

  for (auto const& [dir, start_time] :
       {std::pair{direction::kForward,
                  iv("2019-05-01 11:00 Europe/Berlin",
                     "2019-05-01 11:01 Europe/Berlin")},
        /* std::pair{direction::kBackward,
                  iv("2019-05-01 12:00 Europe/Berlin",
                     "2019-05-01 12:01 Europe/Berlin")}*/}) {
    auto const results = search(
        tt, nullptr,
        routing::query{.start_time_ = start_time,
                       .start_ = {{loc_idx(tt, "M"), 0_minutes, 0U}},
                       .destination_ = {{loc_idx(tt, "Q"), 0_minutes, 0U}},
                       .via_stops_ = {{loc_idx(tt, "Q"), 10_minutes}}},
        dir);

    EXPECT_EQ(expected_M_Q_via_O_10min, results_to_str(results, tt));
  }
}

TEST(routing, via_test_32_M_intermodal_Q_via_Q_10m) {
  // test: last via = destination, with stay duration (before intermodal fp)
  // M -> Q, via Q (10 min)
  auto tt = load_timetable(test_files_1);

  constexpr auto const expected_M_intermodal_Q_via_O_10min =
      R"(
[2019-05-01 09:00, 2019-05-01 10:25]
TRANSFERS: 0
     FROM: (M, M) [2019-05-01 09:00]
       TO: (END, END) [2019-05-01 10:25]
leg 0: (M, M) [2019-05-01 09:00] -> (Q, Q) [2019-05-01 10:00]
   0: M       M...............................................                               d: 01.05 09:00 [01.05 11:00]  [{name=7, day=2019-05-01, id=T9, src=0}]
   1: N       N............................................... a: 01.05 09:13 [01.05 11:13]  d: 01.05 09:15 [01.05 11:15]  [{name=7, day=2019-05-01, id=T9, src=0}]
   2: O       O............................................... a: 01.05 09:28 [01.05 11:28]  d: 01.05 09:30 [01.05 11:30]  [{name=7, day=2019-05-01, id=T9, src=0}]
   3: P       P............................................... a: 01.05 09:43 [01.05 11:43]  d: 01.05 09:45 [01.05 11:45]  [{name=7, day=2019-05-01, id=T9, src=0}]
   4: Q       Q............................................... a: 01.05 10:00 [01.05 12:00]
leg 1: (Q, Q) [2019-05-01 10:10] -> (END, END) [2019-05-01 10:25]
  MUMO (id=0, duration=15)


)"sv;

  for (auto const& [dir, start_time] :
       {/*std::pair{direction::kForward, iv("2019-05-01 11:00 Europe/Berlin",
                                          "2019-05-01 11:01 Europe/Berlin")},*/
        std::pair{direction::kBackward,
                  iv("2019-05-01 11:59 Europe/Berlin",
                     "2019-05-01 13:01 Europe/Berlin")}}) {
    auto const results = search(
        tt, nullptr,
        routing::query{
            .start_time_ = start_time,
            .dest_match_mode_ = routing::location_match_mode::kIntermodal,
            .start_ = {{loc_idx(tt, "M"), 0_minutes, 0U}},
            .destination_ = {{loc_idx(tt, "Q"), 15_minutes, 0U}},
            .via_stops_ = {{loc_idx(tt, "Q"), 10_minutes}}},
        dir);

    auto results_str = results_to_str(results, tt);
    if (dir == direction::kBackward) {
      results_str = std::regex_replace(results_str, std::regex("START"), "END");
    }
    EXPECT_EQ(expected_M_intermodal_Q_via_O_10min, results_str);
  }
}

TEST(routing, via_test_33_M_intermodal_Q_via_Q_0m) {
  // test: last via = destination
  // M -> Q, via Q (0 min)
  auto tt = load_timetable(test_files_1);

  constexpr auto const expected_M_intermodal_Q_via_O_0min =
      R"(
[2019-05-01 09:00, 2019-05-01 10:15]
TRANSFERS: 0
     FROM: (M, M) [2019-05-01 09:00]
       TO: (END, END) [2019-05-01 10:15]
leg 0: (M, M) [2019-05-01 09:00] -> (Q, Q) [2019-05-01 10:00]
   0: M       M...............................................                               d: 01.05 09:00 [01.05 11:00]  [{name=7, day=2019-05-01, id=T9, src=0}]
   1: N       N............................................... a: 01.05 09:13 [01.05 11:13]  d: 01.05 09:15 [01.05 11:15]  [{name=7, day=2019-05-01, id=T9, src=0}]
   2: O       O............................................... a: 01.05 09:28 [01.05 11:28]  d: 01.05 09:30 [01.05 11:30]  [{name=7, day=2019-05-01, id=T9, src=0}]
   3: P       P............................................... a: 01.05 09:43 [01.05 11:43]  d: 01.05 09:45 [01.05 11:45]  [{name=7, day=2019-05-01, id=T9, src=0}]
   4: Q       Q............................................... a: 01.05 10:00 [01.05 12:00]
leg 1: (Q, Q) [2019-05-01 10:00] -> (END, END) [2019-05-01 10:15]
  MUMO (id=0, duration=15)


)"sv;

  for (auto const& [dir, start_time] :
       {std::pair{direction::kForward, iv("2019-05-01 11:00 Europe/Berlin",
                                          "2019-05-01 11:01 Europe/Berlin")},
        std::pair{direction::kBackward,
                  iv("2019-05-01 12:15 Europe/Berlin",
                     "2019-05-01 12:16 Europe/Berlin")}}) {
    auto const results = search(
        tt, nullptr,
        routing::query{
            .start_time_ = start_time,
            .dest_match_mode_ = routing::location_match_mode::kIntermodal,
            .start_ = {{loc_idx(tt, "M"), 0_minutes, 0U}},
            .destination_ = {{loc_idx(tt, "Q"), 15_minutes, 0U}},
            .via_stops_ = {{loc_idx(tt, "Q"), 0_minutes}}},
        dir);

    auto results_str = results_to_str(results, tt);
    if (dir == direction::kBackward) {
      results_str = std::regex_replace(results_str, std::regex("START"), "END");
    }
    EXPECT_EQ(expected_M_intermodal_Q_via_O_0min, results_str);
  }
}
