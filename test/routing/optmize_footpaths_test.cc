#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/common/interval.h"
#include "nigiri/common/parse_time.h"
#include "nigiri/footpath.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/reconstruct.h"
#include "nigiri/special_stations.h"
#include "nigiri/types.h"
#include <string_view>

using namespace nigiri;
using namespace date;
using namespace std::chrono_literals;
using namespace std::string_view_literals;

namespace {

std::string print_journey(timetable const& tt,
                          routing::journey const& journey) {
  std::stringstream ss;
  ss << "\n";
  journey.print(ss, tt);
  return ss.str();
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

location_idx_t loc_idx(timetable const& tt, std::string_view const id) {
  return tt.find(location_id{id, source_idx_t{0}}).value();
}

unixtime_t time(std::string_view const time) {
  return parse_time_tz(time, "%Y-%m-%d %H:%M %Z");
}

constexpr auto const test_files_1 = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

#stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,
C,C,,4.0,5.0,,
D,D,,6.0,7.0,,
E,E,,8.0,9.0,,

#routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DB,0,,,3
R1,DB,1,,,3
R2,DB,2,,,3

#trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,S1,T0,,
R1,S1,T1,,

#stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T0,10:10:00,10:10:00,A,0
T0,10:12:00,10:12:00,B,1
T0,10:15:00,10:15:00,C,2
T1,10:17:00,10:17:00,C,0
T1,10:33:00,10:33:00,D,1
T1,10:40:00,10:40:00,E,2

#calendar_dates.txt
service_id,date,exception_type
S1,20190501,1

#transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
)"sv;

}  // namespace

TEST(routing, optimize_initial_departure_test) {
  constexpr auto const expected_optimize_initial_departure = R"(
[2019-05-01 08:00, 2019-05-01 08:15]
TRANSFERS: 0
     FROM: (START, START) [2019-05-01 08:00]
       TO: (C, C) [2019-05-01 08:15]
leg 0: (START, START) [2019-05-01 08:00] -> (A, A) [2019-05-01 08:10]
  MUMO (id=0, duration=10)
leg 1: (A, A) [2019-05-01 08:10] -> (C, C) [2019-05-01 08:15]
   0: A       A...............................................                               d: 01.05 08:10 [01.05 10:10]  [{name=0, day=2019-05-01, id=T0, src=0}]
   1: B       B............................................... a: 01.05 08:12 [01.05 10:12]  d: 01.05 08:12 [01.05 10:12]  [{name=0, day=2019-05-01, id=T0, src=0}]
   2: C       C............................................... a: 01.05 08:15 [01.05 10:15]
)"sv;

  auto tt = load_timetable(test_files_1);

  auto const A = loc_idx(tt, "A");
  auto const B = loc_idx(tt, "B");
  auto const C = loc_idx(tt, "C");

  auto const q = routing::query{.start_ = {{B, 8_minutes, 0U}},
                                .via_stops_ = {{loc_idx(tt, "A"), 0_minutes}}};

  auto journey = routing::journey{};
  journey.transfers_ = 0U;
  journey.start_time_ = time("2019-05-01 10:00 Europe/Berlin");
  journey.dest_time_ = time("2019-05-01 10:15 Europe/Berlin");

  // Start -> A
  journey.add(routing::journey::leg{
      direction::kForward, get_special_station(special_station::kStart), A,
      time("2019-05-01 10:00 Europe/Berlin"),
      time("2019-05-01 10:10 Europe/Berlin"),
      routing::offset{A, 10_minutes, 0U}});

  // A -> B -> C
  journey.add(routing::journey::leg{
      direction::kForward, A, C, time("2019-05-01 10:10 Europe/Berlin"),
      time("2019-05-01 10:15 Europe/Berlin"),
      routing::journey::run_enter_exit{
          {
              .t_ = {transport_idx_t{0U}, day_idx_t{5U}},
              .stop_range_ = interval<stop_idx_t>{0U, 3U},
          },
          0,
          2}});

  optimize_footpaths(tt, nullptr, q, journey);
  EXPECT_EQ(expected_optimize_initial_departure, print_journey(tt, journey));
}

TEST(routing, optimize_last_arrival_test) {
  constexpr auto const expected_optimize_last_arrival = R"(
[2019-05-01 08:10, 2019-05-01 08:25]
TRANSFERS: 0
     FROM: (A, A) [2019-05-01 08:10]
       TO: (END, END) [2019-05-01 08:25]
leg 0: (A, A) [2019-05-01 08:10] -> (C, C) [2019-05-01 08:15]
   0: A       A...............................................                               d: 01.05 08:10 [01.05 10:10]  [{name=0, day=2019-05-01, id=T0, src=0}]
   1: B       B............................................... a: 01.05 08:12 [01.05 10:12]  d: 01.05 08:12 [01.05 10:12]  [{name=0, day=2019-05-01, id=T0, src=0}]
   2: C       C............................................... a: 01.05 08:15 [01.05 10:15]
leg 1: (C, C) [2019-05-01 08:15] -> (END, END) [2019-05-01 08:25]
  MUMO (id=0, duration=10)
)"sv;

  auto tt = load_timetable(test_files_1);

  auto const A = loc_idx(tt, "A");
  auto const B = loc_idx(tt, "B");
  auto const C = loc_idx(tt, "C");

  auto const q = routing::query{.destination_ = {{B, 8_minutes, 0U}},
                                .via_stops_ = {{loc_idx(tt, "C"), 0_minutes}}};

  auto journey = routing::journey{};
  journey.transfers_ = 0U;
  journey.start_time_ = time("2019-05-01 10:10 Europe/Berlin");
  journey.dest_time_ = time("2019-05-01 10:25 Europe/Berlin");

  // A -> B -> C
  journey.add(routing::journey::leg{
      direction::kForward, A, C, time("2019-05-01 10:10 Europe/Berlin"),
      time("2019-05-01 10:15 Europe/Berlin"),
      routing::journey::run_enter_exit{
          {
              .t_ = {transport_idx_t{0U}, day_idx_t{5U}},
              .stop_range_ = interval<stop_idx_t>{0U, 3U},
          },
          0,
          2}});

  // C -> END
  journey.add(routing::journey::leg{
      direction::kForward, C, get_special_station(special_station::kEnd),
      time("2019-05-01 10:15 Europe/Berlin"),
      time("2019-05-01 10:25 Europe/Berlin"),
      routing::offset{get_special_station(special_station::kEnd), 10_minutes,
                      0U}});

  optimize_footpaths(tt, nullptr, q, journey);
  EXPECT_EQ(expected_optimize_last_arrival, print_journey(tt, journey));
}

TEST(routing, optimize_transfers_test) {
  constexpr auto const expected_optimize_transfers = R"(
[2019-05-01 08:10, 2019-05-01 08:40]
TRANSFERS: 1
     FROM: (A, A) [2019-05-01 08:10]
       TO: (E, E) [2019-05-01 08:40]
leg 0: (A, A) [2019-05-01 08:10] -> (C, C) [2019-05-01 08:15]
   0: A       A...............................................                               d: 01.05 08:10 [01.05 10:10]  [{name=0, day=2019-05-01, id=T0, src=0}]
   1: B       B............................................... a: 01.05 08:12 [01.05 10:12]  d: 01.05 08:12 [01.05 10:12]  [{name=0, day=2019-05-01, id=T0, src=0}]
   2: C       C............................................... a: 01.05 08:15 [01.05 10:15]
leg 1: (C, C) [2019-05-01 08:15] -> (C, C) [2019-05-01 08:17]
  FOOTPATH (duration=2)
leg 2: (C, C) [2019-05-01 08:17] -> (E, E) [2019-05-01 08:40]
   0: C       C...............................................                               d: 01.05 08:17 [01.05 10:17]  [{name=1, day=2019-05-01, id=T1, src=0}]
   1: D       D............................................... a: 01.05 08:33 [01.05 10:33]  d: 01.05 08:33 [01.05 10:33]  [{name=1, day=2019-05-01, id=T1, src=0}]
   2: E       E............................................... a: 01.05 08:40 [01.05 10:40]
)"sv;

  auto tt = load_timetable(test_files_1);

  auto const A = loc_idx(tt, "A");
  auto const B = loc_idx(tt, "B");
  auto const C = loc_idx(tt, "C");
  auto const D = loc_idx(tt, "D");
  auto const E = loc_idx(tt, "E");

  tt.locations_.footpaths_out_[0U][B].push_back(footpath{D, 1_minutes});

  auto const q = routing::query{.prf_idx_ = 0U,
                                .via_stops_ = {{loc_idx(tt, "C"), 0_minutes}}};

  auto journey = routing::journey{};
  journey.transfers_ = 1U;
  journey.start_time_ = time("2019-05-01 10:10 Europe/Berlin");
  journey.dest_time_ = time("2019-05-01 10:40 Europe/Berlin");

  // A -> B -> C
  journey.add(routing::journey::leg{
      direction::kForward, A, C, time("2019-05-01 10:10 Europe/Berlin"),
      time("2019-05-01 10:15 Europe/Berlin"),
      routing::journey::run_enter_exit{
          {
              .t_ = {transport_idx_t{0U}, day_idx_t{5U}},
              .stop_range_ = interval<stop_idx_t>{0U, 3U},
          },
          0,
          2}});

  // Transfer C
  journey.add(routing::journey::leg{
      direction::kForward, C, C, time("2019-05-01 10:15 Europe/Berlin"),
      time("2019-05-01 10:17 Europe/Berlin"), footpath{C, 2_minutes}});

  // C -> D -> E
  journey.add(routing::journey::leg{
      direction::kForward, C, E, time("2019-05-01 10:17 Europe/Berlin"),
      time("2019-05-01 10:40 Europe/Berlin"),
      routing::journey::run_enter_exit{
          {
              .t_ = {transport_idx_t{1U}, day_idx_t{5U}},
              .stop_range_ = interval<stop_idx_t>{0U, 3U},
          },
          0,
          2}});

  optimize_footpaths(tt, nullptr, q, journey);
  EXPECT_EQ(expected_optimize_transfers, print_journey(tt, journey));
}
