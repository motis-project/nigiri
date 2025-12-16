#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/loader/load.h"

#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

#include "../raptor_search.h"
#include "../rt/util.h"
#include "results_to_string.h"

using namespace nigiri;
using namespace date;
using namespace std::chrono_literals;
using namespace std::string_view_literals;
using nigiri::test::raptor_search;

namespace {

// ROUTING CONNECTIONS:
// 10:00 - 11:00 A-C    airplane direct
// 10:00 - 12:00 A-B-C  train, one transfer
constexpr auto const test_files = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin
LH,Lufthansa,https://lufthansa.de,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,
C,C,,4.0,5.0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
AIR,LH,X,,,1100
R1,DB,1,,,3
R2,DB,2,,,2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
AIR,S1,AIR,,
R1,S1,T1,,
R2,S1,T2,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
AIR,10:00:00,10:00:00,A,0,0,0
AIR,11:00:00,11:00:00,C,1,0,0
T1,10:00:00,10:00:00,A,0,0,0
T1,10:55:00,10:55:00,B,1,0,0
T2,11:05:00,11:05:00,B,0,0,0
T2,12:00:00,12:00:00,C,1,0,0

# calendar_dates.txt
service_id,date,exception_type
S1,20240301,1
)"sv;

constexpr auto const expected =
    R"(
[2024-03-01 09:00, 2024-03-01 10:00]
TRANSFERS: 0
     FROM: (A, A) [2024-03-01 09:00]
       TO: (C, C) [2024-03-01 10:00]
leg 0: (A, A) [2024-03-01 09:00] -> (C, C) [2024-03-01 10:00]
   0: A       A...............................................                               d: 01.03 09:00 [01.03 10:00]  [{name=X, day=2024-03-01, id=AIR, src=0}]
   1: C       C............................................... a: 01.03 10:00 [01.03 11:00]

)"sv;

constexpr auto const expected_1 =
    R"(
[2024-03-01 09:00, 2024-03-01 11:00]
TRANSFERS: 1
     FROM: (A, A) [2024-03-01 09:00]
       TO: (C, C) [2024-03-01 11:00]
leg 0: (A, A) [2024-03-01 09:00] -> (B, B) [2024-03-01 09:55]
   0: A       A...............................................                               d: 01.03 09:00 [01.03 10:00]  [{name=1, day=2024-03-01, id=T1, src=0}]
   1: B       B............................................... a: 01.03 09:55 [01.03 10:55]
leg 1: (B, B) [2024-03-01 09:55] -> (B, B) [2024-03-01 09:57]
  FOOTPATH (duration=2)
leg 2: (B, B) [2024-03-01 10:05] -> (C, C) [2024-03-01 11:00]
   0: B       B...............................................                               d: 01.03 10:05 [01.03 11:05]  [{name=2, day=2024-03-01, id=T2, src=0}]
   1: C       C............................................... a: 01.03 11:00 [01.03 12:00]

)"sv;

constexpr auto const expected_rt =
    R"(
[2024-03-01 09:10, 2024-03-01 10:10]
TRANSFERS: 0
     FROM: (A, A) [2024-03-01 09:10]
       TO: (C, C) [2024-03-01 10:10]
leg 0: (A, A) [2024-03-01 09:10] -> (C, C) [2024-03-01 10:10]
   0: A       A...............................................                               d: 01.03 09:00 [01.03 10:00]  [{name=X, day=2024-03-01, id=AIR, src=0}]
   1: C       C............................................... a: 01.03 10:00 [01.03 11:00]

)"sv;

constexpr auto const expected_rt_1 =
    R"(
[2024-03-01 09:10, 2024-03-01 11:10]
TRANSFERS: 1
     FROM: (A, A) [2024-03-01 09:10]
       TO: (C, C) [2024-03-01 11:10]
leg 0: (A, A) [2024-03-01 09:10] -> (B, B) [2024-03-01 10:05]
   0: A       A...............................................                               d: 01.03 09:00 [01.03 10:00]  [{name=1, day=2024-03-01, id=T1, src=0}]
   1: B       B............................................... a: 01.03 09:55 [01.03 10:55]
leg 1: (B, B) [2024-03-01 10:05] -> (B, B) [2024-03-01 10:07]
  FOOTPATH (duration=2)
leg 2: (B, B) [2024-03-01 10:15] -> (C, C) [2024-03-01 11:10]
   0: B       B...............................................                               d: 01.03 10:05 [01.03 11:05]  [{name=2, day=2024-03-01, id=T2, src=0}]
   1: C       C............................................... a: 01.03 11:00 [01.03 12:00]

)"sv;

}  // namespace

template <typename... T>
routing::clasz_mask_t make_mask(T... c) {
  auto allowed = routing::clasz_mask_t{0U};
  ((allowed |= (1U << static_cast<std::underlying_type_t<clasz>>(c))), ...);
  return allowed;
}

TEST(routing, clasz_filter_test) {
  auto tt = timetable{};

  tt.date_range_ = {date::sys_days{2024_y / March / 1},
                    date::sys_days{2024_y / March / 2}};
  loader::register_special_stations(tt);
  loader::gtfs::load_timetable({}, source_idx_t{0},
                               loader::mem_dir::read(test_files), tt);
  loader::finalize(tt);

  {  // All classes.
    auto const results =
        raptor_search(tt, nullptr, "A", "C", tt.date_range_,
                      direction::kForward, routing::all_clasz_allowed());

    EXPECT_EQ(expected, to_string(tt, results));
  }

  {  // All available classes.
    auto const results = raptor_search(
        tt, nullptr, "A", "C", tt.date_range_, direction::kForward,
        make_mask(clasz::kBus, clasz::kRegionalFast, clasz::kAir));

    EXPECT_EQ(expected, to_string(tt, results));
  }

  {  // No plane - one transfer, 2h
    auto const results = raptor_search(
        tt, nullptr, "A", "C", tt.date_range_, direction::kForward,
        make_mask(clasz::kBus, clasz::kRegionalFast));

    EXPECT_EQ(expected_1, to_string(tt, results));
  }

  {  // No connection.
    auto const results =
        raptor_search(tt, nullptr, "A", "C", tt.date_range_,
                      direction::kForward, make_mask(clasz::kShip));

    EXPECT_TRUE(results.size() == 0U);
  }

  // Update.
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / March / 1});
  auto const msg = test::to_feed_msg(
      {test::trip{.trip_id_ = "AIR",
                  .delays_ = {{.stop_id_ = "A",
                               .ev_type_ = nigiri::event_type::kArr,
                               .delay_minutes_ = 10}}},
       test::trip{.trip_id_ = "T1",
                  .delays_ = {{.stop_id_ = "A",
                               .ev_type_ = event_type::kDep,
                               .delay_minutes_ = 10U}}},
       test::trip{.trip_id_ = "T2",
                  .delays_ = {{.stop_id_ = "B",
                               .ev_type_ = event_type::kDep,
                               .delay_minutes_ = 10U}}}},
      date::sys_days{2024_y / March / 1} + 1h);
  rt::gtfsrt_update_msg(tt, rtt, source_idx_t{0}, "", msg);

  {  // All available classes.
    auto const results = raptor_search(
        tt, &rtt, "A", "C", tt.date_range_, direction::kForward,
        make_mask(clasz::kBus, clasz::kRegionalFast, clasz::kAir));

    EXPECT_EQ(expected_rt, to_string(tt, results));
  }

  {  // No plane - one transfer, 2h
    auto const results =
        raptor_search(tt, &rtt, "A", "C", tt.date_range_, direction::kForward,
                      make_mask(clasz::kBus, clasz::kRegionalFast));

    EXPECT_EQ(expected_rt_1, to_string(tt, results));
  }
}
