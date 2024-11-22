
#include "gtest/gtest.h"

#include "utl/erase_if.h"

#include "nigiri/loader/gtfs/agency.h"
#include "nigiri/loader/gtfs/calendar.h"
#include "nigiri/loader/gtfs/calendar_date.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/gtfs/local_to_utc.h"
#include "nigiri/loader/gtfs/noon_offsets.h"
#include "nigiri/loader/init_finish.h"

#include "nigiri/routing/query.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

#include "../raptor_search.h"

using namespace nigiri;
using namespace date;
using namespace std::chrono_literals;
using namespace std::string_view_literals;
using nigiri::test::raptor_search;

namespace {

// Trips:
// T1: A 10:00 -> B 10:10
// T2: B 10:13 -> C 10:22 (departure 3 min after T1 arrival)
// T3: B 10:20 -> C 10:30 (departure 10 min after T1 arrival)
// Transfer times for all stations: 2 min (default)
constexpr auto const test_files = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,
C,C,,4.0,5.0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,1,,,3
R2,DB,2,,,3
R3,DB,3,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S1,T1,,
R2,S1,T2,,
R2,S1,T3,,
R3,S1,T4,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T1,10:00:00,10:00:00,A,0
T1,10:10:00,10:10:00,B,1
T2,10:13:00,10:13:00,B,0
T2,10:22:00,10:22:00,C,1
T3,10:20:00,10:20:00,B,0
T3,10:30:00,10:30:00,C,1
T4,09:55:00,09:55:00,A,0
T4,10:35:00,10:35:00,C,1

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)"sv;

constexpr auto const expected_A_C_default =
    R"(
[2019-05-01 08:00, 2019-05-01 08:22]
TRANSFERS: 1
     FROM: (A, A) [2019-05-01 08:00]
       TO: (C, C) [2019-05-01 08:22]
leg 0: (A, A) [2019-05-01 08:00] -> (B, B) [2019-05-01 08:10]
   0: A       A...............................................                               d: 01.05 08:00 [01.05 10:00]  [{name=Bus 1, day=2019-05-01, id=T1, src=0}]
   1: B       B............................................... a: 01.05 08:10 [01.05 10:10]
leg 1: (B, B) [2019-05-01 08:10] -> (B, B) [2019-05-01 08:12]
  FOOTPATH (duration=2)
leg 2: (B, B) [2019-05-01 08:13] -> (C, C) [2019-05-01 08:22]
   0: B       B...............................................                               d: 01.05 08:13 [01.05 10:13]  [{name=Bus 2, day=2019-05-01, id=T2, src=0}]
   1: C       C............................................... a: 01.05 08:22 [01.05 10:22]


)"sv;

constexpr auto const expected_A_C_dur10 =
    R"(
[2019-05-01 08:00, 2019-05-01 08:30]
TRANSFERS: 1
     FROM: (A, A) [2019-05-01 08:00]
       TO: (C, C) [2019-05-01 08:30]
leg 0: (A, A) [2019-05-01 08:00] -> (B, B) [2019-05-01 08:10]
   0: A       A...............................................                               d: 01.05 08:00 [01.05 10:00]  [{name=Bus 1, day=2019-05-01, id=T1, src=0}]
   1: B       B............................................... a: 01.05 08:10 [01.05 10:10]
leg 1: (B, B) [2019-05-01 08:10] -> (B, B) [2019-05-01 08:20]
  FOOTPATH (duration=10)
leg 2: (B, B) [2019-05-01 08:20] -> (C, C) [2019-05-01 08:30]
   0: B       B...............................................                               d: 01.05 08:20 [01.05 10:20]  [{name=Bus 2, day=2019-05-01, id=T3, src=0}]
   1: C       C............................................... a: 01.05 08:30 [01.05 10:30]


)"sv;

constexpr auto const expected_A_C_dur3 =
    R"(
[2019-05-01 08:00, 2019-05-01 08:22]
TRANSFERS: 1
     FROM: (A, A) [2019-05-01 08:00]
       TO: (C, C) [2019-05-01 08:22]
leg 0: (A, A) [2019-05-01 08:00] -> (B, B) [2019-05-01 08:10]
   0: A       A...............................................                               d: 01.05 08:00 [01.05 10:00]  [{name=Bus 1, day=2019-05-01, id=T1, src=0}]
   1: B       B............................................... a: 01.05 08:10 [01.05 10:10]
leg 1: (B, B) [2019-05-01 08:10] -> (B, B) [2019-05-01 08:13]
  FOOTPATH (duration=3)
leg 2: (B, B) [2019-05-01 08:13] -> (C, C) [2019-05-01 08:22]
   0: B       B...............................................                               d: 01.05 08:13 [01.05 10:13]  [{name=Bus 2, day=2019-05-01, id=T2, src=0}]
   1: C       C............................................... a: 01.05 08:22 [01.05 10:22]


)"sv;

constexpr auto const expected_A_C_dur4 =
    R"(
[2019-05-01 08:00, 2019-05-01 08:30]
TRANSFERS: 1
     FROM: (A, A) [2019-05-01 08:00]
       TO: (C, C) [2019-05-01 08:30]
leg 0: (A, A) [2019-05-01 08:00] -> (B, B) [2019-05-01 08:10]
   0: A       A...............................................                               d: 01.05 08:00 [01.05 10:00]  [{name=Bus 1, day=2019-05-01, id=T1, src=0}]
   1: B       B............................................... a: 01.05 08:10 [01.05 10:10]
leg 1: (B, B) [2019-05-01 08:10] -> (B, B) [2019-05-01 08:14]
  FOOTPATH (duration=4)
leg 2: (B, B) [2019-05-01 08:20] -> (C, C) [2019-05-01 08:30]
   0: B       B...............................................                               d: 01.05 08:20 [01.05 10:20]  [{name=Bus 2, day=2019-05-01, id=T3, src=0}]
   1: C       C............................................... a: 01.05 08:30 [01.05 10:30]


)"sv;

constexpr auto const expected_A_C_direct =
    R"(
[2019-05-01 07:55, 2019-05-01 08:35]
TRANSFERS: 0
     FROM: (A, A) [2019-05-01 07:55]
       TO: (C, C) [2019-05-01 08:35]
leg 0: (A, A) [2019-05-01 07:55] -> (C, C) [2019-05-01 08:35]
   0: A       A...............................................                               d: 01.05 07:55 [01.05 09:55]  [{name=Bus 3, day=2019-05-01, id=T4, src=0}]
   1: C       C............................................... a: 01.05 08:35 [01.05 10:35]


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

std::string add_direct(std::string_view const journey, direction const dir) {
  auto const [first, second] = dir == direction::kForward
                                   ? std::tie(expected_A_C_direct, journey)
                                   : std::tie(journey, expected_A_C_direct);
  // Concat and remove one '\n'
  return std::string{first} + std::string{second.substr(1U)};
}

pareto_set<routing::journey> search(timetable const& tt,
                                    std::string_view const from,
                                    std::string_view const to,
                                    direction const dir,
                                    routing::query&& q) {
  return raptor_search(tt, nullptr, std::move(q),
                       dir == direction::kForward ? from : to,
                       dir == direction::kForward ? to : from, {}, dir);
}

}  // namespace

TEST(routing, transfer_travel_test) {
  auto tt = timetable{};

  tt.date_range_ = {date::sys_days{2019_y / May / 1},
                    date::sys_days{2019_y / May / 2}};
  loader::register_special_stations(tt);
  loader::gtfs::load_timetable({}, source_idx_t{0},
                               loader::mem_dir::read(test_files), tt);
  loader::finalize(tt);

  // Tests for transfer_time_settings
  for (auto const dir : {direction::kForward, direction::kBackward}) {

    {  // A -> C, default transfer time (= 2 min)
      auto const results = search(tt, "A", "C", dir,
                                  {.start_time_ = tt.date_range_,
                                   .max_travel_time_ = 30_minutes,
                                   .transfer_time_settings_ = {}});
      EXPECT_EQ(expected_A_C_default, results_to_str(results, tt));
    }

    {  // A -> C, min 10 min transfer time (= 10 min)
      auto const results = search(
          tt, "A", "C", dir,
          {.start_time_ = tt.date_range_,
           .max_travel_time_ = 30_minutes,
           .transfer_time_settings_ = {.default_ = false,
                                       .min_transfer_time_ = duration_t{10}}});
      EXPECT_EQ(expected_A_C_dur10, results_to_str(results, tt));
    }

    {  // A -> C, 1.5x transfer time (= 3 min)
      auto const results = search(
          tt, "A", "C", dir,
          {.start_time_ = tt.date_range_,
           .max_travel_time_ = 30_minutes,
           .transfer_time_settings_ = {.default_ = false, .factor_ = 1.5F}});
      EXPECT_EQ(expected_A_C_dur3, results_to_str(results, tt));
    }

    {  // A -> C, 2.0x transfer time (= 4 min)
      auto const results = search(
          tt, "A", "C", dir,
          {.start_time_ = tt.date_range_,
           .max_travel_time_ = 30_minutes,
           .transfer_time_settings_ = {.default_ = false, .factor_ = 2.0F}});
      EXPECT_EQ(expected_A_C_dur4, results_to_str(results, tt));
    }

    {  // A -> C, min 10 min transfer time, 2.0x transfer time (= 10 min)
      auto const results = search(
          tt, "A", "C", dir,
          {.start_time_ = tt.date_range_,
           .max_travel_time_ = 30_minutes,
           .transfer_time_settings_ = {.default_ = false,
                                       .min_transfer_time_ = duration_t{10},
                                       .factor_ = 2.0F}});
      EXPECT_EQ(expected_A_C_dur10, results_to_str(results, tt));
    }

    {  // A -> C, min 3 min transfer time, 2.0x transfer time (= 4 min)
      auto const results = search(
          tt, "A", "C", dir,
          {.start_time_ = tt.date_range_,
           .max_travel_time_ = 30_minutes,
           .transfer_time_settings_ = {.default_ = false,
                                       .min_transfer_time_ = duration_t{3},
                                       .factor_ = 2.0F}});
      EXPECT_EQ(expected_A_C_dur4, results_to_str(results, tt));
    }
    {
      // A -> C, default transfer time, 2 min additional (= 4 min)
      auto const results =
          search(tt, "A", "C", dir,
                 {.start_time_ = tt.date_range_,
                  .max_travel_time_ = 30_minutes,
                  .transfer_time_settings_ = {
                      .default_ = false, .additional_time_ = duration_t{2}}});
      EXPECT_EQ(expected_A_C_dur4, results_to_str(results, tt));
    }
    {
      // A -> C, 1.5x transfer time, 1 min additional (= 4 min)
      auto const results =
          search(tt, "A", "C", dir,
                 {.start_time_ = tt.date_range_,
                  .max_travel_time_ = 30_minutes,
                  .transfer_time_settings_ = {.default_ = false,
                                              .additional_time_ = duration_t{1},
                                              .factor_ = 1.5F}});
      EXPECT_EQ(expected_A_C_dur4, results_to_str(results, tt));
    }
    {
      // A -> C, min 3 min transfer time, 1 min additional (= 4 min)
      auto const results = search(
          tt, "A", "C", dir,
          {.start_time_ = tt.date_range_,
           .max_travel_time_ = 30_minutes,
           .transfer_time_settings_ = {.default_ = false,
                                       .min_transfer_time_ = duration_t{3},
                                       .additional_time_ = duration_t{1}}});
      EXPECT_EQ(expected_A_C_dur4, results_to_str(results, tt));
    }
    {
      // A -> C, min 3 min transfer time, 2.5x transfer time, 5 min additional
      // (= 10 min)
      auto const results = search(
          tt, "A", "C", dir,
          {.start_time_ = tt.date_range_,
           .max_travel_time_ = 30_minutes,
           .transfer_time_settings_ = {.default_ = false,
                                       .min_transfer_time_ = duration_t{3},
                                       .additional_time_ = duration_t{5},
                                       .factor_ = 2.5F}});
      EXPECT_EQ(expected_A_C_dur10, results_to_str(results, tt));
    }
    // Test with search restrictions
    {
      // A -> C, default max_travel_time
      auto const results = search(
          tt, "A", "C", dir,
          {.start_time_ = tt.date_range_, .transfer_time_settings_ = {}});
      EXPECT_EQ(add_direct(expected_A_C_default, dir),
                results_to_str(results, tt));
    }
  }

  // Tests with search restrictions
  {
    // Edge case: A -> C, 2.0x transfer time (= 4 min) (see above)
    // max_travel_time = 29 min < result travel time (= 30 min)
    constexpr auto const dir = direction::kForward;
    auto const results = search(
        tt, "A", "C", dir,
        {.start_time_ = tt.date_range_,
         .max_travel_time_ = 29_minutes,
         .transfer_time_settings_ = {.default_ = false, .factor_ = 2.0F}});
    EXPECT_EQ("\n", results_to_str(results, tt));
  }
  {
    // A -> C, invalid max_travel_time
    constexpr auto const dir = direction::kForward;
    auto const results =
        search(tt, "A", "C", dir,
               {.start_time_ = tt.date_range_, .max_travel_time_ = -1_days});
    EXPECT_EQ(add_direct(expected_A_C_default, dir),
              results_to_str(results, tt));
  }
  {
    // A -> C, max_transfers = 1
    auto const results =
        search(tt, "A", "C", direction::kForward,
               {.start_time_ = tt.date_range_, .max_transfers_ = 1U});
    EXPECT_EQ(expected_A_C_direct, results_to_str(results, tt));
  }
  {
    // A -> C, max_transfers = 0
    auto const results =
        search(tt, "A", "C", direction::kForward,
               {.start_time_ = tt.date_range_, .max_transfers_ = 0U});
    EXPECT_EQ("\n", results_to_str(results, tt));
  }
  {
    // A -> C, max_transfers = 1, max_travel_time = 35
    auto const results = search(tt, "A", "C", direction::kForward,
                                {.start_time_ = tt.date_range_,
                                 .max_transfers_ = 1U,
                                 .max_travel_time_ = 35_minutes});
    EXPECT_EQ("\n", results_to_str(results, tt));
  }
}
