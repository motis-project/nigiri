
#include "gtest/gtest.h"

#include "utl/erase_if.h"

#include "nigiri/loader/gtfs/agency.h"
#include "nigiri/loader/gtfs/calendar.h"
#include "nigiri/loader/gtfs/calendar_date.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/gtfs/local_to_utc.h"
#include "nigiri/loader/gtfs/noon_offsets.h"
#include "nigiri/loader/init_finish.h"

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

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S1,T1,,
R2,S1,T2,,
R2,S1,T3,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T1,10:00:00,10:00:00,A,0
T1,10:10:00,10:10:00,B,1
T2,10:13:00,10:13:00,B,0
T2,10:22:00,10:22:00,C,1
T3,10:20:00,10:20:00,B,0
T3,10:30:00,10:30:00,C,1

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

constexpr auto const expected_A_C_min10 =
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

constexpr auto const expected_A_C_f15 =
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

constexpr auto const expected_A_C_f20 =
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

pareto_set<routing::journey> search(timetable const& tt,
                                    rt_timetable const* rtt,
                                    std::string_view const from,
                                    std::string_view const to,
                                    routing::start_time_t const start_time,
                                    direction const dir,
                                    routing::transfer_time_settings const tts) {
  return raptor_search(tt, rtt, dir == direction::kForward ? from : to,
                       dir == direction::kForward ? to : from, start_time, dir,
                       routing::all_clasz_allowed(), false, 0U, tts);
}

}  // namespace

TEST(routing, transfer_time_settings_test) {
  auto tt = timetable{};

  tt.date_range_ = {date::sys_days{2019_y / May / 1},
                    date::sys_days{2019_y / May / 2}};
  loader::register_special_stations(tt);
  loader::gtfs::load_timetable({}, source_idx_t{0},
                               loader::mem_dir::read(test_files), tt);
  loader::finalize(tt);

  for (auto const dir : {direction::kForward, direction::kBackward}) {

    {  // A -> C, default transfer time (= 2 min)
      auto const results =
          search(tt, nullptr, "A", "C", tt.date_range_, dir, {});
      EXPECT_EQ(expected_A_C_default, results_to_str(results, tt));
    }

    {  // A -> C, min 10 min transfer time (= 10 min)
      auto const results =
          search(tt, nullptr, "A", "C", tt.date_range_, dir,
                 {.default_ = false, .min_transfer_time_ = duration_t{10}});
      EXPECT_EQ(expected_A_C_min10, results_to_str(results, tt));
    }

    {  // A -> C, 1.5x transfer time (= 3 min)
      auto const results = search(tt, nullptr, "A", "C", tt.date_range_, dir,
                                  {.default_ = false, .factor_ = 1.5F});
      EXPECT_EQ(expected_A_C_f15, results_to_str(results, tt));
    }

    {  // A -> C, 2.0x transfer time (= 4 min)
      auto const results = search(tt, nullptr, "A", "C", tt.date_range_, dir,
                                  {.default_ = false, .factor_ = 2.0F});
      EXPECT_EQ(expected_A_C_f20, results_to_str(results, tt));
    }

    {  // A -> C, min 10 min transfer time, 2.0x transfer time (= 10 min)
      auto const results = search(tt, nullptr, "A", "C", tt.date_range_, dir,
                                  {.default_ = false,
                                   .min_transfer_time_ = duration_t{10},
                                   .factor_ = 2.0F});
      EXPECT_EQ(expected_A_C_min10, results_to_str(results, tt));
    }

    {  // A -> C, min 3 min transfer time, 2.0x transfer time (= 4 min)
      auto const results = search(tt, nullptr, "A", "C", tt.date_range_, dir,
                                  {.default_ = false,
                                   .min_transfer_time_ = duration_t{3},
                                   .factor_ = 2.0F});
      EXPECT_EQ(expected_A_C_f20, results_to_str(results, tt));
    }
  }
}
