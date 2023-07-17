#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"

#include "../loader/hrd/hrd_timetable.h"

#include "./util.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace std::chrono_literals;
using namespace std::string_view_literals;
using namespace nigiri::test;

namespace {

constexpr auto const test_files = R"(
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

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,RE 1,,,3
R2,DB,RE 2,,,3
R3,DB,RE 3,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S1,T1,RE 1,1
R2,S1,T2,RE 2,1
R3,S1,T3,RE 3,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,00:30:00,00:30:00,A,1,0,0
T1,10:00:00,10:00:00,B,2,0,0
T2,26:10:00,26:10:00,B,1,0,0
T2,27:00:00,27:00:00,C,2,0,0
T2,28:00:00,28:00:00,D,3,0,0
T3,28:30:00,28:30:00,D,1,0,0
T3,28:40:00,28:40:00,E,2,0,0

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)"sv;

constexpr auto const expected =
    R"(   0: A       A...............................................                                                             d: 30.04 22:30 [01.05 00:30]  RT 30.04 23:30 [01.05 01:30]  [{name=Bus RE 1, day=2019-04-30, id=T1, src=0}]
   1: B       B............................................... a: 01.05 08:00 [01.05 10:00]  RT 01.05 09:00 [01.05 11:00]

   1: B       B...............................................                                                             d: 02.05 00:10 [02.05 02:10]  RT 02.05 00:11 [02.05 02:11]  [{name=Bus RE 2, day=2019-04-30, id=T2, src=0}]
   2: C       C............................................... a: 02.05 01:00 [02.05 03:00]  RT 02.05 02:00 [02.05 04:00]  d: 02.05 01:00 [02.05 03:00]  RT 02.05 02:00 [02.05 04:00]  [{name=Bus RE 2, day=2019-04-30, id=T2, src=0}]
   3: D       D............................................... a: 02.05 02:00 [02.05 04:00]  RT 02.05 03:00 [02.05 05:00]

   3: D       D...............................................                                                             d: 02.05 02:30 [02.05 04:30]  RT 02.05 03:00 [02.05 05:00]  [{name=Bus RE 3, day=2019-04-30, id=T3, src=0}]
   4: E       E............................................... a: 02.05 02:40 [02.05 04:40]  RT 02.05 03:00 [02.05 05:00]

)";

}  // namespace

TEST(rt, rt_block_id_test) {
  auto tt = timetable{};
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  load_timetable({}, source_idx_t{0}, mem_dir::read(test_files), tt);
  finalize(tt);
  auto rtt = rt::create_rt_timetable(tt, May / 1 / 2019);

  auto const msg1 = test::to_feed_msg(
      {trip{.trip_id_ = "T1",
            .delays_ = {trip::delay{.stop_id_ = "A",
                                    .ev_type_ = nigiri::event_type::kArr,
                                    .delay_minutes_ = 60}}},
       trip{.trip_id_ = "T2",
            .delays_ = {{.seq_ = 1,
                         .ev_type_ = event_type::kDep,
                         .delay_minutes_ = 1U},
                        {.seq_ = 2,
                         .ev_type_ = event_type::kArr,
                         .delay_minutes_ = 60U}}},
       trip{.trip_id_ = "T3",
            .delays_ = {{.seq_ = 1,
                         .ev_type_ = event_type::kDep,
                         .delay_minutes_ = 0U}}}},
      date::sys_days{May / 1 / 2019} + 22h);

  auto const stats =
      rt::gtfsrt_update_msg(tt, rtt, source_idx_t{0}, "tag", msg1);

  EXPECT_EQ(3U, stats.total_entities_success_);

  auto const [r1, t1] = rt::gtfsrt_resolve_run(
      date::sys_days{May / 1 / 2019}, tt, rtt, source_idx_t{0},
      msg1.entity(0).trip_update().trip());
  ASSERT_TRUE(r1.valid());

  auto const [r2, t2] = rt::gtfsrt_resolve_run(
      date::sys_days{May / 1 / 2019}, tt, rtt, source_idx_t{0},
      msg1.entity(1).trip_update().trip());
  ASSERT_TRUE(r2.valid());

  auto const [r3, t3] = rt::gtfsrt_resolve_run(
      date::sys_days{May / 1 / 2019}, tt, rtt, source_idx_t{0},
      msg1.entity(2).trip_update().trip());
  ASSERT_TRUE(r3.valid());

  std::stringstream ss;
  ss << rt::frun{tt, &rtt, r1} << "\n"
     << rt::frun{tt, &rtt, r2} << "\n"
     << rt::frun{tt, &rtt, r3} << "\n";
  EXPECT_EQ(expected, ss.str());
}