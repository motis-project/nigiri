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
#include "nigiri/rt/util.h"
#include "nigiri/timetable.h"

#include "../loader/hrd/hrd_timetable.h"
#include "../raptor_search.h"

#include "./util.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace nigiri::rt;
using namespace std::chrono_literals;
using namespace std::string_literals;
using namespace std::string_view_literals;
using namespace nigiri::test;
using nigiri::test::raptor_search;

namespace {

constexpr auto const test_files = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,43.422095,-80.462740,,
B,B,,43.419023,-80.466600,,
C,C,,43.417796,-80.473666,,
D,D,,43.415733,-80.480340,,
E,E,,43.412766,-80.491494,,
F,F,,1.0,2.0,,
G,G,,3.0,4.0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,RE 1,,,3
R2,DB,RE 2,,,3
R3,DB,RE 3,,,3
R4,DB,RE 4,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S1,T1,RE 1,1
R2,S1,T2,RE 2,1
R3,S1,T3,RE 3,1
R4,S1,T4,RE 4,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,00:30:00,00:30:00,A,1,0,0
T1,10:00:00,10:00:00,B,2,0,0
T2,26:10:00,26:10:00,B,1,0,0
T2,27:00:00,27:00:00,C,2,0,0
T2,28:00:00,28:00:00,D,3,0,0
T3,28:30:00,28:30:00,D,1,0,0
T3,28:40:00,28:40:00,E,2,0,0
T4,29:00:00,29:00:00,E,1,0,0
T4,36:00:00,36:00:00,F,2,0,0
T4,49:00:00,49:00:00,G,3,0,0

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)"sv;

// Test VP
// Position: At Stop B
// Timestamp: 5min after scheduled arrival
auto const kVehiclePosition =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1556697900"
 },
 "entity": [
  {
    "id": "3248651",
    "isDeleted": false,
    "vehicle": {
     "trip": {
      "tripId": "T1",
      "startTime": "00:30:00",
      "startDate": "20190501",
      "routeId": "R1"
     },
     "position": {
      "latitude": "43.419023",
      "longitude": "-80.466600"
     },
     "timestamp": "1556697900",
     "vehicle": {
      "id": "v1"
     },
     "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  }
 ]
})"s;

constexpr auto const expected =
    R"(   0: A       A...............................................                                                             d: 30.04 22:30 [01.05 00:30]  RT 30.04 22:30 [01.05 00:30]  [{name=RE 1, day=2019-04-30, id=T1, src=0}]
   1: B       B............................................... a: 01.05 08:00 [01.05 10:00]  RT 01.05 08:00 [01.05 10:00]

   1: B       B...............................................                                                             d: 02.05 00:10 [02.05 02:10]  RT 02.05 00:10 [02.05 02:10]  [{name=RE 2, day=2019-04-30, id=T2, src=0}]
   2: C       C............................................... a: 02.05 01:00 [02.05 03:00]  RT 02.05 01:05 [02.05 03:05]  d: 02.05 01:00 [02.05 03:00]  RT 02.05 01:05 [02.05 03:05]  [{name=RE 2, day=2019-04-30, id=T2, src=0}]
   3: D       D............................................... a: 02.05 02:00 [02.05 04:00]  RT 02.05 02:05 [02.05 04:05]

   3: D       D...............................................                                                             d: 02.05 02:30 [02.05 04:30]  RT 02.05 02:35 [02.05 04:35]  [{name=RE 3, day=2019-04-30, id=T3, src=0}]
   4: E       E............................................... a: 02.05 02:40 [02.05 04:40]  RT 02.05 02:45 [02.05 04:45]

   4: E       E...............................................                                                             d: 02.05 03:00 [02.05 05:00]  RT 02.05 03:05 [02.05 05:05]  [{name=RE 4, day=2019-04-30, id=T4, src=0}]
   5: F       F............................................... a: 02.05 10:00 [02.05 12:00]  RT 02.05 10:05 [02.05 12:05]  d: 02.05 10:00 [02.05 12:00]  RT 02.05 10:05 [02.05 12:05]  [{name=RE 4, day=2019-04-30, id=T4, src=0}]
   6: G       G............................................... a: 02.05 23:00 [03.05 01:00]  RT 02.05 23:05 [03.05 01:05]

)";

}  // namespace

TEST(rt, rt_VP_block_id_test) {
  auto tt = timetable{};
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  load_timetable({}, source_idx_t{0}, mem_dir::read(test_files), tt);
  finalize(tt);
  auto rtt = rt::create_rt_timetable(tt, May / 1 / 2019);

  auto const msg = rt::json_to_protobuf(kVehiclePosition);

  transit_realtime::TripDescriptor td1;
  td1.set_start_date("20190501");
  td1.set_trip_id("T1");
  td1.set_start_time("00:30:00");
  transit_realtime::TripDescriptor td2;
  td2.set_start_date("20190501");
  td2.set_trip_id("T2");
  td2.set_start_time("26:10:00");
  transit_realtime::TripDescriptor td3;
  td3.set_start_date("20190501");
  td3.set_trip_id("T3");
  td3.set_start_time("28:30:00");
  transit_realtime::TripDescriptor td4;
  td4.set_start_date("20190501");
  td4.set_trip_id("T4");
  td4.set_start_time("29:00:00");

  auto const stats = gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg, true);

  EXPECT_EQ(1U, stats.total_entities_success_);

  auto const [r1, t1] = rt::gtfsrt_resolve_run(date::sys_days{May / 1 / 2019}, tt,
                                             &rtt, source_idx_t{0}, td1);
  ASSERT_TRUE(r1.valid());

  auto const [r2, t2] = rt::gtfsrt_resolve_run(date::sys_days{May / 1 / 2019}, tt,
                                             &rtt, source_idx_t{0}, td2);
  ASSERT_TRUE(r2.valid());

  auto const [r3, t3] = rt::gtfsrt_resolve_run(date::sys_days{May / 1 / 2019}, tt,
                                             &rtt, source_idx_t{0}, td3);
  ASSERT_TRUE(r3.valid());

  auto const [r4, t4] = rt::gtfsrt_resolve_run(date::sys_days{May / 1 / 2019}, tt,
                                             &rtt, source_idx_t{0}, td4);
  ASSERT_TRUE(r4.valid());

  std::stringstream ss;
  ss << rt::frun{tt, &rtt, r1} << "\n"
     << rt::frun{tt, &rtt, r2} << "\n"
     << rt::frun{tt, &rtt, r3} << "\n"
     << rt::frun{tt, &rtt, r4} << "\n";
  EXPECT_EQ(expected, ss.str());
}