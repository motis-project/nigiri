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
A,A,,1.0,2.0,,
B,B,,3.0,4.0,,
C,C,,5.0,6.0,,
D,D,,7.0,8.0,,
E,E,,9.0,10.0,,
F,F,,11.0,12.0,,
G,G,,13.0,14.0,,

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
T3,28:30:00,28:35:00,D,1,0,0
T3,28:40:00,28:40:00,E,2,0,0
T4,29:00:00,29:00:00,E,1,0,0
T4,36:00:00,36:10:00,F,2,0,0
T4,49:00:00,49:00:00,G,3,0,0

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)"sv;

// Test: correct VehiclePosition for delay beyond trip borders
// Test: arr/dep times > 24:00:00; > 48:00:00
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
      "latitude": "1.0",
      "longitude": "2.0"
     },
     "timestamp": "1556698000",
     "vehicle": {
      "id": "v1"
     },
     "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
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
      "latitude": "3.0",
      "longitude": "4.0"
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

// Test: located at several stops at the same time
// Position: At Stop A, B, C, D, E
auto const kVehiclePosition2 =
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
      "latitude": "1.0",
      "longitude": "2.0"
     },
     "timestamp": "1556697900",
     "vehicle": {
      "id": "v1"
     },
     "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "3248652",
    "isDeleted": false,
    "vehicle": {
     "trip": {
      "tripId": "T1",
      "startTime": "00:30:00",
      "startDate": "20190501",
      "routeId": "R1"
     },
     "position": {
      "latitude": "3.0",
      "longitude": "4.0"
     },
     "timestamp": "1556697900",
     "vehicle": {
      "id": "v1"
     },
     "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "3248653",
    "isDeleted": false,
    "vehicle": {
     "trip": {
      "tripId": "T1",
      "startTime": "00:30:00",
      "startDate": "20190501",
      "routeId": "R1"
     },
     "position": {
      "latitude": "5.0",
      "longitude": "6.0"
     },
     "timestamp": "1556697900",
     "vehicle": {
      "id": "v1"
     },
     "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "3248654",
    "isDeleted": false,
    "vehicle": {
     "trip": {
      "tripId": "T1",
      "startTime": "00:30:00",
      "startDate": "20190501",
      "routeId": "R1"
     },
     "position": {
      "latitude": "7.0",
      "longitude": "8.0"
     },
     "timestamp": "1556697900",
     "vehicle": {
      "id": "v1"
     },
     "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "3248655",
    "isDeleted": false,
    "vehicle": {
     "trip": {
      "tripId": "T1",
      "startTime": "00:30:00",
      "startDate": "20190501",
      "routeId": "R1"
     },
     "position": {
      "latitude": "9.0",
      "longitude": "10.0"
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

// Test: timestamps before start time of run and after the present point of time
auto const kVehiclePosition3 =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1556670300"
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
      "latitude": "3.0",
      "longitude": "4.0"
     },
     "timestamp": "1556670300",
     "vehicle": {
      "id": "v1"
     },
     "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
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
      "latitude": "1.0",
      "longitude": "2.0"
     },
     "timestamp": "9999999999",
     "vehicle": {
      "id": "v1"
     },
     "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  }
 ]
})"s;

// Test: several correct and incorrect VehiclePositions
// new delay (20min) from stop D
// new delay (-5min) from stop F
auto const kVehiclePosition4 =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1556670300"
 },
 "entity": [
  {
    "id": "3248640",
    "isDeleted": false,
    "vehicle": {
     "trip": {
      "tripId": "T1",
      "startTime": "00:30:00",
      "startDate": "20190501",
      "routeId": "R1"
     },
     "position": {
      "latitude": "1.0",
      "longitude": "2.0"
     },
     "timestamp": "1556697900"
    }
  },
  {
    "id": "3248640",
    "isDeleted": false,
    "vehicle": {
     "trip": {
      "tripId": "T1",
      "startTime": "00:30:00",
      "startDate": "20190501",
      "routeId": "R1"
     },
     "position": {
      "latitude": "20.0",
      "longitude": "25.0"
     },
     "timestamp": "1556670300"
    }
  },
  {
    "id": "3248652",
    "isDeleted": false,
    "vehicle": {
     "trip": {
      "tripId": "T2",
      "startTime": "26:10:00",
      "startDate": "20190501",
      "routeId": "R2"
     },
     "position": {
      "latitude": "7.0",
      "longitude": "8.0"
     },
     "timestamp": "1556763600"
    }
  },
  {
    "id": "3248652",
    "isDeleted": false,
    "vehicle": {
     "trip": {
      "tripId": "T3",
      "startTime": "28:35:00",
      "startDate": "20190501",
      "routeId": "R3"
     },
     "position": {
      "latitude": "9.0",
      "longitude": "10.0"
     },
     "timestamp": "1556763400"
    }
  },
  {
    "id": "3248653",
    "isDeleted": false,
    "vehicle": {
     "trip": {
      "tripId": "T3",
      "startTime": "28:35:00",
      "startDate": "20190501",
      "routeId": "R3"
     },
     "position": {
      "latitude": "9.0",
      "longitude": "10.0"
     },
     "timestamp": "1556791000"
    }
  },
  {
    "id": "3248653",
    "isDeleted": false,
    "vehicle": {
     "trip": {
      "tripId": "T4",
      "startTime": "29:00:00",
      "startDate": "20190501",
      "routeId": "R4"
     },
     "position": {
      "latitude": "11.0",
      "longitude": "12.0"
     },
     "timestamp": "1556790900"
    }
  }
 ]
})"s;

// Test: Invalid TripDescriptors -> expected not to crash
// Test: no timestamp -> expected not to crash
auto const kVehiclePosition5 =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1556670300"
 },
 "entity": [
  {
    "id": "3248640",
    "isDeleted": false,
    "vehicle": {
     "trip": {
      "tripId": "T14",
      "startTime": "00:30:00",
      "startDate": "20190501",
      "routeId": "R1"
     },
     "position": {
      "latitude": "20.0",
      "longitude": "25.0"
     },
     "timestamp": "1556670300"
    }
  },
  {
    "id": "3248652",
    "isDeleted": false,
    "vehicle": {
     "trip": {
      "tripId": "T3",
      "startTime": "24:35:00",
      "startDate": "20190501",
      "routeId": "R3"
     },
     "position": {
      "latitude": "9.0",
      "longitude": "10.0"
     },
     "timestamp": "1556763400"
    }
  },
  {
    "id": "3248653",
    "isDeleted": false,
    "vehicle": {
     "trip": {
      "tripId": "T3",
      "startTime": "28:35:00",
      "routeId": "R3"
     },
     "position": {
      "latitude": "9.0",
      "longitude": "10.0"
     },
     "timestamp": "1556791000"
    }
  },
  {
    "id": "3248653",
    "isDeleted": false,
    "vehicle": {
     "trip": {
      "tripId": "T4",
      "startTime": "29:00:00",
      "startDate": "20190501",
      "routeId": "R4"
     },
     "position": {
      "latitude": "11.0",
      "longitude": "12.0"
     }
    }
  }
 ]
})"s;

constexpr auto const expected =
    R"(   0: A       A...............................................                                                             d: 30.04 22:30 [01.05 00:30]  RT 30.04 22:35 [01.05 00:35]  [{name=RE 1, day=2019-04-30, id=T1, src=0}]
   1: B       B............................................... a: 01.05 08:00 [01.05 10:00]  RT 01.05 08:05 [01.05 10:05]

   1: B       B...............................................                                                             d: 02.05 00:10 [02.05 02:10]  RT 02.05 00:15 [02.05 02:15]  [{name=RE 2, day=2019-04-30, id=T2, src=0}]
   2: C       C............................................... a: 02.05 01:00 [02.05 03:00]  RT 02.05 01:05 [02.05 03:05]  d: 02.05 01:00 [02.05 03:00]  RT 02.05 01:05 [02.05 03:05]  [{name=RE 2, day=2019-04-30, id=T2, src=0}]
   3: D       D............................................... a: 02.05 02:00 [02.05 04:00]  RT 02.05 02:05 [02.05 04:05]

   3: D       D...............................................                                                             d: 02.05 02:35 [02.05 04:35]  RT 02.05 02:40 [02.05 04:40]  [{name=RE 3, day=2019-04-30, id=T3, src=0}]
   4: E       E............................................... a: 02.05 02:40 [02.05 04:40]  RT 02.05 02:45 [02.05 04:45]

   4: E       E...............................................                                                             d: 02.05 03:00 [02.05 05:00]  RT 02.05 03:05 [02.05 05:05]  [{name=RE 4, day=2019-04-30, id=T4, src=0}]
   5: F       F............................................... a: 02.05 10:00 [02.05 12:00]  RT 02.05 10:05 [02.05 12:05]  d: 02.05 10:10 [02.05 12:10]  RT 02.05 10:15 [02.05 12:15]  [{name=RE 4, day=2019-04-30, id=T4, src=0}]
   6: G       G............................................... a: 02.05 23:00 [03.05 01:00]  RT 02.05 23:05 [03.05 01:05]

)";

constexpr auto const expected4 =
    R"(   0: A       A...............................................                                                             d: 30.04 22:30 [01.05 00:30]  RT 30.04 22:25 [01.05 00:25]  [{name=RE 1, day=2019-04-30, id=T1, src=0}]
   1: B       B............................................... a: 01.05 08:00 [01.05 10:00]  RT 01.05 07:55 [01.05 09:55]

   1: B       B...............................................                                                             d: 02.05 00:10 [02.05 02:10]  RT 02.05 00:05 [02.05 02:05]  [{name=RE 2, day=2019-04-30, id=T2, src=0}]
   2: C       C............................................... a: 02.05 01:00 [02.05 03:00]  RT 02.05 00:55 [02.05 02:55]  d: 02.05 01:00 [02.05 03:00]  RT 02.05 00:55 [02.05 02:55]  [{name=RE 2, day=2019-04-30, id=T2, src=0}]
   3: D       D............................................... a: 02.05 02:00 [02.05 04:00]  RT 02.05 01:55 [02.05 03:55]

   3: D       D...............................................                                                             d: 02.05 02:35 [02.05 04:35]  RT 02.05 02:30 [02.05 04:30]  [{name=RE 3, day=2019-04-30, id=T3, src=0}]
   4: E       E............................................... a: 02.05 02:40 [02.05 04:40]  RT 02.05 02:35 [02.05 04:35]

   4: E       E...............................................                                                             d: 02.05 03:00 [02.05 05:00]  RT 02.05 02:55 [02.05 04:55]  [{name=RE 4, day=2019-04-30, id=T4, src=0}]
   5: F       F............................................... a: 02.05 10:00 [02.05 12:00]  RT 02.05 09:55 [02.05 11:55]  d: 02.05 10:10 [02.05 12:10]  RT 02.05 10:05 [02.05 12:05]  [{name=RE 4, day=2019-04-30, id=T4, src=0}]
   6: G       G............................................... a: 02.05 23:00 [03.05 01:00]  RT 02.05 22:55 [03.05 00:55]

)";

constexpr auto const expected_stats =
    R"(total_entities=2, total_entities_success=2 (100%), total_vehicles=2 (100%))";

constexpr auto const expected_stats2 =
    R"(total_entities=5, total_entities_success=5 (100%), total_vehicles=5 (100%))";

constexpr auto const expected_stats3 =
    R"(total_entities=2, total_entities_success=2 (100%), total_vehicles=2 (100%))";

constexpr auto const expected_stats4 =
    R"(total_entities=6, total_entities_success=5 (83.3333%), total_vehicles=6 (100%), vehicle_position_position_not_at_stop=1 (16.6667%))";

}  // namespace

TEST(rt, rt_VP_block_id_test) {
  auto tt = timetable{};
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  load_timetable({}, source_idx_t{0}, mem_dir::read(test_files), tt);
  finalize(tt);
  auto rtt = rt::create_rt_timetable(tt, May / 1 / 2019);

  auto const msg = rt::json_to_protobuf(kVehiclePosition);
  auto const msg2 = rt::json_to_protobuf(kVehiclePosition2);
  auto const msg3 = rt::json_to_protobuf(kVehiclePosition3);
  auto const msg4 = rt::json_to_protobuf(kVehiclePosition4);
  auto const msg5 = rt::json_to_protobuf(kVehiclePosition5);

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
  td3.set_start_time("28:35:00");
  transit_realtime::TripDescriptor td4;
  td4.set_start_date("20190501");
  td4.set_trip_id("T4");
  td4.set_start_time("29:00:00");

  auto const stats = gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg, true);

  auto const [r1, t1] = rt::gtfsrt_resolve_run(date::sys_days{May / 1 / 2019},
                                               tt, &rtt, source_idx_t{0}, td1);
  ASSERT_TRUE(r1.valid());

  auto const [r2, t2] = rt::gtfsrt_resolve_run(date::sys_days{May / 1 / 2019},
                                               tt, &rtt, source_idx_t{0}, td2);
  ASSERT_TRUE(r2.valid());

  auto const [r3, t3] = rt::gtfsrt_resolve_run(date::sys_days{May / 1 / 2019},
                                               tt, &rtt, source_idx_t{0}, td3);
  ASSERT_TRUE(r3.valid());

  auto const [r4, t4] = rt::gtfsrt_resolve_run(date::sys_days{May / 1 / 2019},
                                               tt, &rtt, source_idx_t{0}, td4);
  ASSERT_TRUE(r4.valid());

  // Test: correct VehiclePosition for delay beyond trip borders
  // Test: arr/dep times > 24:00:00; > 48:00:00
  std::stringstream ss;
  ss << rt::frun{tt, &rtt, r1} << "\n"
     << rt::frun{tt, &rtt, r2} << "\n"
     << rt::frun{tt, &rtt, r3} << "\n"
     << rt::frun{tt, &rtt, r4} << "\n";
  EXPECT_EQ(expected, ss.str());

  // Test: correct VehiclePosition for delay beyond trip borders
  // Test: one VehiclePosition with illegal stop
  // Test: arr/dep times > 24:00:00; > 48:00:00
  auto stats_ss = std::stringstream{};
  stats_ss << stats;
  EXPECT_EQ(expected_stats, stats_ss.str());

  // Test: located at several stops at the same time
  auto const stats2 =
      gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg2, true);
  auto stats2_ss = std::stringstream{};
  stats2_ss << stats2;
  EXPECT_EQ(expected_stats2, stats2_ss.str());

  // Test: timestamps before start time of run and after the present point of
  // time
  auto const stats3 =
      gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg3, true);
  auto stats3_ss = std::stringstream{};
  stats3_ss << stats3;
  EXPECT_EQ(expected_stats3, stats3_ss.str());

  // Test: several correct and incorrect VehiclePositions
  // new delay (20min) from stop D
  // new delay (-5min) from stop F
  auto const [r4_1, t4_1] = rt::gtfsrt_resolve_run(
      date::sys_days{May / 1 / 2019}, tt, &rtt, source_idx_t{0}, td1);
  ASSERT_TRUE(r4_1.valid());
  ASSERT_TRUE(r4_1.is_rt());

  auto const [r4_2, t4_2] = rt::gtfsrt_resolve_run(
      date::sys_days{May / 1 / 2019}, tt, &rtt, source_idx_t{0}, td2);
  ASSERT_TRUE(r4_2.valid());
  ASSERT_TRUE(r4_2.is_rt());

  auto const [r4_3, t4_3] = rt::gtfsrt_resolve_run(
      date::sys_days{May / 1 / 2019}, tt, &rtt, source_idx_t{0}, td3);
  ASSERT_TRUE(r4_3.valid());
  ASSERT_TRUE(r4_3.is_rt());

  auto const [r4_4, t4_4] = rt::gtfsrt_resolve_run(
      date::sys_days{May / 1 / 2019}, tt, &rtt, source_idx_t{0}, td4);
  ASSERT_TRUE(r4_4.valid());
  ASSERT_TRUE(r4_4.is_rt());

  auto const stats4 =
      gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg4, true);

  std::stringstream ss4;
  ss4 << rt::frun{tt, &rtt, r4_1} << "\n"
      << rt::frun{tt, &rtt, r4_2} << "\n"
      << rt::frun{tt, &rtt, r4_3} << "\n"
      << rt::frun{tt, &rtt, r4_4} << "\n";
  EXPECT_EQ(expected4, ss4.str());

  auto stats4_ss = std::stringstream{};
  stats4_ss << stats4;
  EXPECT_EQ(expected_stats4, stats4_ss.str());

  // Test: Invalid TripDescriptors -> expected not to crash
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg5, true);
}