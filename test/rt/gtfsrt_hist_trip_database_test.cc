#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/delay_prediction.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/util.h"
#include "nigiri/timetable.h"

#include "./util.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace nigiri::rt;
using namespace date;
using namespace std::chrono_literals;
using namespace std::string_literals;
using namespace std::string_view_literals;

namespace {

mem_dir test_files() {
  return mem_dir::read(R"(
     "(
# agency.txt
agency_name,agency_url,agency_timezone,agency_lang,agency_phone,agency_fare_url,agency_id
"grt",https://grt.ca,America/New_York,en,519-585-7555,http://www.grt.ca/en/fares/FarePrices.asp,grt

# stops.txt
stop_id,stop_code,stop_name,stop_desc,stop_lat,stop_lon,zone_id,stop_url,location_type,parent_station,wheelchair_boarding,platform_code
2351,2351,Block Line Station,,  43.422095, -80.462740,,
1033,1033,Block Line / Hanover,,  43.419023, -80.466600,,,0,,1,
2086,2086,Block Line / Kingswood,,  43.417796, -80.473666,,,0,,1,
2885,2885,Block Line / Strasburg,,  43.415733, -80.480340,,,0,,1,
2888,2888,Block Line / Laurentian,,  43.412766, -80.491494,,,0,,1,
3189,3189,Block Line / Westmount,,  43.411515, -80.498966,,,0,,1,
3895,3895,Fischer-Hallman / Westmount,,  43.406717, -80.500091,,,0,,1,
3893,3893,Fischer-Hallman / Activa,,  43.414221, -80.508534,,,0,,1,
2969,2969,Fischer-Hallman / Ottawa,,  43.416570, -80.510880,,,0,,1,
2971,2971,Fischer-Hallman / Mcgarry,,  43.423420, -80.518818,,,0,,1,
2986,2986,Fischer-Hallman / Queens,,  43.428585, -80.523337,,,0,,1,
3891,3891,Fischer-Hallman / Highland,,  43.431587, -80.525376,,,0,,1,
3143,3143,Fischer-Hallman / Victoria,,  43.436843, -80.529202,,,0,,1,
3144,3144,Fischer-Hallman / Stoke,,  43.439462, -80.535435,,,0,,1,
3146,3146,Fischer-Hallman / University Ave.,,  43.444402, -80.545691,,,0,,1,
1992,1992,Fischer-Hallman / Thorndale,,  43.448678, -80.550034,,,0,,1,
1972,1972,Fischer-Hallman / Erb,,  43.452906, -80.553686,,,0,,1,
3465,3465,Fischer-Hallman / Keats Way,,  43.458370, -80.557824,,,0,,1,
3890,3890,Fischer-Hallman / Columbia,,  43.467368, -80.565646,,,0,,1,
1117,1117,Columbia / U.W. - Columbia Lake Village,,  43.469091, -80.561788,,,0,,1,
3899,3899,Columbia / University Of Waterloo,,  43.474462, -80.546591,,,0,,1,
1223,1223,University Of Waterloo Station,,  43.474023, -80.540433,,
3887,3887,Phillip / Columbia,,  43.476409, -80.539399,,,0,,1,
2524,2524,Columbia / Hazel,,  43.480027, -80.531130,,,0,,1,
4073,4073,King / Columbia,,  43.482448, -80.526106,,,0,,1,
1916,1916,King / Weber,,  43.484988, -80.526677,,,0,,1,
1918,1918,King / Manulife,,  43.491207, -80.528026,,,0,,1,
1127,1127,Conestoga Station,,  43.498036, -80.528999,,

# calendar_dates.txt
service_id,date,exception_type
201-Weekday-66-23SUMM-1111100,20230703,1
201-Weekday-66-23SUMM-1111100,20230704,1
201-Weekday-66-23SUMM-1111100,20230705,1
201-Weekday-66-23SUMM-1111100,20230706,1
201-Weekday-66-23SUMM-1111100,20230707,1
201-Weekday-66-23SUMM-1111100,20230710,1
201-Weekday-66-23SUMM-1111100,20230711,1
201-Weekday-66-23SUMM-1111100,20230712,1
201-Weekday-66-23SUMM-1111100,20230713,1
201-Weekday-66-23SUMM-1111100,20230714,1
201-Weekday-66-23SUMM-1111100,20230717,1
201-Weekday-66-23SUMM-1111100,20230718,1
201-Weekday-66-23SUMM-1111100,20230719,1
201-Weekday-66-23SUMM-1111100,20230720,1
201-Weekday-66-23SUMM-1111100,20230721,1
201-Weekday-66-23SUMM-1111100,20230724,1
201-Weekday-66-23SUMM-1111100,20230725,1
201-Weekday-66-23SUMM-1111100,20230726,1
201-Weekday-66-23SUMM-1111100,20230727,1
201-Weekday-66-23SUMM-1111100,20230728,1
201-Weekday-66-23SUMM-1111100,20230731,1
201-Weekday-66-23SUMM-1111100,20230801,1
201-Weekday-66-23SUMM-1111100,20230802,1
201-Weekday-66-23SUMM-1111100,20230803,1
201-Weekday-66-23SUMM-1111100,20230804,1
201-Weekday-66-23SUMM-1111100,20230808,1
201-Weekday-66-23SUMM-1111100,20230809,1
201-Weekday-66-23SUMM-1111100,20230810,1
201-Weekday-66-23SUMM-1111100,20230811,1
201-Weekday-66-23SUMM-1111100,20230814,1
201-Weekday-66-23SUMM-1111100,20230815,1
201-Weekday-66-23SUMM-1111100,20230816,1
201-Weekday-66-23SUMM-1111100,20230817,1
201-Weekday-66-23SUMM-1111100,20230818,1
201-Weekday-66-23SUMM-1111100,20230821,1
201-Weekday-66-23SUMM-1111100,20230822,1
201-Weekday-66-23SUMM-1111100,20230823,1
201-Weekday-66-23SUMM-1111100,20230824,1
201-Weekday-66-23SUMM-1111100,20230825,1
201-Weekday-66-23SUMM-1111100,20230828,1
201-Weekday-66-23SUMM-1111100,20230829,1
201-Weekday-66-23SUMM-1111100,20230830,1
201-Weekday-66-23SUMM-1111100,20230831,1
201-Weekday-66-23SUMM-1111100,20230901,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
201,grt,iXpress Fischer-Hallman,,3,https://www.grt.ca/en/schedules-maps/schedules.aspx
202,grt,iXpress Station-Fischer,,3,https://www.grt.ca/en/schedules-maps/schedules.aspx

# trips.txt
route_id,service_id,trip_id,trip_headsign,direction_id,block_id,shape_id,wheelchair_accessible,bikes_allowed
201,201-Weekday-66-23SUMM-1111100,3248651,Conestoga Station,0,340341,2010025,1,1
201,201-Weekday-66-23SUMM-1111100,3248652,Conestoga Station,0,340342,2010025,1,1
202,201-Weekday-66-23SUMM-1111100,3248653,Conestoga Station,0,340343,2010025,1,1


# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
3248651,05:15:00,05:15:00,2351,1,0,0
3248651,05:16:00,05:16:00,1033,2,0,0
3248651,05:18:00,05:18:00,2086,3,0,0
3248651,05:19:00,05:19:00,2885,4,0,0
3248651,05:21:00,05:21:00,2888,5,0,0
3248651,05:22:00,05:22:00,3189,6,0,0
3248651,05:24:00,05:24:00,3895,7,0,0
3248651,05:26:00,05:26:00,3893,8,0,0
3248651,05:27:00,05:27:00,2969,9,0,0
3248651,05:29:00,05:29:00,2971,10,0,0
3248651,05:31:00,05:31:00,2986,11,0,0
3248651,05:32:00,05:32:00,3891,12,0,0
3248651,05:33:00,05:33:00,3143,13,0,0
3248651,05:35:00,05:35:00,3144,14,0,0
3248651,05:37:00,05:37:00,3146,15,0,0
3248651,05:38:00,05:38:00,1992,16,0,0
3248651,05:39:00,05:39:00,1972,17,0,0
3248651,05:40:00,05:40:00,3465,18,0,0
3248651,05:42:00,05:42:00,3890,19,0,0
3248651,05:43:00,05:43:00,1117,20,0,0
3248651,05:46:00,05:46:00,3899,21,0,0
3248651,05:47:00,05:49:00,1223,22,0,0
3248651,05:50:00,05:50:00,3887,23,0,0
3248651,05:53:00,05:53:00,2524,24,0,0
3248651,05:54:00,05:54:00,4073,25,0,0
3248651,05:55:00,05:55:00,1916,26,0,0
3248651,05:56:00,05:56:00,1918,27,0,0
3248651,05:58:00,05:58:00,1127,28,1,0
3248652,06:15:00,06:15:00,2351,1,0,0
3248652,06:16:00,06:16:00,1033,2,0,0
3248652,06:18:00,06:18:00,2086,3,0,0
3248652,06:19:00,06:19:00,2885,4,0,0
3248652,06:21:00,06:21:00,2888,5,0,0
3248652,06:22:00,06:22:00,3189,6,0,0
3248652,06:24:00,06:24:00,3895,7,0,0
3248652,06:26:00,06:26:00,3893,8,0,0
3248652,06:27:00,06:27:00,2969,9,0,0
3248652,06:29:00,06:29:00,2971,10,0,0
3248652,06:31:00,06:31:00,2986,11,0,0
3248652,06:32:00,06:32:00,3891,12,0,0
3248652,06:33:00,06:33:00,3143,13,0,0
3248652,06:35:00,06:35:00,3144,14,0,0
3248652,06:37:00,06:37:00,3146,15,0,0
3248652,06:38:00,06:38:00,1992,16,0,0
3248652,06:39:00,06:39:00,1972,17,0,0
3248652,06:40:00,06:40:00,3465,18,0,0
3248652,06:42:00,06:42:00,3890,19,0,0
3248652,06:43:00,06:43:00,1117,20,0,0
3248652,06:46:00,06:46:00,3899,21,0,0
3248652,06:47:00,06:49:00,1223,22,0,0
3248652,06:50:00,06:50:00,3887,23,0,0
3248652,06:53:00,06:53:00,2524,24,0,0
3248652,06:54:00,06:54:00,4073,25,0,0
3248652,06:55:00,06:55:00,1916,26,0,0
3248652,06:56:00,06:56:00,1918,27,0,0
3248652,06:58:00,06:58:00,1127,28,1,0
3248653,05:15:00,05:15:00,2351,1,0,0
3248653,05:16:00,05:16:00,1033,2,0,0
3248653,05:18:00,05:18:00,2086,3,0,0
3248653,05:19:00,05:19:00,2885,4,0,0
3248653,05:21:00,05:21:00,2888,5,0,0
3248653,05:22:00,05:22:00,3189,6,0,0
3248653,05:24:00,05:24:00,3895,7,0,0
3248653,05:26:00,05:26:00,3893,8,0,0
3248653,05:27:00,05:27:00,2969,9,0,0
3248653,05:29:00,05:29:00,2971,10,0,0
)");
}

// Pos 1 at Seg 0
// Pos 2 at Seg 5
// Pos 3 at Seg 5
// Pos 4 at Seg 6
auto const kVehiclePosition =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1691659440"
 },
 "entity": [
  {
    "id": "32486517",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "3248651",
      "startTime": "05:15:00",
      "startDate": "20230810",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.420946",
      "longitude": "-80.463974"
    },
    "timestamp": "1691658931",
    "vehicle": {
      "id": "v1"
    },
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "32486518",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "3248651",
      "startTime": "05:15:00",
      "startDate": "20230810",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.409295",
      "longitude": "-80.499004"
    },
    "timestamp": "1691659380",
    "vehicle": {
      "id": "v1"
    },
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "32486519",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "3248651",
      "startTime": "05:15:00",
      "startDate": "20230810",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.407145",
      "longitude": "-80.498714"
    },
    "timestamp": "1691659680",
    "vehicle": {
      "id": "v1"
    },
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "32486520",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "3248651",
      "startTime": "05:15:00",
      "startDate": "20230810",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.412195",
      "longitude": "-80.506761"
    },
    "timestamp": "1691659800",
    "vehicle": {
      "id": "v1"
    },
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  }
 ]
})"s;

auto const kVehiclePosition2 =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1691663040"
 },
 "entity": [
  {
    "id": "32486521",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "3248652",
      "startTime": "06:15:00",
      "startDate": "20230810",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.420946",
      "longitude": "-80.463974"
    },
    "timestamp": "1691662531",
    "vehicle": {
      "id": "v2"
    },
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "32486522",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "3248652",
      "startTime": "06:15:00",
      "startDate": "20230810",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.409295",
      "longitude": "-80.499004"
    },
    "timestamp": "1691662980",
    "vehicle": {
      "id": "v2"
    },
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "32486523",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "3248652",
      "startTime": "06:15:00",
      "startDate": "20230810",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.407145",
      "longitude": "-80.498714"
    },
    "timestamp": "1691663280",
    "vehicle": {
      "id": "v2"
    },
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "32486524",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "3248652",
      "startTime": "06:15:00",
      "startDate": "20230810",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.412195",
      "longitude": "-80.506761"
    },
    "timestamp": "1691663400",
    "vehicle": {
      "id": "v2"
    },
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  }
 ]
})"s;

auto const kVehiclePosition3 =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1691659440"
 },
 "entity": [
  {
    "id": "32486525",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "3248653",
      "startTime": "05:15:00",
      "startDate": "20230810",
      "routeId": "202"
    },
    "position": {
      "latitude": "43.420946",
      "longitude": "-80.463974"
    },
    "timestamp": "1691658931",
    "vehicle": {
      "id": "v3"
    },
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "32486526",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "3248653",
      "startTime": "05:15:00",
      "startDate": "20230810",
      "routeId": "202"
    },
    "position": {
      "latitude": "43.409295",
      "longitude": "-80.499004"
    },
    "timestamp": "1691659380",
    "vehicle": {
      "id": "v3"
    },
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "32486527",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "3248653",
      "startTime": "05:15:00",
      "startDate": "20230810",
      "routeId": "202"
    },
    "position": {
      "latitude": "43.407145",
      "longitude": "-80.498714"
    },
    "timestamp": "1691659680",
    "vehicle": {
      "id": "v3"
    },
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "32486528",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "3248653",
      "startTime": "05:15:00",
      "startDate": "20230810",
      "routeId": "202"
    },
    "position": {
      "latitude": "43.412195",
      "longitude": "-80.506761"
    },
    "timestamp": "1691659800",
    "vehicle": {
      "id": "v3"
    },
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  }
 ]
})"s;

constexpr auto const expected =
    R"(
cs_key_coord_seq_:
Key: Source: 0 Transport: 0
Coord_seq_Idx: 0
Key: Source: 0 Transport: 1
Coord_seq_Idx: 0
Key: Source: 0 Transport: 2
Coord_seq_Idx: 1

coord_seq_idx_coord_seq_:
Coord_seq_Idx: 0
Location_Sequence: 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,
Coord_seq_Idx: 1
Location_Sequence: 9,10,11,12,13,14,15,16,17,18,

coord_seq_idx_ttd_:
Coord_seq_Idx: 0
Trip_Time_Data_Idxs: 0,1,
Coord_seq_Idx: 1
Trip_Time_Data_Idxs: 2,

ttd_idx_trip_time_data_:
Trip_Time_Data_Idx: 0 Start_Time: 2023-08-10 09:15
Segment: 0 Progress: 0.122255 Timestamp: 2023-08-10 09:15
Segment: 5 Progress: 0.203121 Timestamp: 2023-08-10 09:23
Segment: 5 Progress: 0.771619 Timestamp: 2023-08-10 09:28
Segment: 6 Progress: 0.56827 Timestamp: 2023-08-10 09:30
Trip_Time_Data_Idx: 1 Start_Time: 2023-08-10 10:15
Segment: 0 Progress: 0.122255 Timestamp: 2023-08-10 10:15
Segment: 5 Progress: 0.203121 Timestamp: 2023-08-10 10:23
Segment: 5 Progress: 0.771619 Timestamp: 2023-08-10 10:28
Segment: 6 Progress: 0.56827 Timestamp: 2023-08-10 10:30
Trip_Time_Data_Idx: 2 Start_Time: 2023-08-10 09:15
Segment: 0 Progress: 0.122255 Timestamp: 2023-08-10 09:15
Segment: 5 Progress: 0.203121 Timestamp: 2023-08-10 09:23
Segment: 5 Progress: 0.771619 Timestamp: 2023-08-10 09:28
Segment: 6 Progress: 0.56827 Timestamp: 2023-08-10 09:30
)";

}  // namespace

TEST(rt, gtfsrt_hist_trip_database_test) {
  std::cout << "Test rt::gtfsrt_hist_trip_database_test" << std::endl;

  // Load static timetable.
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2023_y / August / 9},
                    date::sys_days{2023_y / August / 12}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  // Create empty RT timetable.
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2023_y / August / 10});

  auto dps = delay_prediction_storage{};

  auto vtm = vehicle_trip_matching{};

  // Create empty Trip Time Data Storage
  auto tts = hist_trip_times_storage{};

  auto dp = delay_prediction{algorithm::kIntelligent,
                             hist_trip_mode::kSameDay,
                             1,
                             5,
                             &dps,
                             &tts,
                             &vtm};

  // Update.
  auto const msg = rt::json_to_protobuf(kVehiclePosition);
  auto const msg2 = rt::json_to_protobuf(kVehiclePosition2);
  auto const msg3 = rt::json_to_protobuf(kVehiclePosition3);

  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg, &dp);
  std::cout << "\nA\n" << std::endl;
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg2, &dp);
  std::cout << "\nB\n" << std::endl;
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg3, &dp);

  // Print trip.
  transit_realtime::TripDescriptor td;
  td.set_start_date("20230810");
  td.set_trip_id("3248651");
  td.set_start_time("05:15:00");
  auto const [r, t] = gtfsrt_resolve_run(date::sys_days{May / 1 / 2019}, tt,
                                         &rtt, source_idx_t{0}, td);

  transit_realtime::TripDescriptor td2;
  td2.set_start_date("20230810");
  td2.set_trip_id("3248652");
  td2.set_start_time("06:15:00");
  auto const [r2, t2] = gtfsrt_resolve_run(date::sys_days{May / 1 / 2019}, tt,
                                           &rtt, source_idx_t{0}, td2);

  transit_realtime::TripDescriptor td3;
  td3.set_start_date("20230810");
  td3.set_trip_id("3248653");
  td3.set_start_time("05:15:00");
  auto const [r3, t3] = gtfsrt_resolve_run(date::sys_days{May / 1 / 2019}, tt,
                                           &rtt, source_idx_t{0}, td3);

  ASSERT_TRUE(r.valid());
  ASSERT_TRUE(r2.valid());
  ASSERT_TRUE(r3.valid());

  std::stringstream ss;
  ss << tts;

  EXPECT_EQ(expected, ss.str());
}