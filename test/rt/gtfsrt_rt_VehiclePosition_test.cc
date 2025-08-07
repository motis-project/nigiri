#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
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

# trips.txt
route_id,service_id,trip_id,trip_headsign,direction_id,block_id,shape_id,wheelchair_accessible,bikes_allowed
201,201-Weekday-66-23SUMM-1111100,3248651,Conestoga Station,0,340341,2010025,1,1

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
)");
}

// Test VP
// Position: At Stop 2885 (Strasburg)
// Timestamp: 5min after scheduled arrival
auto const kVehiclePosition =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1691659440"
 },
 "entity": [
  {
    "id": "3248651",
    "isDeleted": false,
    "vehicle": {
     "trip": {
      "tripId": "3248651",
      "startTime": "05:15:00",
      "startDate": "20230810",
      "routeId": "201"
     },
     "position": {
      "latitude": "43.415733",
      "longitude": "-80.480340"
     },
     "timestamp": "1691659440",
     "vehicle": {
      "id": "v1"
     },
     "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
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
      "latitude": "43.415733",
      "longitude": "-80.480340"
    },
    "timestamp": "1691658440",
    "vehicle": {
      "id": "v1"
    },
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  }
 ]
})"s;

auto const kVehiclePosition1 =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1691659440"
 },
 "entity": [
  {
    "id": "32486511",
    "isDeleted": false,
  }
 ]
})"s;

auto const kVehiclePosition2 =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1691659440"
 },
 "entity": [
  {
    "id": "32486512",
    "isDeleted": false,
    "vehicle": {
     "trip": {
      "tripId": "3248651",
      "startTime": "05:15:00",
      "startDate": "20230810",
      "routeId": "201"
     },
     "timestamp": "1691659440",
     "vehicle": {
      "id": "v1"
     },
     "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  }
 ]
})"s;

auto const kVehiclePosition4 =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1691659440"
 },
 "entity": [
  {
    "id": "32486514",
    "isDeleted": false,
    "vehicle": {
     "position": {
      "latitude": "43.415733",
      "longitude": "-80.480340"
     },
     "timestamp": "1691659440",
     "vehicle": {
      "id": "v1"
     },
     "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  }
 ]
})"s;

auto const kVehiclePosition5 =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1691659440"
 },
 "entity": [
  {
    "id": "32486515",
    "isDeleted": false,
    "vehicle": {
     "trip": {
      "startTime": "05:15:00",
      "startDate": "20230810",
      "routeId": "201"
     },
     "position": {
      "latitude": "43.415733",
      "longitude": "-80.480340"
     },
     "timestamp": "1691659440",
     "vehicle": {
      "id": "v1"
     },
     "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  }
 ]
})"s;

auto const kVehiclePosition6 =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1691659440"
 },
 "entity": [
  {
    "id": "32486516",
    "isDeleted": false,
    "vehicle": {
     "trip": {
      "tripId": "3248651",
      "startTime": "05:15:00",
      "startDate": "20230810"
     },
     "position": {
      "latitude": "43.415733",
      "longitude": "-80.480340"
     },
     "timestamp": "1691659440",
     "vehicle": {
      "id": "v1"
     },
     "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
 ]
})"s;

constexpr auto const expected = R"(
   0: 2351    Block Line Station..............................                                                             d: 10.08 09:15 [10.08 05:15]  RT 10.08 09:15 [10.08 05:15]  [{name=iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
   1: 1033    Block Line / Hanover............................ a: 10.08 09:16 [10.08 05:16]  RT 10.08 09:16 [10.08 05:16]  d: 10.08 09:16 [10.08 05:16]  RT 10.08 09:16 [10.08 05:16]  [{name=iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
   2: 2086    Block Line / Kingswood.......................... a: 10.08 09:18 [10.08 05:18]  RT 10.08 09:18 [10.08 05:18]  d: 10.08 09:18 [10.08 05:18]  RT 10.08 09:18 [10.08 05:18]  [{name=iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
   3: 2885    Block Line / Strasburg.......................... a: 10.08 09:19 [10.08 05:19]  RT 10.08 09:19 [10.08 05:19]  d: 10.08 09:19 [10.08 05:19]  RT 10.08 09:19 [10.08 05:19]  [{name=iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
   4: 2888    Block Line / Laurentian......................... a: 10.08 09:21 [10.08 05:21]  RT 10.08 09:26 [10.08 05:26]  d: 10.08 09:21 [10.08 05:21]  RT 10.08 09:26 [10.08 05:26]  [{name=iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
   5: 3189    Block Line / Westmount.......................... a: 10.08 09:22 [10.08 05:22]  RT 10.08 09:27 [10.08 05:27]  d: 10.08 09:22 [10.08 05:22]  RT 10.08 09:27 [10.08 05:27]  [{name=iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
   6: 3895    Fischer-Hallman / Westmount..................... a: 10.08 09:24 [10.08 05:24]  RT 10.08 09:29 [10.08 05:29]  d: 10.08 09:24 [10.08 05:24]  RT 10.08 09:29 [10.08 05:29]  [{name=iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
   7: 3893    Fischer-Hallman / Activa........................ a: 10.08 09:26 [10.08 05:26]  RT 10.08 09:31 [10.08 05:31]  d: 10.08 09:26 [10.08 05:26]  RT 10.08 09:31 [10.08 05:31]  [{name=iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
   8: 2969    Fischer-Hallman / Ottawa........................ a: 10.08 09:27 [10.08 05:27]  RT 10.08 09:32 [10.08 05:32]  d: 10.08 09:27 [10.08 05:27]  RT 10.08 09:32 [10.08 05:32]  [{name=iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
   9: 2971    Fischer-Hallman / Mcgarry....................... a: 10.08 09:29 [10.08 05:29]  RT 10.08 09:34 [10.08 05:34]  d: 10.08 09:29 [10.08 05:29]  RT 10.08 09:34 [10.08 05:34]  [{name=iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  10: 2986    Fischer-Hallman / Queens........................ a: 10.08 09:31 [10.08 05:31]  RT 10.08 09:36 [10.08 05:36]  d: 10.08 09:31 [10.08 05:31]  RT 10.08 09:36 [10.08 05:36]  [{name=iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  11: 3891    Fischer-Hallman / Highland...................... a: 10.08 09:32 [10.08 05:32]  RT 10.08 09:37 [10.08 05:37]  d: 10.08 09:32 [10.08 05:32]  RT 10.08 09:37 [10.08 05:37]  [{name=iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  12: 3143    Fischer-Hallman / Victoria...................... a: 10.08 09:33 [10.08 05:33]  RT 10.08 09:38 [10.08 05:38]  d: 10.08 09:33 [10.08 05:33]  RT 10.08 09:38 [10.08 05:38]  [{name=iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  13: 3144    Fischer-Hallman / Stoke......................... a: 10.08 09:35 [10.08 05:35]  RT 10.08 09:40 [10.08 05:40]  d: 10.08 09:35 [10.08 05:35]  RT 10.08 09:40 [10.08 05:40]  [{name=iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  14: 3146    Fischer-Hallman / University Ave................ a: 10.08 09:37 [10.08 05:37]  RT 10.08 09:42 [10.08 05:42]  d: 10.08 09:37 [10.08 05:37]  RT 10.08 09:42 [10.08 05:42]  [{name=iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  15: 1992    Fischer-Hallman / Thorndale..................... a: 10.08 09:38 [10.08 05:38]  RT 10.08 09:43 [10.08 05:43]  d: 10.08 09:38 [10.08 05:38]  RT 10.08 09:43 [10.08 05:43]  [{name=iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  16: 1972    Fischer-Hallman / Erb........................... a: 10.08 09:39 [10.08 05:39]  RT 10.08 09:44 [10.08 05:44]  d: 10.08 09:39 [10.08 05:39]  RT 10.08 09:44 [10.08 05:44]  [{name=iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  17: 3465    Fischer-Hallman / Keats Way..................... a: 10.08 09:40 [10.08 05:40]  RT 10.08 09:45 [10.08 05:45]  d: 10.08 09:40 [10.08 05:40]  RT 10.08 09:45 [10.08 05:45]  [{name=iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  18: 3890    Fischer-Hallman / Columbia...................... a: 10.08 09:42 [10.08 05:42]  RT 10.08 09:47 [10.08 05:47]  d: 10.08 09:42 [10.08 05:42]  RT 10.08 09:47 [10.08 05:47]  [{name=iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  19: 1117    Columbia / U.W. - Columbia Lake Village......... a: 10.08 09:43 [10.08 05:43]  RT 10.08 09:48 [10.08 05:48]  d: 10.08 09:43 [10.08 05:43]  RT 10.08 09:48 [10.08 05:48]  [{name=iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  20: 3899    Columbia / University Of Waterloo............... a: 10.08 09:46 [10.08 05:46]  RT 10.08 09:51 [10.08 05:51]  d: 10.08 09:46 [10.08 05:46]  RT 10.08 09:51 [10.08 05:51]  [{name=iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  21: 1223    University Of Waterloo Station.................. a: 10.08 09:47 [10.08 05:47]  RT 10.08 09:52 [10.08 05:52]  d: 10.08 09:49 [10.08 05:49]  RT 10.08 09:54 [10.08 05:54]  [{name=iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  22: 3887    Phillip / Columbia.............................. a: 10.08 09:50 [10.08 05:50]  RT 10.08 09:55 [10.08 05:55]  d: 10.08 09:50 [10.08 05:50]  RT 10.08 09:55 [10.08 05:55]  [{name=iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  23: 2524    Columbia / Hazel................................ a: 10.08 09:53 [10.08 05:53]  RT 10.08 09:58 [10.08 05:58]  d: 10.08 09:53 [10.08 05:53]  RT 10.08 09:58 [10.08 05:58]  [{name=iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  24: 4073    King / Columbia................................. a: 10.08 09:54 [10.08 05:54]  RT 10.08 09:59 [10.08 05:59]  d: 10.08 09:54 [10.08 05:54]  RT 10.08 09:59 [10.08 05:59]  [{name=iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  25: 1916    King / Weber.................................... a: 10.08 09:55 [10.08 05:55]  RT 10.08 10:00 [10.08 06:00]  d: 10.08 09:55 [10.08 05:55]  RT 10.08 10:00 [10.08 06:00]  [{name=iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  26: 1918    King / Manulife................................. a: 10.08 09:56 [10.08 05:56]  RT 10.08 10:01 [10.08 06:01]  d: 10.08 09:56 [10.08 05:56]  RT 10.08 10:01 [10.08 06:01]  [{name=iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  27: 1127    Conestoga Station............................... a: 10.08 09:58 [10.08 05:58]  RT 10.08 10:03 [10.08 06:03]
)"sv;

}  // namespace

TEST(rt, gtfs_rt_vp_update) {

  // Load static timetable.
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2023_y / August / 9},
                    date::sys_days{2023_y / August / 12}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  // Create empty RT timetable.
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2023_y / August / 10});

  // Update.
  auto const msg = rt::json_to_protobuf(kVehiclePosition);
  auto const msg1 = rt::json_to_protobuf(kVehiclePosition1);
  auto const msg2 = rt::json_to_protobuf(kVehiclePosition2);
  auto const msg4 = rt::json_to_protobuf(kVehiclePosition4);
  auto const msg5 = rt::json_to_protobuf(kVehiclePosition5);
  auto const msg6 = rt::json_to_protobuf(kVehiclePosition6);
  auto const stats = gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg, true);
  /**auto const stats1 =
      gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg1, true);
  auto const stats2 =
      gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg2, true);
  auto const stats4 =
      gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg4, true);
  auto const stats5 =
      gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg5, true);
  auto const stats6 =
      gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg6, true);**/

  // Print trip.
  transit_realtime::TripDescriptor td;
  td.set_start_date("20230810");
  td.set_trip_id("3248651");
  td.set_start_time("05:15:00");
  auto const [r, t] = rt::gtfsrt_resolve_run(date::sys_days{May / 1 / 2019}, tt,
                                             &rtt, source_idx_t{0}, td);
  ASSERT_TRUE(r.valid());

  auto const fr = rt::frun{tt, &rtt, r};
  auto ss = std::stringstream{};
  ss << "\n" << fr;
  EXPECT_EQ(expected, ss.str());
  EXPECT_EQ(0, stats.no_vehicle_position_);
  /**EXPECT_EQ(1, stats1.no_vehicle_position_);
  EXPECT_EQ(1, stats2.vehicle_position_without_position_);
  EXPECT_EQ(1, stats4.vehicle_position_without_trip_);
  EXPECT_EQ(1, stats5.vehicle_position_trip_without_trip_id_);
  EXPECT_EQ(1, stats6.vehicle_position_trip_without_route_id_);**/
  ASSERT_FALSE(fr.is_cancelled());
}