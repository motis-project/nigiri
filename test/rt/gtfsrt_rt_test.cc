#include "gtest/gtest.h"

#include "google/protobuf/util/json_util.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/timetable.h"

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

auto const kTripUpdate =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1691660324"
 },
 "entity": [
  {
    "id": "3248651",
    "isDeleted": false,
    "tripUpdate": {
     "trip": {
      "tripId": "3248651",
      "startTime": "05:15:00",
      "startDate": "20230810",
      "routeId": "201"
     },
     "stopTimeUpdate": [
      {
       "stopSequence": 15,
       "arrival": {
        "time": "1691660288"
       },
       "departure": {
        "time": "1691660288"
       },
       "stopId": "3146",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 16,
       "arrival": {
        "time": "1691660351"
       },
       "departure": {
        "time": "1691660351"
       },
       "stopId": "1992",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 17,
       "arrival": {
        "time": "1691660431"
       },
       "departure": {
        "time": "1691660431"
       },
       "stopId": "1972",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 18,
       "arrival": {
        "time": "1691660496"
       },
       "departure": {
        "time": "1691660496"
       },
       "stopId": "3465",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 19,
       "arrival": {
        "time": "1691660669"
       },
       "departure": {
        "time": "1691660669"
       },
       "stopId": "3890",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 20,
       "arrival": {
        "time": "1691660718"
       },
       "departure": {
        "time": "1691660718"
       },
       "stopId": "1117",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 21,
       "arrival": {
        "time": "1691660869"
       },
       "departure": {
        "time": "1691660869"
       },
       "stopId": "3899",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 22,
       "arrival": {
        "time": "1691660943"
       },
       "departure": {
        "time": "1691660943"
       },
       "stopId": "1223",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 23,
       "arrival": {
        "time": "1691661004"
       },
       "departure": {
        "time": "1691661004"
       },
       "stopId": "3887",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 24,
       "arrival": {
        "time": "1691661152"
       },
       "departure": {
        "time": "1691661152"
       },
       "stopId": "2524",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 25,
       "arrival": {
        "time": "1691661240"
       },
       "departure": {
        "time": "1691661240"
       },
       "stopId": "4073",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 26,
       "arrival": {
        "time": "1691661276"
       },
       "departure": {
        "time": "1691661276"
       },
       "stopId": "1916",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 27,
       "arrival": {
        "time": "1691661366"
       },
       "departure": {
        "time": "1691661366"
       },
       "stopId": "1918",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 28,
       "arrival": {
        "time": "1691661480"
       },
       "departure": {
        "time": "1691661480"
       },
       "stopId": "1127",
       "scheduleRelationship": "SCHEDULED"
      }
     ]
    }
  }
 ]
})"s;

constexpr auto const expected = R"(
   0: 2351    Block Line Station..............................                                                             d: 10.08 09:15 [10.08 05:15]  RT 10.08 09:15 [10.08 05:15]  [{name=Tram iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
   1: 1033    Block Line / Hanover............................ a: 10.08 09:16 [10.08 05:16]  RT 10.08 09:16 [10.08 05:16]  d: 10.08 09:16 [10.08 05:16]  RT 10.08 09:16 [10.08 05:16]  [{name=Tram iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
   2: 2086    Block Line / Kingswood.......................... a: 10.08 09:18 [10.08 05:18]  RT 10.08 09:18 [10.08 05:18]  d: 10.08 09:18 [10.08 05:18]  RT 10.08 09:18 [10.08 05:18]  [{name=Tram iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
   3: 2885    Block Line / Strasburg.......................... a: 10.08 09:19 [10.08 05:19]  RT 10.08 09:19 [10.08 05:19]  d: 10.08 09:19 [10.08 05:19]  RT 10.08 09:19 [10.08 05:19]  [{name=Tram iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
   4: 2888    Block Line / Laurentian......................... a: 10.08 09:21 [10.08 05:21]  RT 10.08 09:21 [10.08 05:21]  d: 10.08 09:21 [10.08 05:21]  RT 10.08 09:21 [10.08 05:21]  [{name=Tram iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
   5: 3189    Block Line / Westmount.......................... a: 10.08 09:22 [10.08 05:22]  RT 10.08 09:22 [10.08 05:22]  d: 10.08 09:22 [10.08 05:22]  RT 10.08 09:22 [10.08 05:22]  [{name=Tram iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
   6: 3895    Fischer-Hallman / Westmount..................... a: 10.08 09:24 [10.08 05:24]  RT 10.08 09:24 [10.08 05:24]  d: 10.08 09:24 [10.08 05:24]  RT 10.08 09:24 [10.08 05:24]  [{name=Tram iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
   7: 3893    Fischer-Hallman / Activa........................ a: 10.08 09:26 [10.08 05:26]  RT 10.08 09:26 [10.08 05:26]  d: 10.08 09:26 [10.08 05:26]  RT 10.08 09:26 [10.08 05:26]  [{name=Tram iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
   8: 2969    Fischer-Hallman / Ottawa........................ a: 10.08 09:27 [10.08 05:27]  RT 10.08 09:27 [10.08 05:27]  d: 10.08 09:27 [10.08 05:27]  RT 10.08 09:27 [10.08 05:27]  [{name=Tram iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
   9: 2971    Fischer-Hallman / Mcgarry....................... a: 10.08 09:29 [10.08 05:29]  RT 10.08 09:29 [10.08 05:29]  d: 10.08 09:29 [10.08 05:29]  RT 10.08 09:29 [10.08 05:29]  [{name=Tram iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  10: 2986    Fischer-Hallman / Queens........................ a: 10.08 09:31 [10.08 05:31]  RT 10.08 09:31 [10.08 05:31]  d: 10.08 09:31 [10.08 05:31]  RT 10.08 09:31 [10.08 05:31]  [{name=Tram iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  11: 3891    Fischer-Hallman / Highland...................... a: 10.08 09:32 [10.08 05:32]  RT 10.08 09:32 [10.08 05:32]  d: 10.08 09:32 [10.08 05:32]  RT 10.08 09:32 [10.08 05:32]  [{name=Tram iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  12: 3143    Fischer-Hallman / Victoria...................... a: 10.08 09:33 [10.08 05:33]  RT 10.08 09:33 [10.08 05:33]  d: 10.08 09:33 [10.08 05:33]  RT 10.08 09:33 [10.08 05:33]  [{name=Tram iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  13: 3144    Fischer-Hallman / Stoke......................... a: 10.08 09:35 [10.08 05:35]  RT 10.08 09:35 [10.08 05:35]  d: 10.08 09:35 [10.08 05:35]  RT 10.08 09:35 [10.08 05:35]  [{name=Tram iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  14: 3146    Fischer-Hallman / University Ave................ a: 10.08 09:37 [10.08 05:37]  RT 10.08 09:38 [10.08 05:38]  d: 10.08 09:37 [10.08 05:37]  RT 10.08 09:38 [10.08 05:38]  [{name=Tram iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  15: 1992    Fischer-Hallman / Thorndale..................... a: 10.08 09:38 [10.08 05:38]  RT 10.08 09:39 [10.08 05:39]  d: 10.08 09:38 [10.08 05:38]  RT 10.08 09:39 [10.08 05:39]  [{name=Tram iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  16: 1972    Fischer-Hallman / Erb........................... a: 10.08 09:39 [10.08 05:39]  RT 10.08 09:40 [10.08 05:40]  d: 10.08 09:39 [10.08 05:39]  RT 10.08 09:40 [10.08 05:40]  [{name=Tram iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  17: 3465    Fischer-Hallman / Keats Way..................... a: 10.08 09:40 [10.08 05:40]  RT 10.08 09:41 [10.08 05:41]  d: 10.08 09:40 [10.08 05:40]  RT 10.08 09:41 [10.08 05:41]  [{name=Tram iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  18: 3890    Fischer-Hallman / Columbia...................... a: 10.08 09:42 [10.08 05:42]  RT 10.08 09:44 [10.08 05:44]  d: 10.08 09:42 [10.08 05:42]  RT 10.08 09:44 [10.08 05:44]  [{name=Tram iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  19: 1117    Columbia / U.W. - Columbia Lake Village......... a: 10.08 09:43 [10.08 05:43]  RT 10.08 09:45 [10.08 05:45]  d: 10.08 09:43 [10.08 05:43]  RT 10.08 09:45 [10.08 05:45]  [{name=Tram iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  20: 3899    Columbia / University Of Waterloo............... a: 10.08 09:46 [10.08 05:46]  RT 10.08 09:47 [10.08 05:47]  d: 10.08 09:46 [10.08 05:46]  RT 10.08 09:47 [10.08 05:47]  [{name=Tram iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  21: 1223    University Of Waterloo Station.................. a: 10.08 09:47 [10.08 05:47]  RT 10.08 09:49 [10.08 05:49]  d: 10.08 09:49 [10.08 05:49]  RT 10.08 09:49 [10.08 05:49]  [{name=Tram iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  22: 3887    Phillip / Columbia.............................. a: 10.08 09:50 [10.08 05:50]  RT 10.08 09:50 [10.08 05:50]  d: 10.08 09:50 [10.08 05:50]  RT 10.08 09:50 [10.08 05:50]  [{name=Tram iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  23: 2524    Columbia / Hazel................................ a: 10.08 09:53 [10.08 05:53]  RT 10.08 09:52 [10.08 05:52]  d: 10.08 09:53 [10.08 05:53]  RT 10.08 09:52 [10.08 05:52]  [{name=Tram iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  24: 4073    King / Columbia................................. a: 10.08 09:54 [10.08 05:54]  RT 10.08 09:54 [10.08 05:54]  d: 10.08 09:54 [10.08 05:54]  RT 10.08 09:54 [10.08 05:54]  [{name=Tram iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  25: 1916    King / Weber.................................... a: 10.08 09:55 [10.08 05:55]  RT 10.08 09:54 [10.08 05:54]  d: 10.08 09:55 [10.08 05:55]  RT 10.08 09:54 [10.08 05:54]  [{name=Tram iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  26: 1918    King / Manulife................................. a: 10.08 09:56 [10.08 05:56]  RT 10.08 09:56 [10.08 05:56]  d: 10.08 09:56 [10.08 05:56]  RT 10.08 09:56 [10.08 05:56]  [{name=Tram iXpress Fischer-Hallman, day=2023-08-10, id=3248651, src=0}]
  27: 1127    Conestoga Station............................... a: 10.08 09:58 [10.08 05:58]  RT 10.08 09:58 [10.08 05:58]
)"sv;

}  // namespace

std::string json_to_protobuf(std::string const& json) {
  transit_realtime::FeedMessage msg;
  google::protobuf::util::JsonStringToMessage(json, &msg);
  return msg.SerializeAsString();
}

TEST(rt, gtfs_rt_update_1) {
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
  auto const msg = json_to_protobuf(kTripUpdate);
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg);

  // Print trip.
  transit_realtime::TripDescriptor td;
  td.set_start_date("20230810");
  td.set_trip_id("3248651");
  td.set_start_time("05:15:00");
  auto const [r, t] = rt::gtfsrt_resolve_run(date::sys_days{May / 1 / 2019}, tt,
                                             rtt, source_idx_t{0}, td);
  ASSERT_TRUE(r.valid());

  auto ss = std::stringstream{};
  ss << "\n" << rt::frun{tt, &rtt, r};
  EXPECT_EQ(expected, ss.str());
}