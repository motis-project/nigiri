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

#include <cstdlib>
#include <filesystem>
#include <fstream>

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
# agency.txt
agency_name,agency_url,agency_timezone,agency_lang,agency_phone,agency_fare_url,agency_id
"Agency",https://agency.com,UTC,en,555-0123,https://agency.com/fares,agency

# stops.txt
stop_id,stop_code,stop_name,stop_desc,stop_lat,stop_lon,zone_id,stop_url,location_type,parent_station,wheelchair_boarding,platform_code
S01,S01,Stop 1,,0.0,0.0,,,
S02,S02,Stop 2,,0.0,0.1,,,
S03,S03,Stop 3,,0.0,0.2,,,
S04,S04,Stop 4,,0.0,0.3,,,

# calendar_dates.txt
service_id,date,exception_type
S_DAILY,20260101,1
S_DAILY,20260108,1
S_DAILY,20260115,1
S_DAILY,20260122,1
S_DAILY,20260129,1
S_DAILY,20260205,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R01,agency,Route 1,,3,

# trips.txt
route_id,service_id,trip_id,trip_headsign,direction_id,block_id,shape_id,wheelchair_accessible,bikes_allowed
R01,S_DAILY,T01,Headsign,0,B1,SH1,1,1
R01,S_DAILY,T02,Headsign,0,B1,SH1,1,1
R01,S_DAILY,T03,Headsign,0,B1,SH1,1,1
R01,S_DAILY,T04,Headsign,0,B1,SH1,1,1
R01,S_DAILY,T05,Headsign,0,B1,SH1,1,1
R01,S_DAILY,T06,Headsign,0,B1,SH1,1,1
R01,S_DAILY,T07,Headsign,0,B1,SH1,1,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T01,10:00:00,10:00:00,S01,1,0,0
T01,11:00:00,12:00:00,S02,2,0,0
T01,13:00:00,14:00:00,S03,3,0,0
T01,15:00:00,15:00:00,S04,4,0,0
T02,10:00:00,10:00:00,S01,1,0,0
T02,11:00:00,12:00:00,S02,2,0,0
T02,13:00:00,14:00:00,S03,3,0,0
T02,15:00:00,15:00:00,S04,4,0,0
T03,10:00:00,10:00:00,S01,1,0,0
T03,11:00:00,12:00:00,S02,2,0,0
T03,13:00:00,14:00:00,S03,3,0,0
T03,15:00:00,15:00:00,S04,4,0,0
T04,10:00:00,10:00:00,S01,1,0,0
T04,11:00:00,12:00:00,S02,2,0,0
T04,13:00:00,14:00:00,S03,3,0,0
T04,15:00:00,15:00:00,S04,4,0,0
T05,10:00:00,10:00:00,S01,1,0,0
T05,11:00:00,12:00:00,S02,2,0,0
T05,13:00:00,14:00:00,S03,3,0,0
T05,15:00:00,15:00:00,S04,4,0,0
T06,08:00:00,08:00:00,S01,1,0,0
T06,09:00:00,10:00:00,S02,2,0,0
T06,11:00:00,12:00:00,S03,3,0,0
T06,13:00:00,13:00:00,S04,4,0,0
T07,10:00:00,10:00:00,S01,1,0,0
T07,11:00:00,12:00:00,S02,2,0,0
T07,13:00:00,14:00:00,S03,3,0,0
T07,15:00:00,15:00:00,S04,4,0,0
)");
}

auto const kVehiclePositionT01 = R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": 1767280140
 },
 "entity": [
  {
   "id": "T01_1",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T01",
     "startTime": "10:00:00",
     "startDate": "20260101",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.0
    },
    "timestamp": 1767261600,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T01_2",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T01",
     "startTime": "10:00:00",
     "startDate": "20260101",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.0
    },
    "timestamp": 1767261840,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T01_3",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T01",
     "startTime": "10:00:00",
     "startDate": "20260101",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.05
    },
    "timestamp": 1767263640,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T01_4",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T01",
     "startTime": "10:00:00",
     "startDate": "20260101",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.0999999
    },
    "timestamp": 1767265500,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T01_5",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T01",
     "startTime": "10:00:00",
     "startDate": "20260101",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.1000001
    },
    "timestamp": 1767269160,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T01_6",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T01",
     "startTime": "10:00:00",
     "startDate": "20260101",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.15
    },
    "timestamp": 1767270960,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T01_7",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T01",
     "startTime": "10:00:00",
     "startDate": "20260101",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.1999999
    },
    "timestamp": 1767272820,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T01_8",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T01",
     "startTime": "10:00:00",
     "startDate": "20260101",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.2000001
    },
    "timestamp": 1767276480,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T01_9",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T01",
     "startTime": "10:00:00",
     "startDate": "20260101",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.25
    },
    "timestamp": 1767278280,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T01_10",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T01",
     "startTime": "10:00:00",
     "startDate": "20260101",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.3
    },
    "timestamp": 1767280140,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  }
 ]
})"s;

auto const kVehiclePositionT02 = R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": 1767885300
 },
 "entity": [
  {
   "id": "T02_1",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T02",
     "startTime": "10:00:00",
     "startDate": "20260108",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.0
    },
    "timestamp": 1767866400,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T02_2",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T02",
     "startTime": "10:00:00",
     "startDate": "20260108",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.0
    },
    "timestamp": 1767867000,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T02_3",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T02",
     "startTime": "10:00:00",
     "startDate": "20260108",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.05
    },
    "timestamp": 1767868800,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T02_4",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T02",
     "startTime": "10:00:00",
     "startDate": "20260108",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.0999999
    },
    "timestamp": 1767870660,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T02_5",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T02",
     "startTime": "10:00:00",
     "startDate": "20260108",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.1000001
    },
    "timestamp": 1767874260,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T02_6",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T02",
     "startTime": "10:00:00",
     "startDate": "20260108",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.15
    },
    "timestamp": 1767876120,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T02_7",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T02",
     "startTime": "10:00:00",
     "startDate": "20260108",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.1999999
    },
    "timestamp": 1767877920,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T02_8",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T02",
     "startTime": "10:00:00",
     "startDate": "20260108",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.2000001
    },
    "timestamp": 1767881580,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T02_9",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T02",
     "startTime": "10:00:00",
     "startDate": "20260108",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.25
    },
    "timestamp": 1767883380,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T02_10",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T02",
     "startTime": "10:00:00",
     "startDate": "20260108",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.3
    },
    "timestamp": 1767885300,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  }
 ]
})"s;

auto const kVehiclePositionT03 = R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": 1768490460
 },
 "entity": [
  {
   "id": "T03_1",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T03",
     "startTime": "10:00:00",
     "startDate": "20260115",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.0
    },
    "timestamp": 1768471200,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T03_2",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T03",
     "startTime": "10:00:00",
     "startDate": "20260115",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.0
    },
    "timestamp": 1768471200,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T03_3",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T03",
     "startTime": "10:00:00",
     "startDate": "20260115",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.05
    },
    "timestamp": 1768473000,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T03_4",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T03",
     "startTime": "10:00:00",
     "startDate": "20260115",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.0999999
    },
    "timestamp": 1768474860,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T03_5",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T03",
     "startTime": "10:00:00",
     "startDate": "20260115",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.1000001
    },
    "timestamp": 1768478460,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T03_6",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T03",
     "startTime": "10:00:00",
     "startDate": "20260115",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.15
    },
    "timestamp": 1768480320,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T03_7",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T03",
     "startTime": "10:00:00",
     "startDate": "20260115",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.1999999
    },
    "timestamp": 1768482120,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T03_8",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T03",
     "startTime": "10:00:00",
     "startDate": "20260115",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.2000001
    },
    "timestamp": 1768486800,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T03_9",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T03",
     "startTime": "10:00:00",
     "startDate": "20260115",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.25
    },
    "timestamp": 1768488600,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T03_10",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T03",
     "startTime": "10:00:00",
     "startDate": "20260115",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.3
    },
    "timestamp": 1768490460,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  }
 ]
})"s;

auto const kVehiclePositionT04 = R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": 1769095020
 },
 "entity": [
  {
   "id": "T04_1",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T04",
     "startTime": "10:00:00",
     "startDate": "20260122",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.0
    },
    "timestamp": 1769076000,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T04_2",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T04",
     "startTime": "10:00:00",
     "startDate": "20260122",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.0
    },
    "timestamp": 1769076060,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T04_3",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T04",
     "startTime": "10:00:00",
     "startDate": "20260122",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.05
    },
    "timestamp": 1769077860,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T04_4",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T04",
     "startTime": "10:00:00",
     "startDate": "20260122",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.0999999
    },
    "timestamp": 1769079720,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T04_5",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T04",
     "startTime": "10:00:00",
     "startDate": "20260122",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.1000001
    },
    "timestamp": 1769083320,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T04_6",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T04",
     "startTime": "10:00:00",
     "startDate": "20260122",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.15
    },
    "timestamp": 1769085180,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T04_7",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T04",
     "startTime": "10:00:00",
     "startDate": "20260122",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.1999999
    },
    "timestamp": 1769086980,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T04_8",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T04",
     "startTime": "10:00:00",
     "startDate": "20260122",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.2000001
    },
    "timestamp": 1769091300,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T04_9",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T04",
     "startTime": "10:00:00",
     "startDate": "20260122",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.25
    },
    "timestamp": 1769093160,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T04_10",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T04",
     "startTime": "10:00:00",
     "startDate": "20260122",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.3
    },
    "timestamp": 1769095020,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  }
 ]
})"s;

auto const kVehiclePositionT05 = R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": 1769698980
 },
 "entity": [
  {
   "id": "T05_1",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T05",
     "startTime": "10:00:00",
     "startDate": "20260129",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.0
    },
    "timestamp": 1769680800,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T05_2",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T05",
     "startTime": "10:00:00",
     "startDate": "20260129",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.0
    },
    "timestamp": 1769680920,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T05_3",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T05",
     "startTime": "10:00:00",
     "startDate": "20260129",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.05
    },
    "timestamp": 1769682720,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T05_4",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T05",
     "startTime": "10:00:00",
     "startDate": "20260129",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.0999999
    },
    "timestamp": 1769684580,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T05_5",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T05",
     "startTime": "10:00:00",
     "startDate": "20260129",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.1000001
    },
    "timestamp": 1769688180,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T05_6",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T05",
     "startTime": "10:00:00",
     "startDate": "20260129",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.15
    },
    "timestamp": 1769690040,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T05_7",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T05",
     "startTime": "10:00:00",
     "startDate": "20260129",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.1999999
    },
    "timestamp": 1769691840,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T05_8",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T05",
     "startTime": "10:00:00",
     "startDate": "20260129",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.2000001
    },
    "timestamp": 1769695380,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T05_9",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T05",
     "startTime": "10:00:00",
     "startDate": "20260129",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.25
    },
    "timestamp": 1769697180,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T05_10",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T05",
     "startTime": "10:00:00",
     "startDate": "20260129",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.3
    },
    "timestamp": 1769698980,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  }
 ]
})"s;

auto const kVehiclePositionT06 = R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": 1770296940
 },
 "entity": [
  {
   "id": "T06_1",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T06",
     "startTime": "08:00:00",
     "startDate": "20260205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.0
    },
    "timestamp": 1770278400,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T06_2",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T06",
     "startTime": "08:00:00",
     "startDate": "20260205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.0
    },
    "timestamp": 1770278700,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T06_3",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T06",
     "startTime": "08:00:00",
     "startDate": "20260205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.05
    },
    "timestamp": 1770280500,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T06_4",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T06",
     "startTime": "08:00:00",
     "startDate": "20260205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.0999999
    },
    "timestamp": 1770282360,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T06_5",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T06",
     "startTime": "08:00:00",
     "startDate": "20260205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.1000001
    },
    "timestamp": 1770285960,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T06_6",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T06",
     "startTime": "08:00:00",
     "startDate": "20260205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.15
    },
    "timestamp": 1770287820,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T06_7",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T06",
     "startTime": "08:00:00",
     "startDate": "20260205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.1999999
    },
    "timestamp": 1770289620,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T06_8",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T06",
     "startTime": "08:00:00",
     "startDate": "20260205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.2000001
    },
    "timestamp": 1770293280,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T06_9",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T06",
     "startTime": "08:00:00",
     "startDate": "20260205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.25
    },
    "timestamp": 1770295080,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T06_10",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T06",
     "startTime": "08:00:00",
     "startDate": "20260205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.3
    },
    "timestamp": 1770296940,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  }
 ]
})"s;

auto const kVehiclePositionT07_1 = R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": 1770285600
 },
 "entity": [
  {
   "id": "T07_1",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T07",
     "startTime": "10:00:00",
     "startDate": "20260205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.0
    },
    "timestamp": 1770285600,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  }
 ]
})"s;
auto const kVehiclePositionT07_2 = R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": 1770286800
 },
 "entity": [
  {
   "id": "T07_2",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T07",
     "startTime": "10:00:00",
     "startDate": "20260205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.0
    },
    "timestamp": 1770286800,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  }
 ]
})"s;
auto const kVehiclePositionT07_3 = R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": 1770288600
 },
 "entity": [
  {
   "id": "T07_3",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T07",
     "startTime": "10:00:00",
     "startDate": "20260205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.05
    },
    "timestamp": 1770288600,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  }
 ]
})"s;
auto const kVehiclePositionT07_4 = R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": 1770290460
 },
 "entity": [
  {
   "id": "T07_4",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T07",
     "startTime": "10:00:00",
     "startDate": "20260205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.0999999
    },
    "timestamp": 1770290460,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  }
 ]
})"s;
auto const kVehiclePositionT07_5 = R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": 1770294060
 },
 "entity": [
  {
   "id": "T07_5",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T07",
     "startTime": "10:00:00",
     "startDate": "20260205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.1000001
    },
    "timestamp": 1770294060,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  }
 ]
})"s;
auto const kVehiclePositionT07_6 = R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": 1770295920
 },
 "entity": [
  {
   "id": "T07_6",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T07",
     "startTime": "10:00:00",
     "startDate": "20260205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.15
    },
    "timestamp": 1770295920,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  }
 ]
})"s;
auto const kVehiclePositionT07_7 = R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": 1770297720
 },
 "entity": [
  {
   "id": "T07_7",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T07",
     "startTime": "10:00:00",
     "startDate": "20260205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.1999999
    },
    "timestamp": 1770297720,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  }
 ]
})"s;
auto const kVehiclePositionT07_8 = R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": 1770301500
 },
 "entity": [
  {
   "id": "T07_8",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T07",
     "startTime": "10:00:00",
     "startDate": "20260205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.2000001
    },
    "timestamp": 1770301500,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  }
 ]
})"s;
auto const kVehiclePositionT07_9 = R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": 1770303300
 },
 "entity": [
  {
   "id": "T07_9",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T07",
     "startTime": "10:00:00",
     "startDate": "20260205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.25
    },
    "timestamp": 1770303300,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  }
 ]
})"s;
auto const kVehiclePositionT07_10 = R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": 1770305160
 },
 "entity": [
  {
   "id": "T07_10",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T07",
     "startTime": "10:00:00",
     "startDate": "20260205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.3
    },
    "timestamp": 1770305160,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  }
 ]
})"s;

}  // namespace

TEST(rt, gtfsrt_rt_delay_calc) {

  // Load static timetable.
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2026_y / January / 1},
                    date::sys_days{2026_y / February / 6}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2026_y / February / 5});

  auto tts = hist_trip_times_storage{};
  auto vtm = vehicle_trip_matching{};
  auto dps = delay_prediction_storage{};

  auto dp = delay_prediction{algorithm::kIntelligent,
                             hist_trip_mode::kSameDay,
                             1,
                             5,
                             &dps,
                             &tts,
                             &vtm,
                             true};

  // Historic updates (inject previous days)
  auto const msg01 = rt::json_to_protobuf(kVehiclePositionT01);
  auto const msg02 = rt::json_to_protobuf(kVehiclePositionT02);
  auto const msg03 = rt::json_to_protobuf(kVehiclePositionT03);
  auto const msg04 = rt::json_to_protobuf(kVehiclePositionT04);
  auto const msg05 = rt::json_to_protobuf(kVehiclePositionT05);
  auto const msg06 = rt::json_to_protobuf(kVehiclePositionT06);

  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg01, &dp);
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg02, &dp);
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg03, &dp);
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg04, &dp);
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg05, &dp);
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg06, &dp);

  transit_realtime::TripDescriptor td01;
  td01.set_start_date("20260101");
  td01.set_trip_id("T01");
  td01.set_start_time("10:00:00");
  auto const [r01, t01] = gtfsrt_resolve_run(date::sys_days{January / 1 / 2026},
                                             tt, &rtt, source_idx_t{0}, td01);

  transit_realtime::TripDescriptor td02;
  td02.set_start_date("20260108");
  td02.set_trip_id("T02");
  td02.set_start_time("10:00:00");
  auto const [r02, t02] = gtfsrt_resolve_run(date::sys_days{January / 1 / 2026},
                                             tt, &rtt, source_idx_t{0}, td02);

  transit_realtime::TripDescriptor td03;
  td03.set_start_date("20260115");
  td03.set_trip_id("T03");
  td03.set_start_time("10:00:00");
  auto const [r03, t03] = gtfsrt_resolve_run(date::sys_days{January / 1 / 2026},
                                             tt, &rtt, source_idx_t{0}, td03);

  transit_realtime::TripDescriptor td04;
  td04.set_start_date("20260122");
  td04.set_trip_id("T04");
  td04.set_start_time("10:00:00");
  auto const [r04, t04] = gtfsrt_resolve_run(date::sys_days{January / 1 / 2026},
                                             tt, &rtt, source_idx_t{0}, td04);

  transit_realtime::TripDescriptor td05;
  td05.set_start_date("20260129");
  td05.set_trip_id("T05");
  td05.set_start_time("10:00:00");
  auto const [r05, t05] = gtfsrt_resolve_run(date::sys_days{January / 1 / 2026},
                                             tt, &rtt, source_idx_t{0}, td05);

  transit_realtime::TripDescriptor td06;
  td06.set_start_date("20260205");
  td06.set_trip_id("T06");
  td06.set_start_time("08:00:00");
  auto const [r06, t06] = gtfsrt_resolve_run(date::sys_days{January / 1 / 2026},
                                             tt, &rtt, source_idx_t{0}, td06);

  ASSERT_TRUE(r01.valid());
  ASSERT_TRUE(r02.valid());
  ASSERT_TRUE(r03.valid());
  ASSERT_TRUE(r04.valid());
  ASSERT_TRUE(r05.valid());
  ASSERT_TRUE(r06.valid());

  // Live updates for T07
  auto const msg07_1 = rt::json_to_protobuf(kVehiclePositionT07_1);
  auto const msg07_2 = rt::json_to_protobuf(kVehiclePositionT07_2);
  auto const msg07_2b = rt::json_to_protobuf(kVehiclePositionT07_3);
  auto const msg07_3 = rt::json_to_protobuf(kVehiclePositionT07_4);
  auto const msg07_4 = rt::json_to_protobuf(kVehiclePositionT07_5);
  auto const msg07_4b = rt::json_to_protobuf(kVehiclePositionT07_6);
  auto const msg07_5 = rt::json_to_protobuf(kVehiclePositionT07_7);
  auto const msg07_6 = rt::json_to_protobuf(kVehiclePositionT07_8);
  auto const msg07_6b = rt::json_to_protobuf(kVehiclePositionT07_9);
  auto const msg07_7 = rt::json_to_protobuf(kVehiclePositionT07_10);

  transit_realtime::TripDescriptor td07;
  td07.set_start_date("20260205");
  td07.set_trip_id("T07");
  td07.set_start_time("10:00:00");
  auto const [r07, t07] = gtfsrt_resolve_run(date::sys_days{January / 1 / 2026},
                                             tt, &rtt, source_idx_t{0}, td07);

  ASSERT_TRUE(r07.valid());

  std::cout << "--- Live Update 1 ---" << std::endl;
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg07_1, &dp);
  std::cout << rt::frun{tt, &rtt, r07} << std::endl;

  std::cout << "--- Live Update 2 ---" << std::endl;
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg07_2, &dp);
  std::cout << rt::frun{tt, &rtt, r07} << std::endl;

  std::cout << "--- Live Update 3 ---" << std::endl;
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg07_2b, &dp);
  std::cout << rt::frun{tt, &rtt, r07} << std::endl;

  std::cout << "--- Live Update 4 ---" << std::endl;
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg07_3, &dp);
  std::cout << rt::frun{tt, &rtt, r07} << std::endl;

  std::cout << "--- Live Update 5 ---" << std::endl;
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg07_4, &dp);
  std::cout << rt::frun{tt, &rtt, r07} << std::endl;

  std::cout << "--- Live Update 6 ---" << std::endl;
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg07_4b, &dp);
  std::cout << rt::frun{tt, &rtt, r07} << std::endl;

  std::cout << "--- Live Update 7 ---" << std::endl;
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg07_5, &dp);
  std::cout << rt::frun{tt, &rtt, r07} << std::endl;

  std::cout << "--- Live Update 8 ---" << std::endl;
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg07_6, &dp);
  std::cout << rt::frun{tt, &rtt, r07} << std::endl;

  std::cout << "--- Live Update 9 ---" << std::endl;
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg07_6b, &dp);
  std::cout << rt::frun{tt, &rtt, r07} << std::endl;

  std::cout << "--- Live Update 10 ---" << std::endl;
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg07_7, &dp);
  std::cout << rt::frun{tt, &rtt, r07} << std::endl;

  std::stringstream ss_tts;
  ss_tts << tts;
  std::cout << ss_tts.str() << std::endl;

  std::stringstream ss_dps;
  ss_dps << dps;
  std::cout << ss_dps.str() << std::endl;

  tts.dump_delays(tt, rtt);

  // simple

  // Load static timetable.
  timetable tt2;
  register_special_stations(tt2);
  tt2.date_range_ = {date::sys_days{2026_y / January / 1},
                    date::sys_days{2026_y / February / 6}};
  load_timetable({}, source_idx_t{0}, test_files(), tt2);
  finalize(tt2);

  auto rtt2 = rt::create_rt_timetable(tt2, date::sys_days{2026_y / February / 5});

  auto tts2 = hist_trip_times_storage{};
  auto vtm2 = vehicle_trip_matching{};
  auto dps2 = delay_prediction_storage{};

  auto dp2 = delay_prediction{algorithm::kSimple,
                             hist_trip_mode::kSameDay,
                             1,
                             5,
                             &dps2,
                             &tts2,
                             &vtm2,
                             true};

  // Historic updates (inject previous days)
  gtfsrt_update_buf(tt2, rtt2, source_idx_t{0}, "", msg01, &dp2);
  gtfsrt_update_buf(tt2, rtt2, source_idx_t{0}, "", msg02, &dp2);
  gtfsrt_update_buf(tt2, rtt2, source_idx_t{0}, "", msg03, &dp2);
  gtfsrt_update_buf(tt2, rtt2, source_idx_t{0}, "", msg04, &dp2);
  gtfsrt_update_buf(tt2, rtt2, source_idx_t{0}, "", msg05, &dp2);
  gtfsrt_update_buf(tt2, rtt2, source_idx_t{0}, "", msg06, &dp2);

  // Live Updates
  gtfsrt_update_buf(tt2, rtt2, source_idx_t{0}, "", msg07_1, &dp2);
  gtfsrt_update_buf(tt2, rtt2, source_idx_t{0}, "", msg07_2, &dp2);
  gtfsrt_update_buf(tt2, rtt2, source_idx_t{0}, "", msg07_2b, &dp2);
  gtfsrt_update_buf(tt2, rtt2, source_idx_t{0}, "", msg07_3, &dp2);
  gtfsrt_update_buf(tt2, rtt2, source_idx_t{0}, "", msg07_4, &dp2);
  gtfsrt_update_buf(tt2, rtt2, source_idx_t{0}, "", msg07_4b, &dp2);
  gtfsrt_update_buf(tt2, rtt2, source_idx_t{0}, "", msg07_5, &dp2);
  gtfsrt_update_buf(tt2, rtt2, source_idx_t{0}, "", msg07_6, &dp2);
  gtfsrt_update_buf(tt2, rtt2, source_idx_t{0}, "", msg07_6b, &dp2);
  gtfsrt_update_buf(tt2, rtt2, source_idx_t{0}, "", msg07_7, &dp2);
}
