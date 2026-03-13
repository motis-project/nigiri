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
"Agency",https://agency.com,UTC,en,555-0123,http://agency.com/fares,agency

# stops.txt
stop_id,stop_code,stop_name,stop_desc,stop_lat,stop_lon,zone_id,stop_url,location_type,parent_station,wheelchair_boarding,platform_code
S01,S01,Stop 1,,0.0,0.0,,,
S02,S02,Stop 2,,0.0,0.1,,,
S03,S03,Stop 3,,0.0,0.2,,,
S04,S04,Stop 4,,0.0,0.3,,,

# calendar_dates.txt
service_id,date,exception_type
S_DAILY,20240101,1
S_DAILY,20240108,1
S_DAILY,20240115,1
S_DAILY,20240122,1
S_DAILY,20240129,1
S_DAILY,20240205,1

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
T01,10:30:00,10:30:00,S02,2,0,0
T01,11:00:00,11:00:00,S03,3,0,0
T01,11:30:00,11:30:00,S04,4,0,0
T02,10:00:00,10:00:00,S01,1,0,0
T02,10:30:00,10:30:00,S02,2,0,0
T02,11:00:00,11:00:00,S03,3,0,0
T02,11:30:00,11:30:00,S04,4,0,0
T03,10:00:00,10:00:00,S01,1,0,0
T03,10:30:00,10:30:00,S02,2,0,0
T03,11:00:00,11:00:00,S03,3,0,0
T03,11:30:00,11:30:00,S04,4,0,0
T04,10:00:00,10:00:00,S01,1,0,0
T04,10:30:00,10:30:00,S02,2,0,0
T04,11:00:00,11:00:00,S03,3,0,0
T04,11:30:00,11:30:00,S04,4,0,0
T05,10:00:00,10:00:00,S01,1,0,0
T05,10:30:00,10:30:00,S02,2,0,0
T05,11:00:00,11:00:00,S03,3,0,0
T05,11:30:00,11:30:00,S04,4,0,0
T06,06:00:00,06:00:00,S01,1,0,0
T06,06:30:00,06:30:00,S02,2,0,0
T06,07:00:00,07:00:00,S03,3,0,0
T06,07:30:00,07:30:00,S04,4,0,0
T07,10:00:00,10:00:00,S01,1,0,0
T07,10:30:00,10:30:00,S02,2,0,0
T07,11:00:00,11:00:00,S03,3,0,0
T07,11:30:00,11:30:00,S04,4,0,0
)");
}

// Day epochs (UTC midnight):
//   2024-01-01 = 1704067200
//   2024-01-08 = 1704672000
//   2024-01-15 = 1705276800
//   2024-01-22 = 1705881600
//   2024-01-29 = 1706486400
//   2024-02-05 = 1707091200
//
// Scheduled positions (lon) → approximate scheduled time-of-day:
//   0.01 → ~sched+03min  (near S01)
//   0.05 → ~sched+15min  (between S01-S02)
//   0.11 → ~sched+33min  (near S02)
//   0.15 → ~sched+45min  (between S02-S03)
//   0.21 → ~sched+63min  (near S03)
//   0.25 → ~sched+75min  (between S03-S04)
//   0.29 → ~sched+87min  (near S04)

// T01: 2024-01-01, start 10:00, delay +3/+4/+5/+5/+6/+7/+8 min
//   10:06, 10:19, 10:38, 10:50, 11:09, 11:22, 11:35
auto const kVehiclePositionT01 = R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": 1704108900
 },
 "entity": [
  {
   "id": "T01_1",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T01",
     "startTime": "10:00:00",
     "startDate": "20240101",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.01
    },
    "timestamp": 1704103560,
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
     "startDate": "20240101",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.05
    },
    "timestamp": 1704104340,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T01_before_S02",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T01",
     "startTime": "10:00:00",
     "startDate": "20240101",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.09
    },
    "timestamp": 1704105120,
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
     "startDate": "20240101",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.11
    },
    "timestamp": 1704105480,
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
     "startDate": "20240101",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.15
    },
    "timestamp": 1704106200,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T01_before_S03",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T01",
     "startTime": "10:00:00",
     "startDate": "20240101",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.19
    },
    "timestamp": 1704106980,
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
     "startDate": "20240101",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.21
    },
    "timestamp": 1704107340,
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
     "startDate": "20240101",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.25
    },
    "timestamp": 1704108120,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T01_before_S04",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T01",
     "startTime": "10:00:00",
     "startDate": "20240101",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.28
    },
    "timestamp": 1704108720,
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
     "startDate": "20240101",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.29
    },
    "timestamp": 1704108900,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  }
 ]
})"s;
// T02: 2024-01-08, start 10:00, delay +5/+6/+7/+8/+9/+10/+10 min
//   10:08, 10:21, 10:40, 10:53, 11:12, 11:25, 11:37
auto const kVehiclePositionT02 = R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": 1704713820
 },
 "entity": [
  {
   "id": "T02_1",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T02",
     "startTime": "10:00:00",
     "startDate": "20240108",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.01
    },
    "timestamp": 1704708480,
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
     "startDate": "20240108",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.05
    },
    "timestamp": 1704709260,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T02_before_S02",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T02",
     "startTime": "10:00:00",
     "startDate": "20240108",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.09
    },
    "timestamp": 1704710040,
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
     "startDate": "20240108",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.11
    },
    "timestamp": 1704710400,
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
     "startDate": "20240108",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.15
    },
    "timestamp": 1704711180,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T02_before_S03",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T02",
     "startTime": "10:00:00",
     "startDate": "20240108",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.19
    },
    "timestamp": 1704711960,
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
     "startDate": "20240108",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.21
    },
    "timestamp": 1704712320,
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
     "startDate": "20240108",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.25
    },
    "timestamp": 1704713100,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T02_before_S04",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T02",
     "startTime": "10:00:00",
     "startDate": "20240108",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.28
    },
    "timestamp": 1704713640,
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
     "startDate": "20240108",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.29
    },
    "timestamp": 1704713820,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  }
 ]
})"s;
// T03: 2024-01-15, start 10:00, delay +2/+3/+4/+6/+7/+8/+9 min
//   10:05, 10:18, 10:37, 10:51, 11:10, 11:23, 11:36
auto const kVehiclePositionT03 = R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": 1705318560
 },
 "entity": [
  {
   "id": "T03_1",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T03",
     "startTime": "10:00:00",
     "startDate": "20240115",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.01
    },
    "timestamp": 1705313100,
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
     "startDate": "20240115",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.05
    },
    "timestamp": 1705313880,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T03_before_S02",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T03",
     "startTime": "10:00:00",
     "startDate": "20240115",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.09
    },
    "timestamp": 1705314660,
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
     "startDate": "20240115",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.11
    },
    "timestamp": 1705315020,
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
     "startDate": "20240115",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.15
    },
    "timestamp": 1705315860,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T03_before_S03",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T03",
     "startTime": "10:00:00",
     "startDate": "20240115",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.19
    },
    "timestamp": 1705316640,
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
     "startDate": "20240115",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.21
    },
    "timestamp": 1705317000,
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
     "startDate": "20240115",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.25
    },
    "timestamp": 1705317780,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T03_before_S04",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T03",
     "startTime": "10:00:00",
     "startDate": "20240115",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.28
    },
    "timestamp": 1705318380,
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
     "startDate": "20240115",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.29
    },
    "timestamp": 1705318560,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  }
 ]
})"s;
// T04: 2024-01-22, start 10:00, delay +7/+8/+8/+9/+10/+11/+12 min
//   10:10, 10:23, 10:41, 10:54, 11:13, 11:26, 11:39
auto const kVehiclePositionT04 = R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": 1705923540
 },
 "entity": [
  {
   "id": "T04_1",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T04",
     "startTime": "10:00:00",
     "startDate": "20240122",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.01
    },
    "timestamp": 1705918200,
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
     "startDate": "20240122",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.05
    },
    "timestamp": 1705918980,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T04_before_S02",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T04",
     "startTime": "10:00:00",
     "startDate": "20240122",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.09
    },
    "timestamp": 1705919700,
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
     "startDate": "20240122",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.11
    },
    "timestamp": 1705920060,
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
     "startDate": "20240122",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.15
    },
    "timestamp": 1705920840,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T04_before_S03",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T04",
     "startTime": "10:00:00",
     "startDate": "20240122",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.19
    },
    "timestamp": 1705921620,
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
     "startDate": "20240122",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.21
    },
    "timestamp": 1705921980,
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
     "startDate": "20240122",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.25
    },
    "timestamp": 1705922760,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T04_before_S04",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T04",
     "startTime": "10:00:00",
     "startDate": "20240122",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.28
    },
    "timestamp": 1705923360,
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
     "startDate": "20240122",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.29
    },
    "timestamp": 1705923540,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  }
 ]
})"s;
// T05: 2024-01-29, start 10:00, delay +4/+5/+6/+7/+8/+8/+9 min
//   10:07, 10:20, 10:39, 10:52, 11:11, 11:23, 11:36
auto const kVehiclePositionT05 = R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": 1706528160
 },
 "entity": [
  {
   "id": "T05_1",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T05",
     "startTime": "10:00:00",
     "startDate": "20240129",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.01
    },
    "timestamp": 1706522820,
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
     "startDate": "20240129",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.05
    },
    "timestamp": 1706523600,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T05_before_S02",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T05",
     "startTime": "10:00:00",
     "startDate": "20240129",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.09
    },
    "timestamp": 1706524380,
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
     "startDate": "20240129",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.11
    },
    "timestamp": 1706524740,
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
     "startDate": "20240129",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.15
    },
    "timestamp": 1706525520,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T05_before_S03",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T05",
     "startTime": "10:00:00",
     "startDate": "20240129",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.19
    },
    "timestamp": 1706526300,
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
     "startDate": "20240129",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.21
    },
    "timestamp": 1706526660,
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
     "startDate": "20240129",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.25
    },
    "timestamp": 1706527380,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T05_before_S04",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T05",
     "startTime": "10:00:00",
     "startDate": "20240129",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.28
    },
    "timestamp": 1706527980,
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
     "startDate": "20240129",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.29
    },
    "timestamp": 1706528160,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  }
 ]
})"s;
// T06: 2024-02-05, start 06:00, delay +6/+7/+8/+9/+10/+11/+12 min
//   06:09, 06:22, 06:41, 06:54, 07:13, 07:26, 07:39
auto const kVehiclePositionT06 = R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": 1707118740
 },
 "entity": [
  {
   "id": "T06_1",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T06",
     "startTime": "06:00:00",
     "startDate": "20240205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.01
    },
    "timestamp": 1707113340,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T06_2",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T06",
     "startTime": "06:00:00",
     "startDate": "20240205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.05
    },
    "timestamp": 1707114120,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T06_before_S02",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T06",
     "startTime": "06:00:00",
     "startDate": "20240205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.09
    },
    "timestamp": 1707114900,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T06_3",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T06",
     "startTime": "06:00:00",
     "startDate": "20240205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.11
    },
    "timestamp": 1707115260,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T06_4",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T06",
     "startTime": "06:00:00",
     "startDate": "20240205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.15
    },
    "timestamp": 1707116040,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T06_before_S03",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T06",
     "startTime": "06:00:00",
     "startDate": "20240205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.19
    },
    "timestamp": 1707116820,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T06_5",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T06",
     "startTime": "06:00:00",
     "startDate": "20240205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.21
    },
    "timestamp": 1707117180,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T06_6",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T06",
     "startTime": "06:00:00",
     "startDate": "20240205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.25
    },
    "timestamp": 1707117960,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T06_before_S04",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T06",
     "startTime": "06:00:00",
     "startDate": "20240205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.28
    },
    "timestamp": 1707118560,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  },
  {
   "id": "T06_7",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T06",
     "startTime": "06:00:00",
     "startDate": "20240205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.29
    },
    "timestamp": 1707118740,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  }
 ]
})"s;
// T07: 2024-02-05, start 10:00, delay +5/+10/+12/+15/+17/+18/+20 min
//   10:08, 10:25, 10:45, 11:00, 11:20, 11:33, 11:47
auto const kVehiclePositionT07_1 = R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": 1707127680
 },
 "entity": [
  {
   "id": "T07_1",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T07",
     "startTime": "10:00:00",
     "startDate": "20240205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.01
    },
    "timestamp": 1707127680,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  }
 ]
})"s;
auto const kVehiclePositionT07_2 = R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": 1707128700
 },
 "entity": [
  {
   "id": "T07_2",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T07",
     "startTime": "10:00:00",
     "startDate": "20240205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.05
    },
    "timestamp": 1707128700,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  }
 ]
})"s;
auto const kVehiclePositionT07_before_S02 = R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": 1707129540
 },
 "entity": [
  {
   "id": "T07_before_S02",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T07",
     "startTime": "10:00:00",
     "startDate": "20240205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.09
    },
    "timestamp": 1707129540,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  }
 ]
})"s;
auto const kVehiclePositionT07_3 = R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": 1707129900
 },
 "entity": [
  {
   "id": "T07_3",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T07",
     "startTime": "10:00:00",
     "startDate": "20240205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.11
    },
    "timestamp": 1707129900,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  }
 ]
})"s;
auto const kVehiclePositionT07_4 = R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": 1707130800
 },
 "entity": [
  {
   "id": "T07_4",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T07",
     "startTime": "10:00:00",
     "startDate": "20240205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.15
    },
    "timestamp": 1707130800,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  }
 ]
})"s;
auto const kVehiclePositionT07_before_S03 = R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": 1707131640
 },
 "entity": [
  {
   "id": "T07_before_S03",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T07",
     "startTime": "10:00:00",
     "startDate": "20240205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.19
    },
    "timestamp": 1707131640,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  }
 ]
})"s;
auto const kVehiclePositionT07_5 = R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": 1707132000
 },
 "entity": [
  {
   "id": "T07_5",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T07",
     "startTime": "10:00:00",
     "startDate": "20240205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.21
    },
    "timestamp": 1707132000,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  }
 ]
})"s;
auto const kVehiclePositionT07_6 = R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": 1707132780
 },
 "entity": [
  {
   "id": "T07_6",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T07",
     "startTime": "10:00:00",
     "startDate": "20240205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.25
    },
    "timestamp": 1707132780,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  }
 ]
})"s;
auto const kVehiclePositionT07_before_S04 = R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": 1707133440
 },
 "entity": [
  {
   "id": "T07_before_S04",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T07",
     "startTime": "10:00:00",
     "startDate": "20240205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.28
    },
    "timestamp": 1707133440,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  }
 ]
})"s;
auto const kVehiclePositionT07_7 = R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": 1707133620
 },
 "entity": [
  {
   "id": "T07_7",
   "isDeleted": false,
   "vehicle": {
    "trip": {
     "tripId": "T07",
     "startTime": "10:00:00",
     "startDate": "20240205",
     "routeId": "R01"
    },
    "position": {
     "latitude": 0.0,
     "longitude": 0.29
    },
    "timestamp": 1707133620,
    "occupancy_status": "MANY_SEATS_AVAILABLE"
   }
  }
 ]
})"s;

}  // namespace

TEST(rt, gtfsrt_rt_delay_calc) {
  std::cout << "Test rt::gtfsrt_rt_delay_calc" << std::endl;

  // Load static timetable.
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2024_y / January / 1},
                    date::sys_days{2024_y / February / 6}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / February / 5});

  auto tts = hist_trip_times_storage{};
  auto vtm = vehicle_trip_matching{};
  auto dps = delay_prediction_storage{};

  auto dp = delay_prediction{algorithm::kIntelligent,
                             hist_trip_mode::kSameDay,
                             1,
                             5,
                             &dps,
                             &tts,
                             &vtm};

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
  td01.set_start_date("20240101");
  td01.set_trip_id("T01");
  td01.set_start_time("10:00:00");
  auto const [r01, t01] = gtfsrt_resolve_run(date::sys_days{January / 1 / 2024},
                                             tt, &rtt, source_idx_t{0}, td01);

  transit_realtime::TripDescriptor td02;
  td02.set_start_date("20240108");
  td02.set_trip_id("T02");
  td02.set_start_time("10:00:00");
  auto const [r02, t02] = gtfsrt_resolve_run(date::sys_days{January / 1 / 2024},
                                             tt, &rtt, source_idx_t{0}, td02);

  transit_realtime::TripDescriptor td03;
  td03.set_start_date("20240115");
  td03.set_trip_id("T03");
  td03.set_start_time("10:00:00");
  auto const [r03, t03] = gtfsrt_resolve_run(date::sys_days{January / 1 / 2024},
                                             tt, &rtt, source_idx_t{0}, td03);

  transit_realtime::TripDescriptor td04;
  td04.set_start_date("20240122");
  td04.set_trip_id("T04");
  td04.set_start_time("10:00:00");
  auto const [r04, t04] = gtfsrt_resolve_run(date::sys_days{January / 1 / 2024},
                                             tt, &rtt, source_idx_t{0}, td04);

  transit_realtime::TripDescriptor td05;
  td05.set_start_date("20240129");
  td05.set_trip_id("T05");
  td05.set_start_time("10:00:00");
  auto const [r05, t05] = gtfsrt_resolve_run(date::sys_days{January / 1 / 2024},
                                             tt, &rtt, source_idx_t{0}, td05);

  transit_realtime::TripDescriptor td06;
  td06.set_start_date("20240205");
  td06.set_trip_id("T06");
  td06.set_start_time("06:00:00");
  auto const [r06, t06] = gtfsrt_resolve_run(date::sys_days{January / 1 / 2024},
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
  auto const msg07_2b = rt::json_to_protobuf(kVehiclePositionT07_before_S02);
  auto const msg07_3 = rt::json_to_protobuf(kVehiclePositionT07_3);
  auto const msg07_4 = rt::json_to_protobuf(kVehiclePositionT07_4);
  auto const msg07_4b = rt::json_to_protobuf(kVehiclePositionT07_before_S03);
  auto const msg07_5 = rt::json_to_protobuf(kVehiclePositionT07_5);
  auto const msg07_6 = rt::json_to_protobuf(kVehiclePositionT07_6);
  auto const msg07_6b = rt::json_to_protobuf(kVehiclePositionT07_before_S04);
  auto const msg07_7 = rt::json_to_protobuf(kVehiclePositionT07_7);

  transit_realtime::TripDescriptor td07;
  td07.set_start_date("20240205");
  td07.set_trip_id("T07");
  td07.set_start_time("10:00:00");
  auto const [r07, t07] = gtfsrt_resolve_run(date::sys_days{January / 1 / 2024},
                                             tt, &rtt, source_idx_t{0}, td07);

  ASSERT_TRUE(r07.valid());

  /**
  std::cout << "--- Live Update 1 ---" << std::endl;
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg07_1, &dp);
  std::cout << rt::frun{tt, &rtt, r07} << std::endl;

  std::cout << "--- Live Update 2 ---" << std::endl;
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg07_2, &dp);
  std::cout << rt::frun{tt, &rtt, r07} << std::endl;

  std::cout << "--- Live Update 2b ---" << std::endl;
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg07_2b, &dp);
  std::cout << rt::frun{tt, &rtt, r07} << std::endl;

  std::cout << "--- Live Update 3 ---" << std::endl;
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg07_3, &dp);
  std::cout << rt::frun{tt, &rtt, r07} << std::endl;

  std::cout << "--- Live Update 4 ---" << std::endl;
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg07_4, &dp);
  std::cout << rt::frun{tt, &rtt, r07} << std::endl;

  std::cout << "--- Live Update 4b ---" << std::endl;
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg07_4b, &dp);
  std::cout << rt::frun{tt, &rtt, r07} << std::endl;

  std::cout << "--- Live Update 5 ---" << std::endl;
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg07_5, &dp);
  std::cout << rt::frun{tt, &rtt, r07} << std::endl;

  std::cout << "--- Live Update 6 ---" << std::endl;
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg07_6, &dp);
  std::cout << rt::frun{tt, &rtt, r07} << std::endl;

  std::cout << "--- Live Update 6b ---" << std::endl;
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg07_6b, &dp);
  std::cout << rt::frun{tt, &rtt, r07} << std::endl;

  std::cout << "--- Live Update 7 ---" << std::endl;
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg07_7, &dp);
  std::cout << rt::frun{tt, &rtt, r07} << std::endl;

  std::stringstream ss_tts;
  ss_tts << tts;
  std::cout << ss_tts.str() << std::endl;

  std::stringstream ss_dps;
  ss_dps << dps;
  std::cout << ss_dps.str() << std::endl;

  **/
}
