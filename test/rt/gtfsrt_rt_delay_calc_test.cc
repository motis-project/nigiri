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


// reads a binary GTFS-RT-FeedMessage from a file and returns it as a serialized
// string
[[maybe_unused]] std::string read_gtfsrt_file(std::filesystem::path const& p) {
  auto f = std::ifstream{p, std::ios::in | std::ios::binary};
  utl::verify(f.good(), "unable to open gtfsrt file {}", p.string());
  auto buf = std::string{std::istreambuf_iterator<char>{f},
                         std::istreambuf_iterator<char>{}};
  utl::verify(!buf.empty(), "gtfsrt file {} is empty", p.string());
  return buf;
}

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
201-Weekday-66-23SUMM-1111100,20230807,1
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
201,201-Weekday-66-23SUMM-1111100,11,Conestoga Station,0,340341,2010025,1,1
201,201-Weekday-66-23SUMM-1111100,12,Conestoga Station,0,340341,2010025,1,1
201,201-Weekday-66-23SUMM-1111100,13,Conestoga Station,0,340341,2010025,1,1
201,201-Weekday-66-23SUMM-1111100,14,Conestoga Station,0,340341,2010025,1,1
201,201-Weekday-66-23SUMM-1111100,15,Conestoga Station,0,340341,2010025,1,1
201,201-Weekday-66-23SUMM-1111100,2,Conestoga Station,0,340341,2010025,1,1
201,201-Weekday-66-23SUMM-1111100,3,Conestoga Station,0,340341,2010025,1,1



# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
11,05:00:00,05:15:00,2351,1,0,0
11,06:00:00,06:15:00,1033,2,0,0
11,07:00:00,07:15:00,2086,3,0,0
11,08:00:00,08:15:00,2885,4,0,0
12,05:00:00,05:15:00,2351,1,0,0
12,06:00:00,06:15:00,1033,2,0,0
12,07:00:00,07:15:00,2086,3,0,0
12,08:00:00,08:15:00,2885,4,0,0
13,05:00:00,05:15:00,2351,1,0,0
13,06:00:00,06:15:00,1033,2,0,0
13,07:00:00,07:15:00,2086,3,0,0
13,08:00:00,08:15:00,2885,4,0,0
14,05:00:00,05:15:00,2351,1,0,0
14,06:00:00,06:15:00,1033,2,0,0
14,07:00:00,07:15:00,2086,3,0,0
14,08:00:00,08:15:00,2885,4,0,0
15,05:00:00,05:15:00,2351,1,0,0
15,06:00:00,06:15:00,1033,2,0,0
15,07:00:00,07:15:00,2086,3,0,0
15,08:00:00,08:15:00,2885,4,0,0
2,04:00:00,04:15:00,2351,1,0,0
2,05:00:00,05:15:00,1033,2,0,0
2,06:00:00,06:15:00,2086,3,0,0
2,07:00:00,07:15:00,2885,4,0,0
3,05:00:00,05:15:00,2351,1,0,0
3,06:00:00,06:15:00,1033,2,0,0
3,07:00:00,07:15:00,2086,3,0,0
3,08:00:00,08:15:00,2885,4,0,0
)");
}

// Test case:
// 3 Messungen pro Segment:
//      1 43.422095, -80.462740; 43.420610844275686,
//      -80.4643359499176; 43.419023, -80.466600; 2 43.419023,
//      -80.466600; 43.41855161265139, -80.47031974600564; 43.417796,
//      -80.473666; 3 43.417796, -80.473666; 43.416960971123125,
//      -80.47671371418541; 43.415733, -80.480340;
// 5 historische Trips:
//      1.
//        1. +4min
//        2. +2min
//        3. +5min
//        4. +1min
//        5. +7min
//        6. +1min
//        7. +0min
//        8. +3min
//        9. +4min
//      2.
//      3.
//      4.
//      5.
// 1 direkter Vorgänger-Trip:
//      10min delay
// 1 "live" Trip
//      25min delay

auto const kVehiclePositionT11 =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1688371440"
 },
 "entity": [
  {
    "id": "111",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "11",
      "startTime": "05:15:00",
      "startDate": "20230703",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.422095",
      "longitude": "-80.462740"
    },
    "timestamp": "1688361540",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "112",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "11",
      "startTime": "05:15:00",
      "startDate": "20230703",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.420610844275686",
      "longitude": "-80.4643359499176"
    },
    "timestamp": "1688362800",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "113",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "11",
      "startTime": "05:15:00",
      "startDate": "20230703",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.419023",
      "longitude": "-80.466600"
    },
    "timestamp": "1688364300",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "114",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "11",
      "startTime": "05:15:00",
      "startDate": "20230703",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.419023",
      "longitude": "-80.466600"
    },
    "timestamp": "1688364960",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "115",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "11",
      "startTime": "05:15:00",
      "startDate": "20230703",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.41855161265139",
      "longitude": "-80.47031974600564"
    },
    "timestamp": "1688366700",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "116",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "11",
      "startTime": "05:15:00",
      "startDate": "20230703",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.417796",
      "longitude": "-80.473666"
    },
    "timestamp": "1688367660",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "117",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "11",
      "startTime": "05:15:00",
      "startDate": "20230703",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.417796",
      "longitude": "-80.473666"
    },
    "timestamp": "1688368500",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "118",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "11",
      "startTime": "05:15:00",
      "startDate": "20230703",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.416960971123125",
      "longitude": "-80.47671371418541"
    },
    "timestamp": "1688370060",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "119",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "11",
      "startTime": "05:15:00",
      "startDate": "20230703",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.415733",
      "longitude": "-80.480340"
    },
    "timestamp": "1688371440",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  }
 ]
})"s;

auto const kVehiclePositionT12 =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1688976300"
 },
 "entity": [
  {
    "id": "121",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "12",
      "startTime": "05:15:00",
      "startDate": "20230710",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.422095",
      "longitude": "-80.462740"
    },
    "timestamp": "1688967300",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "122",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "12",
      "startTime": "05:15:00",
      "startDate": "20230710",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.420610844275686",
      "longitude": "-80.4643359499176"
    },
    "timestamp": "1688968800",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "123",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "12",
      "startTime": "05:15:00",
      "startDate": "20230710",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.419023",
      "longitude": "-80.466600"
    },
    "timestamp": "1688969940",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "124",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "12",
      "startTime": "05:15:00",
      "startDate": "20230710",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.419023",
      "longitude": "-80.466600"
    },
    "timestamp": "1688970720",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "125",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "12",
      "startTime": "05:15:00",
      "startDate": "20230710",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.41855161265139",
      "longitude": "-80.47031974600564"
    },
    "timestamp": "1688971980",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "126",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "12",
      "startTime": "05:15:00",
      "startDate": "20230710",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.417796",
      "longitude": "-80.473666"
    },
    "timestamp": "1688973180",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "127",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "12",
      "startTime": "05:15:00",
      "startDate": "20230710",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.417796",
      "longitude": "-80.473666"
    },
    "timestamp": "1688973900",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "128",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "12",
      "startTime": "05:15:00",
      "startDate": "20230710",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.416960971123125",
      "longitude": "-80.47671371418541"
    },
    "timestamp": "1688975220",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "129",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "12",
      "startTime": "05:15:00",
      "startDate": "20230710",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.415733",
      "longitude": "-80.480340"
    },
    "timestamp": "1688976300",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  }
 ]
})"s;

auto const kVehiclePositionT13 =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1689580200"
 },
 "entity": [
  {
    "id": "131",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "13",
      "startTime": "05:15:00",
      "startDate": "20230717",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.422095",
      "longitude": "-80.462740"
    },
    "timestamp": "1689570300",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "132",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "13",
      "startTime": "05:15:00",
      "startDate": "20230717",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.420610844275686",
      "longitude": "-80.4643359499176"
    },
    "timestamp": "1689571740",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "133",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "13",
      "startTime": "05:15:00",
      "startDate": "20230717",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.419023",
      "longitude": "-80.466600"
    },
    "timestamp": "1689573120",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "134",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "13",
      "startTime": "05:15:00",
      "startDate": "20230717",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.419023",
      "longitude": "-80.466600"
    },
    "timestamp": "1689574080",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "135",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "13",
      "startTime": "05:15:00",
      "startDate": "20230717",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.41855161265139",
      "longitude": "-80.47031974600564"
    },
    "timestamp": "1689575520",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "136",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "13",
      "startTime": "05:15:00",
      "startDate": "20230717",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.417796",
      "longitude": "-80.473666"
    },
    "timestamp": "1689576780",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "137",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "13",
      "startTime": "05:15:00",
      "startDate": "20230717",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.417796",
      "longitude": "-80.473666"
    },
    "timestamp": "1689577620",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "138",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "13",
      "startTime": "05:15:00",
      "startDate": "20230717",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.416960971123125",
      "longitude": "-80.47671371418541"
    },
    "timestamp": "1689578940",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "139",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "13",
      "startTime": "05:15:00",
      "startDate": "20230717",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.415733",
      "longitude": "-80.480340"
    },
    "timestamp": "1689580200",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  }
 ]
})"s;

auto const kVehiclePositionT14 =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1690185600"
 },
 "entity": [
  {
    "id": "141",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "14",
      "startTime": "05:15:00",
      "startDate": "20230724",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.422095",
      "longitude": "-80.462740"
    },
    "timestamp": "1690175700",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "142",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "14",
      "startTime": "05:15:00",
      "startDate": "20230724",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.420610844275686",
      "longitude": "-80.4643359499176"
    },
    "timestamp": "1690177080",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "143",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "14",
      "startTime": "05:15:00",
      "startDate": "20230724",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.419023",
      "longitude": "-80.466600"
    },
    "timestamp": "1690178400",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "144",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "14",
      "startTime": "05:15:00",
      "startDate": "20230724",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.419023",
      "longitude": "-80.466600"
    },
    "timestamp": "1690179540",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "145",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "14",
      "startTime": "05:15:00",
      "startDate": "20230724",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.41855161265139",
      "longitude": "-80.47031974600564"
    },
    "timestamp": "1690180920",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "146",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "14",
      "startTime": "05:15:00",
      "startDate": "20230724",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.417796",
      "longitude": "-80.473666"
    },
    "timestamp": "1690182240",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "147",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "14",
      "startTime": "05:15:00",
      "startDate": "20230724",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.417796",
      "longitude": "-80.473666"
    },
    "timestamp": "1690182900",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "148",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "14",
      "startTime": "05:15:00",
      "startDate": "20230724",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.416960971123125",
      "longitude": "-80.47671371418541"
    },
    "timestamp": "1690184280",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "149",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "14",
      "startTime": "05:15:00",
      "startDate": "20230724",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.415733",
      "longitude": "-80.480340"
    },
    "timestamp": "1690185600",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  }
 ]
})"s;

auto const kVehiclePositionT15 =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1690790640"
 },
 "entity": [
  {
    "id": "151",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "15",
      "startTime": "05:15:00",
      "startDate": "20230731",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.422095",
      "longitude": "-80.462740"
    },
    "timestamp": "1690780620",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "152",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "15",
      "startTime": "05:15:00",
      "startDate": "20230731",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.420610844275686",
      "longitude": "-80.4643359499176"
    },
    "timestamp": "1690782120",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "153",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "15",
      "startTime": "05:15:00",
      "startDate": "20230731",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.419023",
      "longitude": "-80.466600"
    },
    "timestamp": "1690783260",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "154",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "15",
      "startTime": "05:15:00",
      "startDate": "20230731",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.419023",
      "longitude": "-80.466600"
    },
    "timestamp": "1690784520",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "155",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "15",
      "startTime": "05:15:00",
      "startDate": "20230731",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.41855161265139",
      "longitude": "-80.47031974600564"
    },
    "timestamp": "1690785540",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "156",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "15",
      "startTime": "05:15:00",
      "startDate": "20230731",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.417796",
      "longitude": "-80.473666"
    },
    "timestamp": "1690786560",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "157",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "15",
      "startTime": "05:15:00",
      "startDate": "20230731",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.417796",
      "longitude": "-80.473666"
    },
    "timestamp": "1690788180",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "158",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "15",
      "startTime": "05:15:00",
      "startDate": "20230731",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.416960971123125",
      "longitude": "-80.47671371418541"
    },
    "timestamp": "1690789440",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "159",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "15",
      "startTime": "05:15:00",
      "startDate": "20230731",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.415733",
      "longitude": "-80.480340"
    },
    "timestamp": "1690790640",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  }
 ]
})"s;

auto const kVehiclePositionT2 =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1691305260"
 },
 "entity": [
  {
    "id": "21",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "2",
      "startTime": "04:15:00",
      "startDate": "20230807",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.422095",
      "longitude": "-80.462740"
    },
    "timestamp": "1691295540",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "22",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "2",
      "startTime": "04:15:00",
      "startDate": "20230807",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.420610844275686",
      "longitude": "-80.4643359499176"
    },
    "timestamp": "1691296800",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "23",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "2",
      "startTime": "04:15:00",
      "startDate": "20230807",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.419023",
      "longitude": "-80.466600"
    },
    "timestamp": "1691297880",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "24",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "2",
      "startTime": "04:15:00",
      "startDate": "20230807",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.419023",
      "longitude": "-80.466600"
    },
    "timestamp": "1691299500",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "25",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "2",
      "startTime": "04:15:00",
      "startDate": "20230807",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.41855161265139",
      "longitude": "-80.47031974600564"
    },
    "timestamp": "1691300340",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "26",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "2",
      "startTime": "04:15:00",
      "startDate": "20230807",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.417796",
      "longitude": "-80.473666"
    },
    "timestamp": "1691301440",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "27",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "2",
      "startTime": "04:15:00",
      "startDate": "20230807",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.417796",
      "longitude": "-80.473666"
    },
    "timestamp": "1691303040",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "28",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "2",
      "startTime": "04:15:00",
      "startDate": "20230807",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.416960971123125",
      "longitude": "-80.47671371418541"
    },
    "timestamp": "1691303880",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  },
  {
    "id": "29",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "2",
      "startTime": "04:15:00",
      "startDate": "20230807",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.415733",
      "longitude": "-80.480340"
    },
    "timestamp": "1691305260",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  }
 ]
})"s;

auto const kVehiclePositionT31 =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1691386200"
 },
 "entity": [
  {
    "id": "31",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "3",
      "startTime": "05:15:00",
      "startDate": "20230807",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.422095",
      "longitude": "-80.462740"
    },
    "timestamp": "1691386200",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  }
 ]
})"s;

auto const kVehiclePositionT32 =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1691387640"
 },
 "entity": [
  {
    "id": "32",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "3",
      "startTime": "05:15:00",
      "startDate": "20230807",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.420610844275686",
      "longitude": "-80.4643359499176"
    },
    "timestamp": "1691387640",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  }
 ]
})"s;

auto const kVehiclePositionT33 =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1691389200"
 },
 "entity": [
  {
    "id": "33",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "3",
      "startTime": "05:15:00",
      "startDate": "20230807",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.419023",
      "longitude": "-80.466600"
    },
    "timestamp": "1691389200",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  }
 ]
})"s;

auto const kVehiclePositionT34 =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1691389980"
 },
 "entity": [
  {
    "id": "34",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "3",
      "startTime": "05:15:00",
      "startDate": "20230807",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.419023",
      "longitude": "-80.466600"
    },
    "timestamp": "1691389980",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  }
 ]
})"s;

auto const kVehiclePositionT35 =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1691391240"
 },
 "entity": [
  {
    "id": "35",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "3",
      "startTime": "05:15:00",
      "startDate": "20230807",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.41855161265139",
      "longitude": "-80.47031974600564"
    },
    "timestamp": "1691391240",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  }
 ]
})"s;

auto const kVehiclePositionT36 =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1691392440"
 },
 "entity": [
  {
    "id": "36",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "3",
      "startTime": "05:15:00",
      "startDate": "20230807",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.417796",
      "longitude": "-80.473666"
    },
    "timestamp": "1691392440",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  }
 ]
})"s;

auto const kVehiclePositionT37 =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1691393400"
 },
 "entity": [
  {
    "id": "37",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "3",
      "startTime": "05:15:00",
      "startDate": "20230807",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.417796",
      "longitude": "-80.473666"
    },
    "timestamp": "1691393400",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  }
 ]
})"s;

auto const kVehiclePositionT38 =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1691394660"
 },
 "entity": [
  {
    "id": "38",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "3",
      "startTime": "05:15:00",
      "startDate": "20230807",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.416960971123125",
      "longitude": "-80.47671371418541"
    },
    "timestamp": "1691394660",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  }
 ]
})"s;

auto const kVehiclePositionT39 =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1691395740"
 },
 "entity": [
  {
    "id": "39",
    "isDeleted": false,
    "vehicle": {
    "trip": {
      "tripId": "3",
      "startTime": "05:15:00",
      "startDate": "20230807",
      "routeId": "201"
    },
    "position": {
      "latitude": "43.415733",
      "longitude": "-80.480340"
    },
    "timestamp": "1691395740",
    "occupancy_status": "MANY_SEATS_AVAILABLE"
    }
  }
 ]
})"s;

}  // namespace

TEST(rt, gtfsrt_rt_delay_calc_test) {
  std::cout << "Test rt::gtfsrt_rt_delay_calc_test" << std::endl;

  // Load static timetable.
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2023_y / July / 3},
                    date::sys_days{2023_y / August / 12}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  // Create empty RT timetable.
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2023_y / August / 12});

  auto tts = hist_trip_times_storage{};

  auto vtm = vehicle_trip_matching{};

  // Create empty Delay Prediction Storage
  auto dps = delay_prediction_storage{};

  auto dp = delay_prediction{algorithm::kIntelligent,
                             hist_trip_mode::kSameDay,
                             1,
                             5,
                             &dps,
                             &tts,
                             &vtm};

  // Update.
  auto const msg11 = rt::json_to_protobuf(kVehiclePositionT11);
  auto const msg12 = rt::json_to_protobuf(kVehiclePositionT12);
  auto const msg13 = rt::json_to_protobuf(kVehiclePositionT13);
  auto const msg14 = rt::json_to_protobuf(kVehiclePositionT14);
  auto const msg15 = rt::json_to_protobuf(kVehiclePositionT15);
  auto const msg2 = rt::json_to_protobuf(kVehiclePositionT2);
  auto const msg31 = rt::json_to_protobuf(kVehiclePositionT31);
  auto const msg32 = rt::json_to_protobuf(kVehiclePositionT32);
  auto const msg33 = rt::json_to_protobuf(kVehiclePositionT33);
  auto const msg34 = rt::json_to_protobuf(kVehiclePositionT34);
  auto const msg35 = rt::json_to_protobuf(kVehiclePositionT35);
  auto const msg36 = rt::json_to_protobuf(kVehiclePositionT36);
  auto const msg37 = rt::json_to_protobuf(kVehiclePositionT37);
  auto const msg38 = rt::json_to_protobuf(kVehiclePositionT38);
  auto const msg39 = rt::json_to_protobuf(kVehiclePositionT39);

  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg11, &dp);
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg12, &dp);
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg13, &dp);
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg14, &dp);
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg15, &dp);
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg2, &dp);
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg31, &dp);

  // Print trip.
  transit_realtime::TripDescriptor td11;
  td11.set_start_date("20230703");
  td11.set_trip_id("11");
  td11.set_start_time("05:15:00");
  auto const [r11, t11] = rt::gtfsrt_resolve_run(
      date::sys_days{May / 1 / 2019}, tt, &rtt, source_idx_t{0}, td11);
  transit_realtime::TripDescriptor td12;
  td12.set_start_date("20230710");
  td12.set_trip_id("12");
  td12.set_start_time("05:15:00");
  auto const [r12, t12] = rt::gtfsrt_resolve_run(
      date::sys_days{May / 1 / 2019}, tt, &rtt, source_idx_t{0}, td12);
  transit_realtime::TripDescriptor td13;
  td13.set_start_date("20230717");
  td13.set_trip_id("13");
  td13.set_start_time("05:15:00");
  auto const [r13, t13] = rt::gtfsrt_resolve_run(
      date::sys_days{May / 1 / 2019}, tt, &rtt, source_idx_t{0}, td13);
  transit_realtime::TripDescriptor td14;
  td14.set_start_date("20230724");
  td14.set_trip_id("14");
  td14.set_start_time("05:15:00");
  auto const [r14, t14] = rt::gtfsrt_resolve_run(
      date::sys_days{May / 1 / 2019}, tt, &rtt, source_idx_t{0}, td14);
  transit_realtime::TripDescriptor td15;
  td15.set_start_date("20230731");
  td15.set_trip_id("15");
  td15.set_start_time("05:15:00");
  auto const [r15, t15] = rt::gtfsrt_resolve_run(
      date::sys_days{May / 1 / 2019}, tt, &rtt, source_idx_t{0}, td15);
  transit_realtime::TripDescriptor td2;
  td2.set_start_date("20230807");
  td2.set_trip_id("2");
  td2.set_start_time("04:15:00");
  auto const [r2, t2] = rt::gtfsrt_resolve_run(date::sys_days{May / 1 / 2019},
                                               tt, &rtt, source_idx_t{0}, td2);
  transit_realtime::TripDescriptor td3;
  td3.set_start_date("20230807");
  td3.set_trip_id("3");
  td3.set_start_time("05:15:00");
  auto const [r3, t3] = rt::gtfsrt_resolve_run(date::sys_days{May / 1 / 2019},
                                               tt, &rtt, source_idx_t{0}, td3);
  ASSERT_TRUE(r11.valid());
  ASSERT_TRUE(r12.valid());
  ASSERT_TRUE(r13.valid());
  ASSERT_TRUE(r14.valid());
  ASSERT_TRUE(r15.valid());
  ASSERT_TRUE(r2.valid());
  ASSERT_TRUE(r3.valid());

  std::stringstream ss_tts;
  ss_tts << tts;
  std::cout << ss_tts.str() << std::endl;
  // EXPECT_EQ(expected_tts, ss_tts.str());

  std::stringstream ss_dps1;
  ss_dps1 << dps;
  std::cout << ss_dps1.str() << std::endl;
  // EXPECT_EQ(expected_dps, ss_dps.str());

  auto const fr = rt::frun{tt, &rtt, r3};
  auto ss1 = std::stringstream{};
  ss1 << "\n" << fr;
  std::cout << ss1.str() << std::endl;

  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg32, &dp);
  auto ss2 = std::stringstream{};
  ss2 << "\n" << fr;
  std::cout << ss2.str() << std::endl;
  std::stringstream ss_dps2;
  ss_dps2 << dps;
  std::cout << ss_dps2.str() << std::endl;

  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg33, &dp);
  auto ss3 = std::stringstream{};
  ss3 << "\n" << fr;
  std::cout << ss3.str() << std::endl;
  std::stringstream ss_dps3;
  ss_dps3 << dps;
  std::cout << ss_dps3.str() << std::endl;

  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg34, &dp);
  auto ss4 = std::stringstream{};
  ss4 << "\n" << fr;
  std::cout << ss4.str() << std::endl;
  std::stringstream ss_dps4;
  ss_dps4 << dps;
  std::cout << ss_dps4.str() << std::endl;

  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg35, &dp);
  auto ss5 = std::stringstream{};
  ss5 << "\n" << fr;
  std::cout << ss5.str() << std::endl;
  std::stringstream ss_dps5;
  ss_dps5 << dps;
  std::cout << ss_dps5.str() << std::endl;

  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg36, &dp);
  auto ss6 = std::stringstream{};
  ss6 << "\n" << fr;
  std::cout << ss6.str() << std::endl;
  std::stringstream ss_dps6;
  ss_dps6 << dps;
  std::cout << ss_dps6.str() << std::endl;

  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg37, &dp);
  auto ss7 = std::stringstream{};
  ss7 << "\n" << fr;
  std::cout << ss7.str() << std::endl;
  std::stringstream ss_dps7;
  ss_dps7 << dps;
  std::cout << ss_dps7.str() << std::endl;

  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg38, &dp);
  auto ss8 = std::stringstream{};
  ss8 << "\n" << fr;
  std::cout << ss8.str() << std::endl;
  std::stringstream ss_dps8;
  ss_dps8 << dps;
  std::cout << ss_dps8.str() << std::endl;

  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg39, &dp);
  auto ss9 = std::stringstream{};
  ss9 << "\n" << fr;
  std::cout << ss9.str() << std::endl;
  std::stringstream ss_dps9;
  ss_dps9 << dps;
  std::cout << ss_dps9.str() << std::endl;
}

/**
TEST(rt, gtfsrt_rt_delay_calc_real_data_test_file) {
  std::cout << "Test rt::gtfsrt_rt_delay_calc_real_data_test" << std::endl;

  auto tts = hist_trip_times_storage{};

  auto vtm = vehicle_trip_matching{};

  // Create empty Delay Prediction Storage
  auto dps = delay_prediction_storage{};

  auto dp = delay_prediction{algorithm::kIntelligent,
                             hist_trip_mode::kSameDay,
                             1,
                             5,
                             &dps,
                             &tts,
                             &vtm};

  // Statisches GTFS laden.
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2026_y / February / 14},
                    date::sys_days{2026_y / February / 17}};

  // Hier: das reale GTFS aus gtfs zip-file verwenden
  auto const p = std::filesystem::path{"NL-20260216.gtfs.zip"};
  auto const dir_ptr = loader::make_dir(p);
  load_timetable({}, source_idx_t{0}, *dir_ptr, tt);
  finalize(tt);

  // RT-Timetable für einen Tag innerhalb der GTFS-Daten anlegen.
  auto const base_day = date::sys_days{2026_y / February / 16};
  auto rtt = rt::create_rt_timetable(tt, base_day);

  // vehiclePositions.pb aus dem Projekt-Root einlesen.
  auto const vp_path = std::filesystem::path{"vehiclePositions.pb"};
  auto const buf = read_gtfsrt_file(vp_path);

  // Als VehiclePositions-Feed einspeisen.
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", buf, &dp);

  std::stringstream ss_tts;
  ss_tts << tts;
  std::cout << "hist_trip_times_storage from vehiclePositions.pb:\n"
            << ss_tts.str() << std::endl;

  std::stringstream ss_dps;
  ss_dps << dps;
  std::cout << "delay_prediction_storage from vehiclePositions.pb:\n"
            << ss_dps.str() << std::endl;

  EXPECT_FALSE(ss_tts.str().empty() && ss_dps.str().empty());
}
**/