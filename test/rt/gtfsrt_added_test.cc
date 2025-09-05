#include <optional>

#include "geo/latlng.h"
#include "gtest/gtest.h"

#include "google/protobuf/util/json_util.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/util.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

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
agency_name,agency_url,agency_timezone,agency_lang,agency_phone,agency_id
invalid,https://test.com,Europe/London,DE,0800123456,INVALID_AGENCY
test,https://test.com,Europe/Berlin,DE,0800123456,AGENCY_1

# stops.txt
stop_id,stop_name,stop_lat,stop_lon
A,A,0.01,0.01
B,B,0.02,0.02
C,C,0.03,0.03
D,D,0.04,0.04
E,E,0.05,0.05
F,F,0.06,0.06

# calendar_dates.txt
service_id,date,exception_type
SERVICE_1,20230810,1
SERVICE_1,20230811,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_type
ROUTE_1,AGENCY_1,Route 1,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id,
ROUTE_1,SERVICE_1,TRIP_1,E,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
TRIP_1,10:00:00,10:00:00,A,1,0,0
TRIP_1,11:00:00,11:00:00,B,2,0,0
TRIP_1,12:00:00,12:00:00,C,3,0,0
TRIP_1,13:00:00,13:00:00,D,4,0,0
TRIP_1,14:00:00,14:00:00,E,5,0,0
TRIP_1,15:00:00,15:00:00,F,6,0,0
)");
}

auto const kTripAdded =
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
      "tripId": "TRIP_ADDED",
      "scheduleRelationship": "ADDED",
      "routeId": "ROUTE_1"
     },
     "stopTimeUpdate": [
      {
       "stopSequence": 1,
       "arrival": {
        "time": "1691658900",
        "delay": 60
       },
       "stopId": "E",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 2,
       "arrival": {
        "time": "1691658960"
       },
       "stopId": "D",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 4,
       "arrival": {
        "time": "1691745480"
       },
       "stopId": "B",
       "scheduleRelationship": "SCHEDULED"
      }
     ]
    }
  }
 ]
})"s;

auto const kTripNew =
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
      "tripId": "TRIP_NEW",
      "startTime": "10:00:00",
      "startDate": "20230810",
      "scheduleRelationship": "NEW",
      "routeId": "ROUTE_1"
     },
     "tripProperties": {
      "tripShortName": "New Route",
      "tripHeadsign": "New Headsign"
     },
     "stopTimeUpdate": [
      {
       "stopSequence": 1,
       "arrival": {
        "time": "1691658900"
       },
       "departure": {
        "time": "1691658900"
       },
       "stopId": "E",
       "scheduleRelationship": "SCHEDULED",
       "stopTimeProperties": {
          "dropOffType": "NONE",
          "pickupType": "REGULAR",
       }
      },
      {
       "stopSequence": 2,
       "arrival": {
        "time": "1691658960"
       },
       "departure": {
        "time": "1691658960"
       },
       "stopId": "D",
       "scheduleRelationship": "SKIPPED"
      },
      {
       "stopSequence": 3,
       "arrival": {
        "time": "1691659080"
       },
       "departure": {
        "time": "1691659080"
       },
       "stopId": "B",
       "scheduleRelationship": "SCHEDULED",
       "stopTimeProperties": {
          "dropOffType": "REGULAR",
          "pickupType": "NONE",
       }
      }
     ]
    }
  }
 ]
})"s;

auto const kTripNewLonger =
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
      "tripId": "TRIP_NEW",
      "scheduleRelationship": "NEW"
     },
     "tripProperties": {
      "tripShortName": "New Route",
      "tripHeadsign": "New Headsign"
     },
     "stopTimeUpdate": [
      {
       "stopSequence": 1,
       "arrival": {
        "time": "1691658900"
       },
       "departure": {
        "time": "1691658900"
       },
       "stopId": "E",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 2,
       "arrival": {
        "time": "1691658960"
       },
       "departure": {
        "time": "1691658960"
       },
       "stopId": "D",
       "scheduleRelationship": "SKIPPED"
      },
      {
       "stopSequence": 3,
       "arrival": {
        "time": "1691660531"
       },
       "departure": {
        "time": "1691659080"
       },
       "stopId": "B",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 4,
       "arrival": {
        "time": "1691661431"
       },
       "departure": {
        "time": "1691661431"
       },
       "stopId": "A",
       "scheduleRelationship": "SCHEDULED"
      }
     ]
    }
  }
 ]
})"s;

auto const kTripNewRouteNonExistent =
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
      "tripId": "TRIP_NEW",
      "startTime": "10:00:00",
      "startDate": "20230811",
      "scheduleRelationship": "NEW",
      "routeId": "ROUTE_NON_EXISTENT"
     },
     "tripProperties": {
      "tripShortName": "New Route",
      "tripHeadsign": "New Headsign"
     },
     "stopTimeUpdate": [
      {
       "stopSequence": 1,
       "arrival": {
        "time": "1691658900"
       },
       "departure": {
        "time": "1691658900"
       },
       "stopId": "E",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 2,
       "arrival": {
        "time": "1691658960"
       },
       "departure": {
        "time": "1691658960"
       },
       "stopId": "D",
       "scheduleRelationship": "SKIPPED"
      },
      {
       "stopSequence": 3,
       "arrival": {
        "time": "1691659080"
       },
       "departure": {
        "time": "1691659080"
       },
       "stopId": "B",
       "scheduleRelationship": "SCHEDULED"
      }
     ]
    }
  }
 ]
})"s;

auto const kTripNewBare =
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
      "tripId": "TRIP_NEW",
      "startTime": "10:00:00",
      "startDate": "20230810",
      "scheduleRelationship": "NEW"
     }
    }    
  }
 ]
})"s;

auto const kTripNewBareWithStops =
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
      "tripId": "TRIP_NEW",
      "scheduleRelationship": "NEW"
     },
     "stopTimeUpdate": [
      {
       "stopSequence": 1,
       "arrival": {
        "time": "1691658900"
       },
       "departure": {
        "time": "1691658900"
       },
       "stopId": "E",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 3,
       "arrival": {
        "time": "1691659080"
       },
       "departure": {
        "time": "1691659080"
       },
       "stopId": "B",
       "scheduleRelationship": "SCHEDULED"
      }
     ]
    }
  }
 ]
})"s;

auto const kTripNewNonExistingStops =
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
      "tripId": "TRIP_NEW",
      "startTime": "10:00:00",
      "startDate": "20230810",
      "scheduleRelationship": "NEW"
     },
     "stopTimeUpdate": [
      {
       "stopSequence": 1,
       "arrival": {
        "time": "1691658900"
       },
       "departure": {
        "time": "1691658900"
       },
       "stopId": "NOT_EXISTING",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 2,
       "arrival": {
        "time": "1691658960"
       },
       "departure": {
        "time": "1691658960"
       },
       "stopId": "NOT_EXISTING",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 3,
       "arrival": {
        "time": "1691745480"
       },
       "departure": {
        "time": "1691745480"
       },
       "stopId": "NOT_EXISTING",
       "scheduleRelationship": "SCHEDULED"
      }
     ]
    }    
  }
 ]
})"s;

auto const kTripNewRelative =
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
      "tripId": "TRIP_NEW",
      "startTime": "10:00:00",
      "startDate": "20230810",
      "scheduleRelationship": "NEW"
     },
     "stopTimeUpdate": [
      {
       "stopSequence": 1,
       "arrival": {
        "delay": 60
       },
       "departure": {
        "delay": 60
       },
       "stopId": "A",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 2,
       "arrival": {
        "time": "1691658960"
       },
       "departure": {
        "time": "1691658960"
       },
       "stopId": "B",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 3,
       "arrival": {
        "time": "1691745480"
       },
       "departure": {
        "time": "1691745480"
       },
       "stopId": "C",
       "scheduleRelationship": "SCHEDULED"
      }
     ]
    }    
  }
 ]
})"s;

auto const kTripReplacement =
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
      "tripId": "TRIP_1",
      "startTime": "10:00:00",
      "startDate": "20230810",
      "scheduleRelationship": "REPLACEMENT"
     },
     "stopTimeUpdate": [
      {
       "stopSequence": 1,
       "arrival": {
        "time": "1691658900"
       },
       "departure": {
        "time": "1691658900"
       },
       "stopId": "E",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 2,
       "arrival": {
        "time": "1691658960"
       },
       "departure": {
        "time": "1691658960"
       },
       "stopId": "D",
       "scheduleRelationship": "SKIPPED"
      },
      {
       "stopSequence": 3,
       "arrival": {
        "time": "1691659080"
       },
       "departure": {
        "time": "1691659080"
       },
       "stopId": "B",
       "scheduleRelationship": "SCHEDULED"
      }
     ]
    }
  }
 ]
})"s;

auto const kTripDuplicatedEmpty =
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
      "tripId": "TRIP_1",
      "startTime": "10:00:00",
      "startDate": "20230810",
      "scheduleRelationship": "DUPLICATED"
     },
     "tripProperties": {
      "tripId": "TRIP_1_DUPL",
      "startTime": "10:10:00",
      "startDate": "20230811"
     }
    }
  }
 ]
})"s;

auto const kTripDuplicated =
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
      "tripId": "TRIP_1",
      "startTime": "10:00:00",
      "startDate": "20230810",
      "scheduleRelationship": "DUPLICATED"
     },
     "tripProperties": {
      "tripId": "TRIP_1_DUPL",
      "startTime": "10:10:00",
      "startDate": "20230811"
     },
     "stopTimeUpdate": [
      {
       "stopSequence": 2,
       "departure": {
        "time": "1691669460"
       },
       "stopId": "B",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 6,
       "arrival": {
        "delay": 300
       },
       "stopId": "F",
       "scheduleRelationship": "SCHEDULED"
      }

     ]
    }
  }
 ]
})"s;

auto const kTripDuplicatedNonExistent =
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
      "tripId": "TRIP_1",
      "startTime": "12:00:00",
      "startDate": "20230810",
      "scheduleRelationship": "DUPLICATED"
     },
     "tripProperties": {
      "tripId": "TRIP_1_DUPL",
      "startTime": "10:10:00",
      "startDate": "20230811"
     }
    }
  }
 ]
})"s;

constexpr auto const expectedOriginal = R"(
   0: A       A...............................................                               d: 10.08 08:00 [10.08 10:00]  [{name=Route 1, day=2023-08-10, id=TRIP_1, src=0}]
   1: B       B............................................... a: 10.08 09:00 [10.08 11:00]  d: 10.08 09:00 [10.08 11:00]  [{name=Route 1, day=2023-08-10, id=TRIP_1, src=0}]
   2: C       C............................................... a: 10.08 10:00 [10.08 12:00]  d: 10.08 10:00 [10.08 12:00]  [{name=Route 1, day=2023-08-10, id=TRIP_1, src=0}]
   3: D       D............................................... a: 10.08 11:00 [10.08 13:00]  d: 10.08 11:00 [10.08 13:00]  [{name=Route 1, day=2023-08-10, id=TRIP_1, src=0}]
   4: E       E............................................... a: 10.08 12:00 [10.08 14:00]  d: 10.08 12:00 [10.08 14:00]  [{name=Route 1, day=2023-08-10, id=TRIP_1, src=0}]
   5: F       F............................................... a: 10.08 13:00 [10.08 15:00]
)"sv;

//[{name=Route 1, day=2023-08-11, id=TRIP_ADDED, src=0}]
constexpr auto const expectedAdded = R"(
   0: E       E...............................................                                                             d: 10.08 09:15 [10.08 11:15]  RT 10.08 09:15 [10.08 11:15]
   1: D       D............................................... a: 10.08 09:16 [10.08 11:16]  RT 10.08 09:16 [10.08 11:16]  d: 10.08 09:16 [10.08 11:16]  RT 10.08 09:16 [10.08 11:16]
   2: B       B............................................... a: 11.08 09:18 [11.08 11:18]  RT 11.08 09:18 [11.08 11:18]
)"sv;

//[{name=New Route, day=2023-08-10, id=TRIP_NEW, src=0}]
constexpr auto const expectedNew = R"(
   0: E       E...............................................                                                             d: 10.08 09:15 [10.08 11:15]  RT 10.08 09:15 [10.08 11:15]
   2: B       B............................................... a: 10.08 09:18 [10.08 11:18]  RT 10.08 09:18 [10.08 11:18]
)"sv;

//[{name=New Route, day=2023-08-10, id=TRIP_NEW, src=0}]
constexpr auto const expectedNewLonger = R"(
   0: E       E...............................................                                                             d: 10.08 09:15 [10.08 11:15]  RT 10.08 09:15 [10.08 11:15]
   2: B       B............................................... a: 10.08 09:42 [10.08 11:42]  RT 10.08 09:42 [10.08 11:42]
)"sv;

constexpr auto const expectedReplacement = R"(
   0: E       E...............................................                                                             d: 10.08 09:15 [10.08 11:15]  RT 10.08 09:15 [10.08 11:15]  [{name=Route 1, day=2023-08-10, id=TRIP_1, src=0}]
   2: B       B............................................... a: 10.08 09:18 [10.08 11:18]  RT 10.08 09:18 [10.08 11:18]
)"sv;

constexpr auto const expectedDuplicatedEmpty = R"(
   0: A       A...............................................                                                             d: 11.08 09:10 [11.08 10:00]  RT 11.08 09:10 [11.08 10:10]  [{name=Route 1, day=2023-08-11, id=TRIP_1_DUPL, src=0}]
   1: B       B............................................... a: 11.08 10:10 [11.08 11:10]  RT 11.08 09:10 [11.08 10:10]  d: 11.08 10:10 [11.08 11:00]  RT 11.08 10:10 [11.08 11:10]  [{name=Route 1, day=2023-08-11, id=TRIP_1_DUPL, src=0}]
   2: C       C............................................... a: 11.08 11:10 [11.08 12:10]  RT 11.08 10:10 [11.08 11:10]  d: 11.08 11:10 [11.08 12:00]  RT 11.08 11:10 [11.08 12:10]  [{name=Route 1, day=2023-08-11, id=TRIP_1_DUPL, src=0}]
   3: D       D............................................... a: 11.08 12:10 [11.08 13:10]  RT 11.08 11:10 [11.08 12:10]  d: 11.08 12:10 [11.08 13:00]  RT 11.08 12:10 [11.08 13:10]  [{name=Route 1, day=2023-08-11, id=TRIP_1_DUPL, src=0}]
   4: E       E............................................... a: 11.08 13:10 [11.08 14:10]  RT 11.08 12:10 [11.08 13:10]  d: 11.08 13:10 [11.08 14:00]  RT 11.08 13:10 [11.08 14:10]  [{name=Route 1, day=2023-08-11, id=TRIP_1_DUPL, src=0}]
   5: F       F............................................... a: 11.08 14:10 [11.08 15:10]  RT 11.08 13:10 [11.08 14:10]
)"sv;

constexpr auto const expectedDuplicated = R"(
   0: A       A...............................................                                                             d: 11.08 09:10 [11.08 10:10]  RT 11.08 09:10 [11.08 10:10]  [{name=Route 1, day=2023-08-11, id=TRIP_1_DUPL, src=0}]
   1: B       B............................................... a: 11.08 10:10 [11.08 11:10]  RT 11.08 09:10 [11.08 10:10]  d: 11.08 10:11 [11.08 11:11]  RT 11.08 10:11 [11.08 11:11]  [{name=Route 1, day=2023-08-11, id=TRIP_1_DUPL, src=0}]
   2: C       C............................................... a: 11.08 11:11 [11.08 12:11]  RT 11.08 10:11 [11.08 11:11]  d: 11.08 11:11 [11.08 12:11]  RT 11.08 11:11 [11.08 12:11]  [{name=Route 1, day=2023-08-11, id=TRIP_1_DUPL, src=0}]
   3: D       D............................................... a: 11.08 12:11 [11.08 13:11]  RT 11.08 11:11 [11.08 12:11]  d: 11.08 12:11 [11.08 13:11]  RT 11.08 12:11 [11.08 13:11]  [{name=Route 1, day=2023-08-11, id=TRIP_1_DUPL, src=0}]
   4: E       E............................................... a: 11.08 13:11 [11.08 14:11]  RT 11.08 12:11 [11.08 13:11]  d: 11.08 13:11 [11.08 14:11]  RT 11.08 13:11 [11.08 14:11]  [{name=Route 1, day=2023-08-11, id=TRIP_1_DUPL, src=0}]
   5: F       F............................................... a: 11.08 14:15 [11.08 15:15]  RT 11.08 13:15 [11.08 14:15]
)"sv;

}  // namespace

TEST(rt, gtfs_rt_added) {
  // Load static timetable.
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2023_y / August / 9},
                    date::sys_days{2023_y / August / 12}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  // Create empty RT timetable.
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2023_y / August / 11});

  // Update.
  auto const msg = rt::json_to_protobuf(kTripAdded);
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg);

  // Print trip.
  transit_realtime::TripDescriptor td;
  td.set_start_date("20230810");
  td.set_trip_id("TRIP_ADDED");
  td.set_start_time("10:00:00");
  auto const [r, t] = rt::gtfsrt_resolve_run(
      date::sys_days{2023_y / August / 10}, tt, &rtt, source_idx_t{0}, td);
  ASSERT_TRUE(r.valid());
  {
    auto const fr = rt::frun{tt, &rtt, r};
    EXPECT_EQ(3, fr.size());
    auto ss = std::stringstream{};
    ss << "\n" << fr;
    EXPECT_EQ(expectedAdded, ss.str());
    // fr.trip_idx()
    EXPECT_EQ("TRIP_ADDED", fr.id().id_);
    EXPECT_EQ(source_idx_t{0}, fr.id().src_);
    EXPECT_EQ("Route 1", fr.name());
    EXPECT_EQ("RT", fr.dbg().path_);
    EXPECT_EQ((std::pair{date::sys_days{2023_y / August / 10},
                         duration_t{9h + 15min}}),
              fr[0].get_trip_start());
    // EXPECT_EQ(, fr.trip_idx());
    EXPECT_EQ(nigiri::clasz::kBus, fr.get_clasz());
    ASSERT_FALSE(fr.is_cancelled());

    EXPECT_EQ(location_idx_t{13}, fr[0].get_stop().location_idx());
    EXPECT_EQ(true, fr[0].get_stop().in_allowed());
    EXPECT_EQ(true, fr[0].get_stop().out_allowed());
    EXPECT_EQ(location_idx_t{13}, fr[0].get_scheduled_stop().location_idx());
    EXPECT_FLOAT_EQ(0.05, fr[0].pos().lat());
    EXPECT_FLOAT_EQ(0.05, fr[0].pos().lng());
    EXPECT_EQ("", fr[0].track());
    EXPECT_EQ("E", fr[0].id());
    EXPECT_EQ("AGENCY_1",
              tt.strings_.get(fr[0].get_provider(event_type::kDep).id_));
    // EXPECT_EQ("", fr[0].get_trip_idx());
    EXPECT_EQ("?", rtt.transport_name(tt, fr.rt_));
    EXPECT_EQ("?", fr[0].trip_short_name(event_type::kDep));
    EXPECT_EQ("Route 1", fr[0].route_short_name(event_type::kDep));
    EXPECT_EQ("Route 1", fr[0].display_name(event_type::kDep));
    EXPECT_EQ(
        unixtime_t{date::sys_days{2023_y / August / 10} + 9_hours + 15_minutes},
        fr[0].scheduled_time(event_type::kDep));
    EXPECT_EQ(
        unixtime_t{date::sys_days{2023_y / August / 10} + 9_hours + 15_minutes},
        fr[0].time(event_type::kDep));
    EXPECT_EQ(duration_t{0}, fr[0].delay(event_type::kDep));
    EXPECT_EQ("", fr[0].line(event_type::kDep));
    EXPECT_EQ("", fr[0].scheduled_line(event_type::kDep));
    EXPECT_EQ("B", fr[0].direction(event_type::kDep));
    EXPECT_EQ(nigiri::clasz::kBus, fr[0].get_clasz(event_type::kDep));
    EXPECT_EQ(nigiri::clasz::kOther,
              fr[0].get_scheduled_clasz(event_type::kDep));
    EXPECT_EQ(false, fr[0].bikes_allowed(event_type::kDep));
    EXPECT_EQ(std::nullopt,
              to_str(fr[0].get_route_color(event_type::kDep).color_));
    EXPECT_EQ(std::nullopt,
              to_str(fr[0].get_route_color(event_type::kDep).text_color_));
    EXPECT_EQ(false, fr[0].in_allowed_wheelchair());
    EXPECT_EQ(false, fr[0].out_allowed_wheelchair());
    EXPECT_EQ(false, fr[0].is_cancelled());
    EXPECT_EQ(0, fr[0].section_idx(event_type::kDep));
  }
  {
    auto const fr = rt::frun{tt, nullptr, r};
    EXPECT_EQ(0, fr.size());
    EXPECT_EQ("", fr.id().id_);
    EXPECT_EQ(source_idx_t{0}, fr.id().src_);
    EXPECT_EQ("", fr.name());
    EXPECT_EQ("", fr.dbg().path_);
    // EXPECT_EQ(, fr.trip_idx());
    EXPECT_EQ(nigiri::clasz::kOther, fr.get_clasz());
    ASSERT_FALSE(fr.is_cancelled());
  }
}

TEST(rt, gtfs_rt_new) {
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
  auto const msg = rt::json_to_protobuf(kTripNew);
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg);

  // Print original trip.
  {
    transit_realtime::TripDescriptor td;
    td.set_start_date("20230810");
    td.set_trip_id("TRIP_1");
    td.set_start_time("10:00:00");
    auto const [r, t] = rt::gtfsrt_resolve_run(
        date::sys_days{2023_y / August / 10}, tt, &rtt, source_idx_t{0}, td);
    ASSERT_TRUE(r.valid());

    auto const fr = rt::frun{tt, &rtt, r};
    auto ss = std::stringstream{};
    ss << "\n" << fr;
    EXPECT_EQ(expectedOriginal, ss.str());
    ASSERT_FALSE(fr.is_cancelled());
  }

  // Print trip.
  transit_realtime::TripDescriptor td;
  td.set_start_date("20230811");
  td.set_trip_id("TRIP_NEW");
  td.set_start_time("10:00:00");

  auto const [r, t] = rt::gtfsrt_resolve_run(
      date::sys_days{2023_y / August / 11}, tt, &rtt, source_idx_t{0}, td);
  ASSERT_TRUE(r.valid());

  auto const fr = rt::frun{tt, &rtt, r};
  EXPECT_EQ(3, fr.size());
  {
    auto ss = std::stringstream{};
    ss << "\n" << fr;
    EXPECT_EQ(expectedNew, ss.str());
    EXPECT_EQ(nigiri::clasz::kBus, fr.get_clasz());
    ASSERT_FALSE(fr.is_cancelled());
    EXPECT_EQ(true, fr[0].in_allowed());
    EXPECT_EQ(false, fr[0].out_allowed());
    EXPECT_EQ(false, fr[1].in_allowed());
    EXPECT_EQ(false, fr[1].out_allowed());
    EXPECT_EQ(false, fr[2].in_allowed());
    EXPECT_EQ(true, fr[2].out_allowed());

    for (auto const [from, to] : utl::pairwise(fr)) {
      EXPECT_EQ(from.id(), "E");
      EXPECT_EQ(to.id(), "B");
    }
  }

  // Update again.
  // TODO different stops in second update
  auto const msg2 = rt::json_to_protobuf(kTripNewLonger);
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg2);
  auto ss = std::stringstream{};

  ss << "\n" << fr;
  EXPECT_EQ(1, rtt.rt_transport_location_seq_.size());
  EXPECT_EQ(expectedNewLonger, ss.str());
}

TEST(rt, gtfs_rt_new_no_route) {
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
  auto const msg = rt::json_to_protobuf(kTripNewRouteNonExistent);
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg);

  // Print trip.
  transit_realtime::TripDescriptor td;
  td.set_start_date("20230811");
  td.set_trip_id("TRIP_NEW");
  td.set_start_time("10:00:00");
  EXPECT_EQ(1, rtt.rt_transport_location_seq_.size());
  auto const [r, t] = rt::gtfsrt_resolve_run(
      date::sys_days{2023_y / August / 11}, tt, &rtt, source_idx_t{0}, td);
  ASSERT_TRUE(r.valid());

  auto const fr = rt::frun{tt, &rtt, r};
  EXPECT_EQ(fr.size(), 3);
  EXPECT_EQ(nigiri::clasz::kOther, fr.get_clasz());
  EXPECT_EQ("New Route", fr[0].trip_short_name());
  EXPECT_EQ(string_idx_t::invalid(), fr[0].get_provider(event_type::kDep).id_);
  ASSERT_FALSE(fr.is_cancelled());
}

TEST(rt, gtfs_rt_new_bare) {
  // Load static timetable.
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2023_y / August / 9},
                    date::sys_days{2023_y / August / 12}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  // Create empty RT timetable.
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2023_y / August / 10});

  {
    // Update.
    auto const msg = rt::json_to_protobuf(kTripNewBare);
    gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg);
  }

  // Print trip.
  transit_realtime::TripDescriptor td;
  td.set_start_date("20230810");
  td.set_trip_id("TRIP_NEW");
  td.set_start_time("10:00:00");
  {
    auto const [r, t] = rt::gtfsrt_resolve_run(
        date::sys_days{2023_y / August / 10}, tt, &rtt, source_idx_t{0}, td);
    EXPECT_EQ(0, rtt.rt_transport_location_seq_.size());
    ASSERT_FALSE(r.valid());
  }
  {
    // Update.
    auto const msg = rt::json_to_protobuf(kTripNewBareWithStops);
    gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg);
  }
  auto const [r, t] = rt::gtfsrt_resolve_run(
      date::sys_days{2023_y / August / 10}, tt, &rtt, source_idx_t{0}, td);
  EXPECT_EQ(1, rtt.rt_transport_location_seq_.size());
  ASSERT_TRUE(r.valid());
  auto const fr = rt::frun{tt, &rtt, r};
  EXPECT_EQ("?", fr.name());
  EXPECT_EQ(string_idx_t::invalid(), fr[0].get_provider(event_type::kDep).id_);
}

TEST(rt, gtfs_rt_new_non_existing_stops) {
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
  auto const msg = rt::json_to_protobuf(kTripNewNonExistingStops);
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg);

  // Check trip.
  transit_realtime::TripDescriptor td;
  td.set_start_date("20230810");
  td.set_trip_id("TRIP_NEW");
  td.set_start_time("10:00:00");
  auto const [r, t] = rt::gtfsrt_resolve_run(
      date::sys_days{2023_y / August / 10}, tt, &rtt, source_idx_t{0}, td);
  EXPECT_EQ(0, rtt.rt_transport_location_seq_.size());
  ASSERT_FALSE(r.valid());
}

TEST(rt, gtfs_rt_new_relative) {
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
  auto const msg = rt::json_to_protobuf(kTripNewRelative);
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg);

  EXPECT_EQ(0, rtt.rt_transport_location_seq_.size());
}

TEST(rt, DISABLED_gtfs_rt_replacement) {
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
  auto const msg = rt::json_to_protobuf(kTripReplacement);
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg);

  // Print trip.
  transit_realtime::TripDescriptor td;
  td.set_start_date("20230810");
  td.set_trip_id("TRIP_1");
  td.set_start_time("10:00:00");
  auto const [r, t] = rt::gtfsrt_resolve_run(
      date::sys_days{2023_y / August / 10}, tt, &rtt, source_idx_t{0}, td);
  ASSERT_TRUE(r.valid());

  auto const fr = rt::frun{tt, &rtt, r};
  auto ss = std::stringstream{};
  ss << "\n" << fr;
  EXPECT_EQ(expectedReplacement, ss.str());
  EXPECT_EQ(nigiri::clasz::kBus, fr.get_clasz());
  ASSERT_FALSE(fr.is_cancelled());

  for (auto const [from, to] : utl::pairwise(fr)) {
    EXPECT_EQ(from.id(), "E");
    EXPECT_EQ(to.id(), "D");
  }

  // Update again.
  auto const msg2 = rt::json_to_protobuf(kTripReplacement);
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg2);
  auto ss2 = std::stringstream{};

  ss2 << "\n" << fr;
  EXPECT_EQ(1, rtt.rt_transport_location_seq_.size());
  EXPECT_EQ(expectedReplacement, ss2.str());
}

TEST(rt, DISABLED_gtfs_rt_duplicated_empty) {
  // Load static timetable.
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2023_y / August / 9},
                    date::sys_days{2023_y / August / 12}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  // Create empty RT timetable.
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2023_y / August / 11});

  // Update.
  auto const msg = rt::json_to_protobuf(kTripDuplicatedEmpty);
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg);

  // Print original trip.
  {
    transit_realtime::TripDescriptor td;
    td.set_start_date("20230810");
    td.set_trip_id("TRIP_1");
    td.set_start_time("10:00:00");
    auto const [r, t] = rt::gtfsrt_resolve_run(
        date::sys_days{2023_y / August / 10}, tt, &rtt, source_idx_t{0}, td);
    ASSERT_TRUE(r.valid());

    auto const fr = rt::frun{tt, &rtt, r};
    auto ss = std::stringstream{};
    ss << "\n" << fr;
    EXPECT_EQ(expectedOriginal, ss.str());
    ASSERT_FALSE(fr.is_cancelled());
  }

  // Print duplicated trip.
  transit_realtime::TripDescriptor td;
  td.set_start_date("20230811");
  td.set_trip_id("TRIP_1_DUPL");
  td.set_start_time("10:10:00");
  auto const [r, t] = rt::gtfsrt_resolve_run(
      date::sys_days{2023_y / August / 11}, tt, &rtt, source_idx_t{0}, td);
  ASSERT_TRUE(r.valid());

  auto const fr = rt::frun{tt, &rtt, r};
  auto ss = std::stringstream{};
  ss << "\n" << fr;
  EXPECT_EQ(expectedDuplicatedEmpty, ss.str());
  EXPECT_EQ(nigiri::clasz::kBus, fr.get_clasz());
  ASSERT_FALSE(fr.is_cancelled());

  // Update again.
  auto const msg2 = rt::json_to_protobuf(kTripDuplicated);
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg2);

  auto ss2 = std::stringstream{};
  ss2 << "\n" << fr;
  EXPECT_EQ(1, rtt.rt_transport_location_seq_.size());
  EXPECT_EQ(expectedDuplicated, ss2.str());
}

TEST(rt, DISABLED_gtfs_rt_duplicated) {
  // Load static timetable.
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2023_y / August / 9},
                    date::sys_days{2023_y / August / 12}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  // Create empty RT timetable.
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2023_y / August / 11});

  // Update.
  auto const msg = rt::json_to_protobuf(kTripDuplicated);
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg);

  // Print original trip.
  {
    transit_realtime::TripDescriptor td;
    td.set_start_date("20230810");
    td.set_trip_id("TRIP_1");
    td.set_start_time("10:00:00");
    auto const [r, t] = rt::gtfsrt_resolve_run(
        date::sys_days{2023_y / August / 11}, tt, &rtt, source_idx_t{0}, td);
    ASSERT_TRUE(r.valid());

    auto const fr = rt::frun{tt, &rtt, r};
    EXPECT_EQ(nigiri::clasz::kBus, fr.get_clasz());
    ASSERT_FALSE(fr.is_cancelled());
  }

  // Print duplicated trip.
  transit_realtime::TripDescriptor td;
  td.set_start_date("20230811");
  td.set_trip_id("TRIP_1_DUPL");
  td.set_start_time("10:10:00");
  auto const [r, t] = rt::gtfsrt_resolve_run(
      date::sys_days{2023_y / August / 11}, tt, &rtt, source_idx_t{0}, td);
  ASSERT_TRUE(r.valid());

  auto const fr = rt::frun{tt, &rtt, r};
  auto ss = std::stringstream{};
  ss << "\n" << fr;
  EXPECT_EQ(expectedDuplicated, ss.str());
  EXPECT_EQ(nigiri::clasz::kBus, fr.get_clasz());
  ASSERT_FALSE(fr.is_cancelled());
}

TEST(rt, DISABLED_gtfs_rt_duplicated_non_existent) {
  // Load static timetable.
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2023_y / August / 9},
                    date::sys_days{2023_y / August / 12}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  // Create empty RT timetable.
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2023_y / August / 11});

  // Update.
  auto const msg = rt::json_to_protobuf(kTripDuplicatedNonExistent);
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg);

  EXPECT_EQ(0, rtt.rt_transport_location_seq_.size());
}

TEST(rt, DISABLED_gtfs_rt_duplicated_bare) {
  // Load static timetable.
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2023_y / August / 9},
                    date::sys_days{2023_y / August / 12}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  // Create empty RT timetable.
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2023_y / August / 11});

  // Update.
  auto const msg = rt::json_to_protobuf(kTripDuplicatedNonExistent);
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg);

  EXPECT_EQ(0, rtt.rt_transport_location_seq_.size());
}