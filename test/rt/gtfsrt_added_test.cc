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
      "startTime": "10:00:00",
      "startDate": "20230810",
      "scheduleRelationship": "ADDED",
      "routeId": "ROUTE_1"
     },
     "stopTimeUpdate": [
      {
       "stopSequence": 1,
       "arrival": {
        "time": "1691660288"
       },
       "departure": {
        "time": "1691660288"
       },
       "stopId": "E",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 2,
       "arrival": {
        "time": "1691660351"
       },
       "departure": {
        "time": "1691660351"
       },
       "stopId": "D",
       "scheduleRelationship": "SKIPPED"
      },
      {
       "stopSequence": 3,
       "arrival": {
        "time": "1691660431"
       },
       "departure": {
        "time": "1691660431"
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
      "startDate": "20230811",
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
        "time": "1691660288"
       },
       "departure": {
        "time": "1691660288"
       },
       "stopId": "E",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 2,
       "arrival": {
        "time": "1691660351"
       },
       "departure": {
        "time": "1691660351"
       },
       "stopId": "D",
       "scheduleRelationship": "SKIPPED"
      },
      {
       "stopSequence": 3,
       "arrival": {
        "time": "1691660431"
       },
       "departure": {
        "time": "1691660431"
       },
       "stopId": "B",
       "scheduleRelationship": "SCHEDULED"
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
      "tripId": "TRIP_1",
      "startTime": "10:00:00",
      "startDate": "20230810",
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
        "time": "1691660288"
       },
       "departure": {
        "time": "1691660288"
       },
       "stopId": "F",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 2,
       "arrival": {
        "time": "1691660351"
       },
       "departure": {
        "time": "1691660351"
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
        "time": "1691660431"
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
        "time": "1691660288"
       },
       "departure": {
        "time": "1691660288"
       },
       "stopId": "E",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 2,
       "arrival": {
        "time": "1691660351"
       },
       "departure": {
        "time": "1691660351"
       },
       "stopId": "D",
       "scheduleRelationship": "SKIPPED"
      },
      {
       "stopSequence": 3,
       "arrival": {
        "time": "1691660431"
       },
       "departure": {
        "time": "1691660431"
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
        "time": "1691660288"
       },
       "departure": {
        "time": "1691660288"
       },
       "stopId": "E",
       "scheduleRelationship": "SCHEDULED"
      },
      {
       "stopSequence": 2,
       "arrival": {
        "time": "1691660351"
       },
       "departure": {
        "time": "1691660351"
       },
       "stopId": "D",
       "scheduleRelationship": "SKIPPED"
      },
      {
       "stopSequence": 3,
       "arrival": {
        "time": "1691660431"
       },
       "departure": {
        "time": "1691660431"
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

constexpr auto const expectedAdded = R"(
   2: E       E...............................................                                                             d: 11.08 09:15 [10.08 05:15]  RT 11.08 09:15 [11.08 05:15]  [{name=Route 1, day=2023-08-11, id=TRIP_ADDED, src=0}]
   4: B       B............................................... a: 11.08 09:18 [10.08 05:18]  RT 11.08 09:18 [11.08 05:18]
)"sv;

constexpr auto const expectedNew = R"(
   2: E       E...............................................                                                             d: 10.08 09:15 [10.08 05:15]  RT 10.08 09:15 [10.08 05:15]  [{name=New Route, day=2023-08-10, id=TRIP_NEW, src=0}]
   4: B       B............................................... a: 10.08 09:18 [10.08 05:18]  RT 10.08 09:18 [10.08 05:18]
)"sv;

/*constexpr auto const expectedNewLonger = R"(
   2: F       F............................................... d: 10.08 09:15
[10.08 05:15]  RT 10.08 09:15 [10.08 05:15]  [{name=New Route, day=2023-08-10,
id=TRIP_NEW, src=0}] 4: B       B...............................................
a: 10.08 09:20 [10.08 05:20]  RT 10.08 09:20 [10.08 05:20]
)"sv;*/

constexpr auto const expectedReplacement = R"(
   2: E       E...............................................                                                             d: 10.08 09:15 [10.08 05:15]  RT 10.08 09:15 [10.08 05:15]  [{name=Route 1, day=2023-08-10, id=TRIP_1, src=0}]
   4: B       B............................................... a: 10.08 09:18 [10.08 05:18]  RT 10.08 09:18 [10.08 05:18]
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

TEST(rt, DISABLED_gtfs_rt_added) {
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
  auto const msg = rt::json_to_protobuf(kTripNew);
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg);

  // Print trip.
  transit_realtime::TripDescriptor td;
  td.set_start_date("20230811");
  td.set_trip_id("TRIP_ADDED");
  td.set_start_time("10:00:00");
  auto const [r, t] = rt::gtfsrt_resolve_run(
      date::sys_days{2023_y / August / 11}, tt, &rtt, source_idx_t{0}, td);
  ASSERT_TRUE(r.valid());

  auto const fr = rt::frun{tt, &rtt, r};
  EXPECT_EQ(2, fr.size());
  auto ss = std::stringstream{};
  ss << "\n" << fr;
  EXPECT_EQ(expectedAdded, ss.str());
  EXPECT_EQ(nigiri::clasz::kBus, fr.get_clasz());
  ASSERT_FALSE(fr.is_cancelled());

  // TODO check all frun fields
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
  EXPECT_EQ(fr.size(), 3);
  auto ss = std::stringstream{};
  ss << "\n" << fr;
  EXPECT_EQ(expectedNew, ss.str());
  EXPECT_EQ(nigiri::clasz::kBus, fr.get_clasz());
  ASSERT_FALSE(fr.is_cancelled());

  for (auto const [from, to] : utl::pairwise(fr)) {
    EXPECT_EQ(from.id(), "E");
    EXPECT_EQ(to.id(), "B");
  }
  /*
  // Update again.
  auto const msg2 = rt::json_to_protobuf(kTripNewLonger);
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg2);
  auto ss2 = std::stringstream{};
  ss2 << "\n" << fr;
  EXPECT_EQ(1, rtt.rt_transport_location_seq_.size());
  EXPECT_EQ(expectedNewLonger, ss.str());*/
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
  EXPECT_EQ("New Route", fr.name());
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

  // Update.
  auto const msg = rt::json_to_protobuf(kTripNewBare);
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg);

  // Print trip.
  transit_realtime::TripDescriptor td;
  td.set_start_date("20230810");
  td.set_trip_id("TRIP_NEW");
  td.set_start_time("10:00:00");
  auto const [r, t] = rt::gtfsrt_resolve_run(
      date::sys_days{2023_y / August / 10}, tt, &rtt, source_idx_t{0}, td);
  ASSERT_FALSE(r.valid());
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