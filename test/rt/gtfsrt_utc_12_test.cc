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
  using std::filesystem::path;
  return {
      {{path{kAgencyFile},
        std::string{
            R"(agency_id,agency_name,agency_url,agency_timezone
NZ,Auckland,https://example.com,Pacific/Auckland
)"}},
       {path{kStopFile},
        std::string{
            R"(stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,
C,C,,4.0,5.0,,
D,D,,6.0,7.0,,
)"}},
       {path{kCalendarDatesFile}, std::string{R"(service_id,date,exception_type
S_RE1,20190503,1
S_RE1_summer,20191103,1
S_RE2,20190504,1
S_RE2_summer,20191104,1
)"}},
       {path{kRoutesFile},
        std::string{
            R"(route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R_RE1,DB,RE 1,,,3
R_RE2,DB,RE 2,,,3
)"}},
       {path{kTripsFile},
        std::string{R"(route_id,service_id,trip_id,trip_headsign,block_id
R_RE1,S_RE1,T_RE1,RE 1,
R_RE1,S_RE1_summer,T_RE1_summer,RE 1,
R_RE1,S_RE1,T_RE11,RE 1,
R_RE1,S_RE1,T_RE12,RE 1,
R_RE2,S_RE2,T_RE2,RE 2,
R_RE2,S_RE2_summer,T_RE2_summer,RE 2,
)"}},
       {path{kStopTimesFile},
        std::string{
            R"(trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T_RE1,13:00:00,13:00:00,A,1,0,0
T_RE1,13:30:00,13:30:00,B,2,0,0
T_RE1_summer,13:00:00,13:00:00,A,1,0,0
T_RE1_summer,13:30:00,13:30:00,B,2,0,0
T_RE11,42:00:00,42:00:00,A,1,0,0
T_RE11,43:00:00,43:00:00,B,2,0,0
T_RE12,33:00:00,33:00:00,A,1,0,0
T_RE12,43:00:00,43:00:00,B,2,0,0
T_RE2,00:30:00,00:30:00,B,1,0,0
T_RE2,00:45:00,00:45:00,C,2,0,0
T_RE2,12:30:00,12:30:00,D,3,0,0
T_RE2_summer,00:30:00,00:30:00,B,1,0,0
T_RE2_summer,00:45:00,00:45:00,C,2,0,0
T_RE2_summer,12:30:00,12:30:00,D,3,0,0
)"}}}};
}

constexpr auto const kTransport1AfterUpdate = std::string_view{
    R"(
   0: A       A...............................................                                                             d: 03.05 01:00 [03.05 13:00]  RT 03.05 01:01 [03.05 13:01]  [{name=RE 1, day=2019-05-03, id=T_RE1, src=0}]
   1: B       B............................................... a: 03.05 01:30 [03.05 13:30]  RT 03.05 01:31 [03.05 13:31]
)"};

constexpr auto const kTransport1SummerAfterUpdate = std::string_view{
    R"(
   0: A       A...............................................                                                             d: 03.11 00:00 [03.11 13:00]  RT 03.11 00:01 [03.11 13:01]  [{name=RE 1, day=2019-11-03, id=T_RE1_summer, src=0}]
   1: B       B............................................... a: 03.11 00:30 [03.11 13:30]  RT 03.11 00:31 [03.11 13:31]
)"};

constexpr auto const kTransport11AfterUpdate = std::string_view{
    R"(
   0: A       A...............................................                                                             d: 04.05 06:00 [04.05 18:00]  RT 04.05 06:00 [04.05 18:00]  [{name=RE 1, day=2019-05-04, id=T_RE11, src=0}]
   1: B       B............................................... a: 04.05 07:00 [04.05 19:00]  RT 04.05 07:00 [04.05 19:00]
)"};

constexpr auto const kTransport12AfterUpdate = std::string_view{
    R"(
   0: A       A...............................................                                                             d: 03.05 21:00 [04.05 09:00]  RT 03.05 21:00 [04.05 09:00]  [{name=RE 1, day=2019-05-03, id=T_RE12, src=0}]
   1: B       B............................................... a: 04.05 07:00 [04.05 19:00]  RT 04.05 07:00 [04.05 19:00]
)"};

constexpr auto const kTransport2AfterUpdate = std::string_view{
    R"(
   0: B       B...............................................                                                             d: 03.05 12:30 [04.05 00:30]  RT 03.05 12:30 [04.05 00:30]  [{name=RE 2, day=2019-05-03, id=T_RE2, src=0}]
   1: C       C............................................... a: 03.05 12:45 [04.05 00:45]  RT 03.05 12:45 [04.05 00:45]  d: 03.05 12:45 [04.05 00:45]  RT 03.05 12:45 [04.05 00:45]  [{name=RE 2, day=2019-05-03, id=T_RE2, src=0}]
   2: D       D............................................... a: 04.05 00:30 [04.05 12:30]  RT 04.05 00:30 [04.05 12:30]
)"};

constexpr auto const kTransport2SummerAfterUpdate = std::string_view{
    R"(
   0: B       B...............................................                                                             d: 03.11 11:30 [04.11 00:30]  RT 03.11 11:30 [04.11 00:30]  [{name=RE 2, day=2019-11-03, id=T_RE2_summer, src=0}]
   1: C       C............................................... a: 03.11 11:45 [04.11 00:45]  RT 03.11 11:45 [04.11 00:45]  d: 03.11 11:45 [04.11 00:45]  RT 03.11 11:45 [04.11 00:45]  [{name=RE 2, day=2019-11-03, id=T_RE2_summer, src=0}]
   2: D       D............................................... a: 03.11 23:30 [04.11 12:30]  RT 03.11 23:30 [04.11 12:30]
)"};

auto const kTripUpdate =
    R"({
"header": {
"gtfsRealtimeVersion": "2.0",
"incrementality": "FULL_DATASET",
"timestamp": "1691660324"
},
"entity": [
{
"id": "1",
"isDeleted": false,
"tripUpdate": {
 "trip": {
  "tripId": "T_RE1",
  "startTime": "13:00:00",
  "startDate": "20190503",
  "routeId": "T_RE1"
 },
 "stopTimeUpdate": [
  {
   "stopSequence": 1,
   "departure": {
    "delay": 60
   },
   "stopId": "A",
   "scheduleRelationship": "SCHEDULED"
  }
 ]
}
},
{
"id": "3",
"isDeleted": false,
"tripUpdate": {
 "trip": {
  "tripId": "T_RE11",
  "startTime": "42:00:00",
  "startDate": "20190503",
  "routeId": "T_RE11"
 },
 "stopTimeUpdate": [
  {
   "stopSequence": 1,
   "departure": {
    "delay": 0
   },
   "stopId": "A",
   "scheduleRelationship": "SCHEDULED"
  }
 ]
}
},
{
"id": "4",
"isDeleted": false,
"tripUpdate": {
 "trip": {
  "tripId": "T_RE12",
  "startTime": "33:00:00",
  "startDate": "20190503",
  "routeId": "T_RE12"
 },
 "stopTimeUpdate": [
  {
   "stopSequence": 1,
   "departure": {
    "delay": 0
   },
   "stopId": "A",
   "scheduleRelationship": "SCHEDULED"
  }
 ]
}
},
{
"id": "5",
"isDeleted": false,
"tripUpdate": {
 "trip": {
  "tripId": "T_RE2",
  "startTime": "00:30:00",
  "startDate": "20190504",
  "routeId": "T_RE2"
 },
 "stopTimeUpdate": [
  {
   "stopSequence": 1,
   "arrival": {
    "delay": 0
   },
   "departure": {
    "delay": 0
   },
   "stopId": "B",
   "scheduleRelationship": "SCHEDULED"
  }
 ]
}
},
]
})"s;

auto const kTripUpdateSummer =
    R"({
"header": {
"gtfsRealtimeVersion": "2.0",
"incrementality": "FULL_DATASET",
"timestamp": "1691660324"
},
"entity": [
{
"id": "2",
"isDeleted": false,
"tripUpdate": {
 "trip": {
  "tripId": "T_RE1_summer",
  "startTime": "13:00:00",
  "startDate": "20191103",
  "routeId": "T_RE1"
 },
 "stopTimeUpdate": [
  {
   "stopSequence": 1,
   "departure": {
    "delay": 60
   },
   "stopId": "A",
   "scheduleRelationship": "SCHEDULED"
  }
 ]
}
},
{
"id": "6",
"isDeleted": false,
"tripUpdate": {
 "trip": {
  "tripId": "T_RE2_summer",
  "startTime": "00:30:00",
  "startDate": "20191104",
  "routeId": "T_RE2"
 },
 "stopTimeUpdate": [
  {
   "stopSequence": 1,
   "arrival": {
    "delay": 0
   },
   "departure": {
    "delay": 0
   },
   "stopId": "B",
   "scheduleRelationship": "SCHEDULED"
  }
 ]
}
}
]
})"s;

TEST(rt, gtfs_rt_utc_12) {
  // Load static timetable.
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2019_y / May / 2},
                    date::sys_days{2019_y / November / 12}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  // Create empty RT timetable.
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2019_y / May / 3});

  // Update.
  auto const msg = rt::json_to_protobuf(kTripUpdate);
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg);

  {
    // Print trip.
    transit_realtime::TripDescriptor td;
    td.set_start_date("20190503");
    td.set_trip_id("T_RE1");
    td.set_start_time("13:00:00");
    auto const [r, t] = rt::gtfsrt_resolve_run(date::sys_days{2019_y / May / 3},
                                               tt, &rtt, source_idx_t{0}, td);
    ASSERT_TRUE(r.valid());

    auto const fr = rt::frun{tt, &rtt, r};
    auto ss = std::stringstream{};
    ss << "\n" << fr;
    EXPECT_EQ(kTransport1AfterUpdate, ss.str());
  }

  {
    // Print trip.
    transit_realtime::TripDescriptor td;
    td.set_start_date("20190503");
    td.set_trip_id("T_RE11");
    td.set_start_time("42:00:00");
    auto const [r, t] = rt::gtfsrt_resolve_run(date::sys_days{2019_y / May / 3},
                                               tt, &rtt, source_idx_t{0}, td);
    ASSERT_TRUE(r.valid());

    auto const fr = rt::frun{tt, &rtt, r};
    auto ss = std::stringstream{};
    ss << "\n" << fr;
    EXPECT_EQ(kTransport11AfterUpdate, ss.str());
  }

  {
    // Print trip.
    transit_realtime::TripDescriptor td;
    td.set_start_date("20190503");
    td.set_trip_id("T_RE12");
    td.set_start_time("33:00:00");
    auto const [r, t] = rt::gtfsrt_resolve_run(date::sys_days{2019_y / May / 3},
                                               tt, &rtt, source_idx_t{0}, td);
    ASSERT_TRUE(r.valid());

    auto const fr = rt::frun{tt, &rtt, r};
    auto ss = std::stringstream{};
    ss << "\n" << fr;
    EXPECT_EQ(kTransport12AfterUpdate, ss.str());
  }

  {
    // Print trip.
    transit_realtime::TripDescriptor td;
    td.set_start_date("20190504");
    td.set_trip_id("T_RE2");
    td.set_start_time("00:30:00");
    auto const [r, t] = rt::gtfsrt_resolve_run(date::sys_days{2019_y / May / 3},
                                               tt, &rtt, source_idx_t{0}, td);
    ASSERT_TRUE(r.valid());

    auto const fr = rt::frun{tt, &rtt, r};
    auto ss = std::stringstream{};
    ss << "\n" << fr;
    EXPECT_EQ(kTransport2AfterUpdate, ss.str());
  }
}

TEST(rt, gtfs_rt_utc_13) {
  // Load static timetable.
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2019_y / May / 2},
                    date::sys_days{2019_y / November / 12}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  // Create empty RT timetable.
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2019_y / November / 3});

  // Update.
  auto const msg = rt::json_to_protobuf(kTripUpdateSummer);
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg);

  {
    // Print trip.
    transit_realtime::TripDescriptor td;
    td.set_start_date("20191103");
    td.set_trip_id("T_RE1_summer");
    td.set_start_time("13:00:00");
    auto const [r, t] = rt::gtfsrt_resolve_run(
        date::sys_days{2019_y / November / 3}, tt, &rtt, source_idx_t{0}, td);
    ASSERT_TRUE(r.valid());

    auto const fr = rt::frun{tt, &rtt, r};
    auto ss = std::stringstream{};
    ss << "\n" << fr;
    EXPECT_EQ(kTransport1SummerAfterUpdate, ss.str());
  }

  {
    // Print trip.
    transit_realtime::TripDescriptor td;
    td.set_start_date("20191104");
    td.set_trip_id("T_RE2_summer");
    td.set_start_time("00:30:00");
    auto const [r, t] =
        rt::gtfsrt_resolve_run(date::sys_days{2019_y / November / 3}, tt,
                               nullptr, source_idx_t{0}, td);
    ASSERT_TRUE(r.valid());

    auto const fr = rt::frun{tt, &rtt, r};
    auto ss = std::stringstream{};
    ss << "\n" << fr;
    EXPECT_EQ(kTransport2SummerAfterUpdate, ss.str());
  }
}

}  // namespace
