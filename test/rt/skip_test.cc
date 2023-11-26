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
A,A,1.0,1.0
B,B,2.0,2.0
C,C,3.0,3.0
D,D,4.0,4.0
E,E,5.0,5.0
F,F,6.0,6.0

# calendar_dates.txt
service_id,date,exception_type
SERVICE_1,20231126,1

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
      "tripId": "TRIP_1",
      "startTime": "10:00:00",
      "startDate": "20231126"
     },
     "stopTimeUpdate": [
      {
       "stopSequence": 1,
       "scheduleRelationship": "SKIPPED"
      },
      {
       "stopSequence": 2,
       "scheduleRelationship": "SKIPPED"
      },
      {
       "stopSequence": 4,
       "scheduleRelationship": "SKIPPED"
      },
      {
       "stopSequence": 6,
       "scheduleRelationship": "SKIPPED"
      }
     ]
    }
  }
 ]
})"s;

constexpr auto const expected = R"(
   2: C       C............................................... a: 26.11 11:00 [26.11 12:00]  RT 26.11 11:00 [26.11 12:00]  d: 26.11 11:00 [26.11 12:00]  RT 26.11 11:00 [26.11 12:00]  [{name=Bus Route 1, day=2023-11-26, id=TRIP_1, src=0}]
   4: E       E............................................... a: 26.11 13:00 [26.11 14:00]  RT 26.11 13:00 [26.11 14:00]  d: 26.11 13:00 [26.11 14:00]  RT 26.11 13:00 [26.11 14:00]  [{name=Bus Route 1, day=2023-11-26, id=TRIP_1, src=0}]
)"sv;

}  // namespace

TEST(rt, gtfs_rt_skip) {
  // Load static timetable.
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2023_y / November / 25},
                    date::sys_days{2023_y / November / 27}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  // Create empty RT timetable.
  auto rtt =
      rt::create_rt_timetable(tt, date::sys_days{2023_y / November / 26});

  // Update.
  auto const msg = test::json_to_protobuf(kTripUpdate);
  gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg);

  // Print trip.
  transit_realtime::TripDescriptor td;
  td.set_start_date("20231126");
  td.set_trip_id("TRIP_1");
  td.set_start_time("10:00:00");
  auto const [r, t] = rt::gtfsrt_resolve_run(
      date::sys_days{2023_y / November / 26}, tt, rtt, source_idx_t{0}, td);
  ASSERT_TRUE(r.valid());

  auto ss = std::stringstream{};
  ss << "\n" << rt::frun{tt, &rtt, r};
  EXPECT_EQ(expected, ss.str());

  for (auto const [from, to] : utl::pairwise(rt::frun{tt, &rtt, r})) {
    EXPECT_EQ(from.id(), "C");
    EXPECT_EQ(to.id(), "E");
  }
}