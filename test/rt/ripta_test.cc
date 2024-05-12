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
  return mem_dir::read(R"__(
     "(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone,agency_lang,agency_phone
"RIPTA","Rhode Island Public Transit Authority",http://www.ripta.com/,America/New_York,en,401-781-9400

# stops.txt
stop_id,stop_code,stop_name,stop_desc,stop_lat,stop_lon,zone_id,stop_url,location_type,parent_station,stop_associated_place,wheelchair_boarding
  16630,016630,"Kennedy Plaza (Stop J)",,  41.825567, -71.411054,,,0,,kepl,0
  51165,051165,"Memorial after Exchange",,  41.826820, -71.412824,,,0,,,0
  16940,016940,"Francis after Finance",,  41.827697, -71.415246,,,0,,,0
  11240,011240,"Broad St Terminus (Montgomery Ave)",,  41.786282, -71.401726,,,0,,bdcl,0
  72000,072000,"Carr Street",,  41.789395, -71.404360,,,0,,,0
  11880,011880,"Roger Williams Park",,  41.792075, -71.406893,,,0,,,0

# calendar_dates.txt
service_id,date,exception_type
hjan2406-Saturday-4-JA2024-0000010,20240511,1

# routes.txt
route_id,route_short_name,route_long_name,route_desc,route_type,route_url,route_color,route_text_color
50,50,"Douglas Avenue",,3,http://www.ripta.com/50,E0004D,FFFFFF
11,R,"Broad/North Main",,3,http://ripta.com/11,00843D,FFFFFF

# trips.txt
route_id,service_id,trip_id,trip_headsign,direction_id,block_id,shape_id,trip_type,trip_footnote
50,hjan2406-Saturday-4-JA2024-0000010,3774835,"Bryant University",1,431352,500005,0,
11,hjan2406-Saturday-4-JA2024-0000010,3776558,"Slater Mill (Pawtucket)",0,431440,110035,0,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
3774835,22:15:00,22:15:00,  16630,1,0,0
3774835,22:15:00,22:15:00,  51165,2,0,0
3774835,22:17:00,22:17:00,  16940,3,0,0
3776558,24:17:00,24:17:00,  11240,1,0,0
3776558,24:18:00,24:18:00,  72000,2,0,0
3776558,24:19:00,24:19:00,  11880,3,0,0
)__");
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
      "tripId": "3774835",
      "startTime": "22:15:00",
      "startDate": "20240511"
     },
     "stopTimeUpdate": [
      {
       "stopSequence": 2,
       "scheduleRelationship": "SKIPPED"
      }
     ]
    }
  }
 ]
})"s;

auto const kTripUpdate1 =
    R"({
 "header": {
  "gtfsRealtimeVersion": "2.0",
  "incrementality": "FULL_DATASET",
  "timestamp": "1691660324"
 },
 "entity": [
  {
    "id": "3248652",
    "isDeleted": false,
    "tripUpdate": {
     "trip": {
      "tripId": "3776558",
      "startTime": "00:17:00",
      "startDate": "20240512"
     },
     "stopTimeUpdate": [
      {
       "stopSequence": 2,
       "scheduleRelationship": "SKIPPED"
      }
     ]
    }
  }
 ]
})"s;

}  // namespace

TEST(rt, gtfs_rt_ripta) {
  // Load static timetable.
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {date::sys_days{2024_y / May / 10},
                    date::sys_days{2024_y / May / 13}};
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  // Create empty RT timetable.
  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / May / 11});

  //  // Update.
  {
    auto const msg = rt::json_to_protobuf(kTripUpdate);
    gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg);

    // Print trip.
    transit_realtime::TripDescriptor td;
    td.set_start_date("20240511");
    td.set_trip_id("3774835");
    td.set_start_time("22:15:00");
    auto const [r, t] = rt::gtfsrt_resolve_run(
        date::sys_days{2023_y / November / 26}, tt, rtt, source_idx_t{0}, td);
    ASSERT_TRUE(r.valid());

    auto ss = std::stringstream{};
    ss << "\n" << rt::frun{tt, &rtt, r};
    std::cout << ss.str();
  }

  // Update 1.
  {
    auto const msg = rt::json_to_protobuf(kTripUpdate1);
    gtfsrt_update_buf(tt, rtt, source_idx_t{0}, "", msg);

    // Print trip.
    transit_realtime::TripDescriptor td;
    td.set_start_date("20240512");
    td.set_trip_id("3776558");
    td.set_start_time("00:17:00");
    auto const [r, t] = rt::gtfsrt_resolve_run(
        date::sys_days{2023_y / November / 26}, tt, rtt, source_idx_t{0}, td);
    ASSERT_TRUE(r.valid());

    auto ss = std::stringstream{};
    ss << "\n" << rt::frun{tt, &rtt, r};
    std::cout << ss.str();
  }
}