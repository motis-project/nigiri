#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"

#include "../raptor_search.h"
#include "results_to_string.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace std::chrono_literals;
using nigiri::test::raptor_search;

namespace {

mem_dir test_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,

# calendar_dates.txt
service_id,date,exception_type
S_RE1,20190501,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R_RE1,DB,RE 1,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R_RE1,S_RE1,T_RE1,RE 1,
R_RE1,S_RE1,T_RE2,RE 1,
R_RE1,S_RE1,T_RE3,RE 1,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T_RE1,12:00:00,12:00:00,A,1,0,0
T_RE1,13:00:00,13:00:00,B,2,0,0
T_RE2,13:00:00,13:00:00,A,1,0,0
T_RE2,14:00:00,14:00:00,B,2,0,0
T_RE3,14:00:00,14:00:00,A,1,0,0
T_RE3,15:00:00,15:00:00,B,2,0,0
)");
}

constexpr auto const expected_journeys = R"(
[2019-05-01 10:00, 2019-05-01 11:00]
TRANSFERS: 0
     FROM: (A, A) [2019-05-01 10:00]
       TO: (B, B) [2019-05-01 11:00]
leg 0: (A, A) [2019-05-01 10:00] -> (B, B) [2019-05-01 11:00]
   0: A       A...............................................                               d: 01.05 10:00 [01.05 12:00]  [{name=RE 1, day=2019-05-01, id=T_RE1, src=0}]
   1: B       B............................................... a: 01.05 11:00 [01.05 13:00]

[2019-05-01 11:00, 2019-05-01 12:00]
TRANSFERS: 0
     FROM: (A, A) [2019-05-01 11:00]
       TO: (B, B) [2019-05-01 12:00]
leg 0: (A, A) [2019-05-01 11:00] -> (B, B) [2019-05-01 12:00]
   0: A       A...............................................                               d: 01.05 11:00 [01.05 13:00]  [{name=RE 1, day=2019-05-01, id=T_RE2, src=0}]
   1: B       B............................................... a: 01.05 12:00 [01.05 14:00]

)";

}  // namespace

TEST(routing, max_interval) {
  timetable tt;
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  auto const src = source_idx_t{0};
  load_timetable({}, src, test_files(), tt);
  finalize(tt);

  auto const results = raptor_search(
      tt, nullptr,
      nigiri::routing::query{
          .start_time_ =
              interval{unixtime_t{date::sys_days{2019_y / May / 1} + 10_hours},
                       unixtime_t{date::sys_days{2019_y / May / 1} + 11_hours}},
          .start_ = {{tt.locations_.location_id_to_idx_.at(
                          {.id_ = "A", .src_ = src}),
                      0_minutes, 0U}},
          .destination_ = {{tt.locations_.location_id_to_idx_.at(
                                {.id_ = "B", .src_ = src}),
                            0_minutes, 0U}},
          .min_connection_count_ = 3U,
          .extend_interval_later_ = true,
          .max_interval_ = interval{
              unixtime_t{date::sys_days{2019_y / May / 1} + 10_hours},
              unixtime_t{date::sys_days{2019_y / May / 1} + 12_hours}}});

  EXPECT_EQ(expected_journeys, to_string(tt, nullptr, results));
}