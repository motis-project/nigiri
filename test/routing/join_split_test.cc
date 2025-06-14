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
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station
"8507290:0:6","Zweisimmen","46.55343812","7.37490797","","Parent8507290"
"8507292:0:2","Boltigen","46.62736961","7.39007153","","Parent8507292"
"8507284","Enge im Simmental","46.64753262","7.41762286","","Parent8507284"
"Parent8507284","Enge im Simmental","46.64753262","7.41762286","1",""
"8507293","Oberwil im Simmental","46.65613462","7.43548137","","Parent8507293"
"Parent8507293","Oberwil im Simmental","46.65613462","7.43548137","1",""
"8507294","Weissenburg","46.65929759","7.47592352","","Parent8507294"
"Parent8507294","Weissenburg","46.65929759","7.47592352","1",""
"8507295:0:2","Därstetten","46.65882901","7.49564155","","Parent8507295"
"8507286","Ringoldingen","46.65755890","7.52457628","","Parent8507286"
"Parent8507286","Ringoldingen","46.65755890","7.52457628","1",""
"8507296:0:1","Erlenbach im Simmental","46.65916195","7.55560409","","Parent8507296"
"8507297:0:1","Oey-Diemtigen","46.65990180","7.57940046","","Parent8507297"
"8507287","Burgholz","46.66575243","7.60323277","","Parent8507287"
"Parent8507287","Burgholz","46.66575243","7.60323277","1",""
"8507298:0:1","Wimmis","46.67568904","7.63555415","","Parent8507298"
"8507288","Eifeld","46.68386750","7.64227355","","Parent8507288"
"Parent8507288","Eifeld","46.68386750","7.64227355","1",""
"8507299","Lattigen bei Spiez","46.69113894","7.65026856","","Parent8507299"
"Parent8507299","Lattigen bei Spiez","46.69113894","7.65026856","1",""
"8507483:0:5","Spiez","46.68641878","7.67947279","","Parent8507483"

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
"TA+cqfp0","1","1","0","0","1","0","0","20241215","20251213"

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
"91-11-N-j25-1","33","R11","","R","106"

# trips.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
"91-11-N-j25-1","TA+cqfp0","59.TA.91-11-N-j25-1.6.H","Spiez","6832","0","","ch:1:sjyid:100015:6832-001","NF"

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
"59.TA.91-11-N-j25-1.6.H","18:02:00","18:02:00","8507290:0:6","1","0","0"
"59.TA.91-11-N-j25-1.6.H","18:10:00","18:10:00","8507292:0:2","2","0","0"
"59.TA.91-11-N-j25-1.6.H","18:13:00","18:13:00","8507284","3","0","0"
"59.TA.91-11-N-j25-1.6.H","18:16:00","18:16:00","8507293","4","0","0"
"59.TA.91-11-N-j25-1.6.H","18:20:00","18:20:00","8507294","5","0","0"
"59.TA.91-11-N-j25-1.6.H","18:22:00","18:22:00","8507295:0:2","6","0","0"
"59.TA.91-11-N-j25-1.6.H","18:25:00","18:25:00","8507286","7","0","0"
"59.TA.91-11-N-j25-1.6.H","18:30:00","18:30:00","8507296:0:1","8","0","0"
"59.TA.91-11-N-j25-1.6.H","18:33:00","18:33:00","8507297:0:1","9","0","0"
"59.TA.91-11-N-j25-1.6.H","18:35:00","18:35:00","8507287","10","0","0"
"59.TA.91-11-N-j25-1.6.H","18:40:00","18:40:00","8507298:0:1","11","0","0"
"59.TA.91-11-N-j25-1.6.H","18:41:00","18:41:00","8507288","12","0","0"
"59.TA.91-11-N-j25-1.6.H","18:43:00","18:43:00","8507299","13","0","0"
"59.TA.91-11-N-j25-1.6.H","18:47:00","18:47:00","8507483:0:5","14","0","0"
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

TEST(routing, join_split) {
  timetable tt;
  tt.date_range_ = {date::sys_days{2025_y / June / 1},
                    date::sys_days{2025_y / June / 30}};
  auto const src = source_idx_t{0};
  load_timetable({}, src, test_files(), tt);
  finalize(tt);

  auto const results = raptor_search(
      tt, nullptr,
      nigiri::routing::query{
          .start_time_ =
              interval{
                  unixtime_t{date::sys_days{2025_y / June / 1} + 10_hours},
                  unixtime_t{date::sys_days{2025_y / June / 1} + 11_hours}},
          .start_ = {{tt.locations_.location_id_to_idx_.at(
                          {.id_ = "A", .src_ = src}),
                      0_minutes, 0U}},
          .destination_ = {{tt.locations_.location_id_to_idx_.at(
                                {.id_ = "B", .src_ = src}),
                            0_minutes, 0U}},
          .min_connection_count_ = 3U,
          .extend_interval_later_ = true,
          .max_interval_ = interval{
              unixtime_t{date::sys_days{2025_y / June / 1} + 10_hours},
              unixtime_t{date::sys_days{2025_y / June / 1} + 12_hours}}});

  EXPECT_EQ(expected_journeys, to_string(tt, nullptr, results));
}