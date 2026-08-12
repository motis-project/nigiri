#include "gtest/gtest.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"

#include "../raptor_search.h"

#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace std::chrono_literals;
using nigiri::test::raptor_search;

namespace {

// A --T1--> B ~~fp 5min~~> B2 --T2--> D ~~fp 5min~~> D2 --T3--> C with
// wide transfer windows and a 10 minute via stay at D (the source of
// the second footpath): the footpath may only leave D after the stay is
// served, the wide windows make early vs. late footpath placement
// distinguishable, and three transit legs let the pong wait-minimization
// rewrite the middle legs.
mem_dir test_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Etc/UTC

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,1.0,2.0,,
B2,B2,,1.0,2.01,,
D,D,,1.5,3.0,,
D2,D2,,1.5,3.01,,
C,C,,2.0,4.0,,

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
B,B2,2,300
D,D2,2,300

# calendar_dates.txt
service_id,date,exception_type
S,20240619,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,T1,,,2
R2,DB,T2,,,2
R3,DB,T3,,,2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S,T1,,
R2,S,T2,,
R3,S,T3,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,10:00:00,10:00:00,A,1,0,0
T1,10:30:00,10:30:00,B,2,0,0
T2,11:00:00,11:00:00,B2,1,0,0
T2,11:30:00,11:30:00,D,2,0,0
T3,12:30:00,12:30:00,D2,1,0,0
T3,13:00:00,13:00:00,C,2,0,0
)");
}

std::string to_string(timetable const& tt,
                      pareto_set<routing::journey> const& results) {
  std::stringstream ss;
  ss << "\n";
  for (auto const& j : results) {
    j.print(ss, tt);
    ss << "\n";
  }
  return ss.str();
}

}  // namespace

TEST(routing, via_footpath_stay) {
  auto tt = timetable{};
  tt.date_range_ = {sys_days{2024_y / June / 18}, sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  auto const A = tt.find(location_id{"A", source_idx_t{0}}).value();
  auto const D = tt.find(location_id{"D", source_idx_t{0}}).value();
  auto const C = tt.find(location_id{"C", source_idx_t{0}}).value();

  auto const day = sys_days{2024_y / June / 19};
  auto const results =
      raptor_search(tt, nullptr,
                    routing::query{
                        .start_time_ = interval<unixtime_t>{day + 10h,
                                                            day + 10h + 15min},
                        .start_match_mode_ =
                            routing::location_match_mode::kEquivalent,
                        .dest_match_mode_ =
                            routing::location_match_mode::kEquivalent,
                        .use_start_footpaths_ = false,
                        .start_ = {{A, 0min, 0U}},
                        .destination_ = {{C, 0min, 0U}},
                        .via_stops_ = {{D, 10min}}},
                    direction::kForward);

  constexpr auto const expected = R"(
[2024-06-19 10:00, 2024-06-19 13:00]
TRANSFERS: 2
     FROM: (A, A) [2024-06-19 10:00]
       TO: (C, C) [2024-06-19 13:00]
leg 0: (A, A) [2024-06-19 10:00] -> (B, B) [2024-06-19 10:30]
   0: A       A...............................................                               d: 19.06 10:00 [19.06 10:00]  [{name=T1, day=2024-06-19, id=T1, src=0}]
   1: B       B............................................... a: 19.06 10:30 [19.06 10:30]
leg 1: (B, B) [2024-06-19 10:30] -> (B2, B2) [2024-06-19 10:35]
  FOOTPATH (duration=5)
leg 2: (B2, B2) [2024-06-19 11:00] -> (D, D) [2024-06-19 11:30]
   0: B2      B2..............................................                               d: 19.06 11:00 [19.06 11:00]  [{name=T2, day=2024-06-19, id=T2, src=0}]
   1: D       D............................................... a: 19.06 11:30 [19.06 11:30]
leg 3: (D, D) [2024-06-19 11:40] -> (D2, D2) [2024-06-19 11:45]
  FOOTPATH (duration=5)
leg 4: (D2, D2) [2024-06-19 12:30] -> (C, C) [2024-06-19 13:00]
   0: D2      D2..............................................                               d: 19.06 12:30 [19.06 12:30]  [{name=T3, day=2024-06-19, id=T3, src=0}]
   1: C       C............................................... a: 19.06 13:00 [19.06 13:00]

)";
  EXPECT_EQ(expected, to_string(tt, results));
}
