#include "gtest/gtest.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"

#include "nigiri/rt/frun.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "../raptor_search.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace std::chrono_literals;
using nigiri::test::raptor_search;

namespace {

// A --T1--> B --T2--> C --T3--> D --T4--> E, all transfers at the same
// stops. T2 serves V in the middle (B -> V -> C), the earlier express T2X
// runs B -> C directly. With a 0 minute via at V, wait minimization must
// not replace T2 by T2X: the express skips the via.
mem_dir test_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Etc/UTC

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,1.0,2.0,,
V,V,,1.5,2.5,,
C,C,,2.0,3.0,,
D,D,,2.5,3.5,,
E,E,,3.0,4.0,,

# calendar_dates.txt
service_id,date,exception_type
S,20240619,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,T1,,,2
R2,DB,T2,,,2
R2X,DB,T2X,,,2
R3,DB,T3,,,2
R4,DB,T4,,,2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S,T1,,
R2,S,T2,,
R2X,S,T2X,,
R3,S,T3,,
R4,S,T4,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,10:00:00,10:00:00,A,1,0,0
T1,10:30:00,10:30:00,B,2,0,0
T2,10:40:00,10:40:00,B,1,0,0
T2,10:50:00,10:50:00,V,2,0,0
T2,11:00:00,11:00:00,C,3,0,0
T2X,10:35:00,10:35:00,B,1,0,0
T2X,10:55:00,10:55:00,C,2,0,0
T3,11:30:00,11:30:00,C,1,0,0
T3,12:00:00,12:00:00,D,2,0,0
T4,12:30:00,12:30:00,D,1,0,0
T4,13:00:00,13:00:00,E,2,0,0
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

TEST(routing, via_earliest_alternative) {
  auto tt = timetable{};
  tt.date_range_ = {sys_days{2024_y / June / 18}, sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  auto const A = tt.find(location_id{"A", source_idx_t{0}}).value();
  auto const V = tt.find(location_id{"V", source_idx_t{0}}).value();
  auto const E = tt.find(location_id{"E", source_idx_t{0}}).value();

  auto const day = sys_days{2024_y / June / 19};
  auto const results = raptor_search(
      tt, nullptr,
      routing::query{
          .start_time_ = interval<unixtime_t>{day + 10h, day + 10h + 15min},
          .start_match_mode_ = routing::location_match_mode::kEquivalent,
          .dest_match_mode_ = routing::location_match_mode::kEquivalent,
          .use_start_footpaths_ = false,
          .start_ = {{A, 0min, 0U}},
          .destination_ = {{E, 0min, 0U}},
          .via_stops_ = {{V, 0min}}},
      direction::kForward);

  ASSERT_EQ(1U, results.size());
  // the journey has to serve the 0 minute via V: only T2 does, the
  // earlier express T2X (which the via-blind wait minimization picks)
  // skips it
  auto serves_via = false;
  for (auto const& l : results.begin()->legs_) {
    if (std::holds_alternative<routing::journey::run_enter_exit>(l.uses_)) {
      auto const ree = std::get<routing::journey::run_enter_exit>(l.uses_);
      auto const fr = rt::frun{tt, nullptr, ree.r_};
      for (auto i = ree.stop_range_.from_; i != ree.stop_range_.to_; ++i) {
        serves_via |= fr[i].get_location_idx() == V;
      }
    }
  }
  EXPECT_TRUE(serves_via) << to_string(tt, results);
}

namespace {

// A --T1--> B (via V mid-route) ~~fp~~> B2 --T2--> C ~~fp~~> C2 --T3--> D
// the delivered journey uses T2 (latest departure, bwd pass) which
// passes V again mid-route; the earlier-departing, earlier-arriving
// express T2X skips it. V is already credited on T1, so the wait
// minimization may replace T2 by T2X.
mem_dir completed_via_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Etc/UTC

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
V,V,,0.5,1.5,,
B,B,,1.0,2.0,,
B2,B2,,1.0,2.01,,
C,C,,2.0,3.0,,
C2,C2,,2.0,3.01,,
D,D,,2.5,3.5,,

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
B,B2,2,300
C,C2,2,300

# calendar_dates.txt
service_id,date,exception_type
S,20240619,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,T1,,,2
R2,DB,T2,,,2
R2X,DB,T2X,,,2
R3,DB,T3,,,2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S,T1,,
R2,S,T2,,
R2X,S,T2X,,
R3,S,T3,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,10:00:00,10:00:00,A,1,0,0
T1,10:15:00,10:15:00,V,2,0,0
T1,10:30:00,10:30:00,B,3,0,0
T2,11:20:00,11:20:00,B2,1,0,0
T2,11:30:00,11:30:00,V,2,1,0
T2,11:40:00,11:40:00,C,3,0,0
T2X,11:00:00,11:00:00,B2,1,0,0
T2X,11:39:00,11:39:00,C,2,0,0
T3,12:30:00,12:30:00,C2,1,0,0
T3,13:00:00,13:00:00,D,2,0,0
)");
}

}  // namespace

TEST(routing, via_earliest_alternative_completed_via) {
  auto tt = timetable{};
  tt.date_range_ = {sys_days{2024_y / June / 18}, sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, completed_via_files(), tt);
  finalize(tt);

  auto const A = tt.find(location_id{"A", source_idx_t{0}}).value();
  auto const B2 = tt.find(location_id{"B2", source_idx_t{0}}).value();
  auto const V = tt.find(location_id{"V", source_idx_t{0}}).value();
  auto const D = tt.find(location_id{"D", source_idx_t{0}}).value();

  auto const day = sys_days{2024_y / June / 19};
  auto q = routing::query{
      .start_time_ = interval<unixtime_t>{day + 10h, day + 10h + 15min},
      .start_match_mode_ = routing::location_match_mode::kEquivalent,
      .dest_match_mode_ = routing::location_match_mode::kEquivalent,
      .use_start_footpaths_ = false,
      .start_ = {{A, 0min, 0U}},
      .destination_ = {{D, 0min, 0U}},
      .via_stops_ = {{V, 0min}}};

  auto const results =
      raptor_search(tt, nullptr, std::move(q), direction::kForward);

  ASSERT_EQ(1U, results.size());
  // V is already credited on T1 -> the earlier express T2X (dep 11:00)
  // may replace T2 even though it skips V
  auto found = false;
  for (auto const& l : results.begin()->legs_) {
    if (l.from_ == B2 &&
        std::holds_alternative<routing::journey::run_enter_exit>(l.uses_)) {
      EXPECT_EQ(day + 11h, l.dep_time_) << "spurious via requirement kept T2?";
      found = true;
    }
  }
  EXPECT_TRUE(found);
}
