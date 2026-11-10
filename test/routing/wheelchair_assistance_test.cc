#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/load.h"
#include "nigiri/routing/raptor/mcraptor.h"
#include "nigiri/routing/raptor_search.h"
#include "nigiri/routing/search.h"

#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"
#include "../raptor_search.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace std::chrono_literals;
using nigiri::test::raptor_search;

namespace {

constexpr auto const kAssistance = R"(name,lat,lng,time
A,0.0,1.0,08:00-22:00
B,2.0,3.0,08:00-22:00
C,4.0,5.0,"Th; 08:00-22:00"
)";

// 00:00
// A -- B -- C  01:00
//      +---C   00:50
mem_dir test_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B1,B1,,2.0,3.0,,
B2,B2,,2.0,3.0,,
C,C,,4.0,5.0,,
D,D,,6.0,7.0,,

# calendar_dates.txt
service_id,date,exception_type
S,20240619,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,RE 1,,,101
R2,DB,RE 2,,,101

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id,wheelchair_accessible
R1,S,T1,RE 1,,1
R2,S,T2,RE 2,,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,00:00:00,00:00:00,A,1,0,0
T1,00:30:00,00:30:00,B1,2,0,0
T1,01:10:00,01:10:00,C,3,0,0
T2,00:40:00,00:40:00,B2,1,0,0
T2,00:50:00,00:50:00,C,2,0,0

# frequencies.txt
trip_id,start_time,end_time,headway_secs
T1,00:00:00,24:00:00,3600
T2,00:40:00,24:40:00,3600
)");
}

}  // namespace

TEST(routing, wheelchair_assistance) {
  auto assistance = read_assistance(kAssistance);

  auto tt = loader::load(test_files(), {2024_y / June / 18, 2024_y / June / 20},
                         {}, &assistance);

  tt.fwd_search_lb_graph_[kWheelchairProfile] =
      tt.fwd_search_lb_graph_[kDefaultProfile];
  tt.bwd_search_lb_graph_[kWheelchairProfile] =
      tt.bwd_search_lb_graph_[kDefaultProfile];

  auto const B1 = tt.find(location_id{"B1", {}}).value();
  auto const B2 = tt.find(location_id{"B2", {}}).value();
  for (auto const profile : {0U, 2U}) {
    tt.locations_.footpaths_out_[profile].resize(tt.n_locations());
    tt.locations_.footpaths_in_[profile].resize(tt.n_locations());
    tt.locations_.footpaths_out_[profile][B1].push_back(footpath{B2, 5min});
    tt.locations_.footpaths_in_[profile][B2].push_back(footpath{B1, 5min});
  }

  auto const iv = interval{unixtime_t{sys_days{2024_y / June / 19} + 5_hours},
                           unixtime_t{sys_days{2024_y / June / 19} + 21_hours}};

  auto const results_walk =
      raptor_search(tt, nullptr, "A", "C", iv, direction::kForward,
                    routing::all_clasz_allowed(), false, false, false, 0U);
  ASSERT_FALSE(results_walk.begin() == results_walk.end());
  EXPECT_EQ((unixtime_t{sys_days{2024_y / June / 19} + 5_hours}),
            results_walk.begin()->start_time_);
  EXPECT_EQ((unixtime_t{sys_days{2024_y / June / 19} + 20_hours}),
            std::prev(results_walk.end())->start_time_);

  auto const results_wheelchair =
      raptor_search(tt, nullptr, "A", "C", iv, direction::kForward,
                    routing::all_clasz_allowed(), false, false, false, 2U);
  ASSERT_FALSE(results_wheelchair.begin() == results_wheelchair.end());
  EXPECT_EQ((unixtime_t{sys_days{2024_y / June / 19} + 6_hours}),
            results_wheelchair.begin()->start_time_);
  EXPECT_EQ((unixtime_t{sys_days{2024_y / June / 19} + 19_hours}),
            std::prev(results_wheelchair.end())->start_time_);

  auto td = transit_realtime::TripDescriptor();
  *td.mutable_start_time() = "23:00:00";
  *td.mutable_start_date() = "20240619";
  *td.mutable_trip_id() = "T1";

  auto rtt = rt::create_rt_timetable(tt, date::sys_days{2024_y / June / 19});
  auto const [r, t] = rt::gtfsrt_resolve_run(date::sys_days{2019_y / May / 4},
                                             tt, &rtt, source_idx_t{0}, td);
  EXPECT_TRUE(r.valid());

  auto const fr = rt::frun{tt, &rtt, r};
  ASSERT_EQ(3U, fr.size());
  ASSERT_TRUE(fr[2].out_allowed_wheelchair());
}

namespace {

// A -> D: FAST_D (10:30) is not accessible, SLOW_D (11:30) is.
// A -> B -> C: BLOCK_AB + BLOCK_BC run through (one block), accessible to B
// only; SLOW_C (11:00) is the accessible way to C.
mem_dir wheelchair_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
MTA,MOTIS Transit Authority,https://motis-project.de/,Etc/UTC

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,0.1,1.1,,
C,C,,0.2,1.2,,
D,D,,0.3,1.3,,

# calendar_dates.txt
service_id,date,exception_type
S,20240619,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R,MTA,R,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id,wheelchair_accessible
R,S,FAST_D,,,2
R,S,SLOW_D,,,1
R,S,BLOCK_AB,,BL,1
R,S,BLOCK_BC,,BL,2
R,S,SLOW_C,,,1

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
FAST_D,10:00:00,10:00:00,A,1,0,0
FAST_D,10:30:00,10:30:00,D,2,0,0
SLOW_D,10:00:00,10:00:00,A,1,0,0
SLOW_D,11:30:00,11:30:00,D,2,0,0
BLOCK_AB,10:05:00,10:05:00,A,1,0,0
BLOCK_AB,10:20:00,10:20:00,B,2,0,0
BLOCK_BC,10:20:00,10:20:00,B,1,0,0
BLOCK_BC,10:40:00,10:40:00,C,2,0,0
SLOW_C,10:00:00,10:00:00,A,1,0,0
SLOW_C,11:00:00,11:00:00,C,2,0,0
)");
}

std::vector<std::string> dep_arr(pareto_set<routing::journey> const& js) {
  auto v = std::vector<std::string>{};
  for (auto const& j : js) {
    v.push_back(date::format("%H:%M", j.departure_time()) + " " +
                date::format("%H:%M", j.arrival_time()));
  }
  std::sort(begin(v), end(v));
  return v;
}

}  // namespace

// Wheelchair queries (profile 2) skip trips that are not accessible and
// sections that are not (CPU raptor's route_flag filter), in every engine.
TEST(routing, wheelchair_accessible_trips) {
  auto tt = loader::load(wheelchair_files(),
                         {2024_y / June / 18, 2024_y / June / 20});
  tt.fwd_search_lb_graph_[kWheelchairProfile] =
      tt.fwd_search_lb_graph_[kDefaultProfile];
  tt.bwd_search_lb_graph_[kWheelchairProfile] =
      tt.bwd_search_lb_graph_[kDefaultProfile];
  tt.locations_.footpaths_out_[kWheelchairProfile].resize(tt.n_locations());
  tt.locations_.footpaths_in_[kWheelchairProfile].resize(tt.n_locations());

  auto const day = sys_days{2024_y / June / 19};
  auto const at = [&](auto const h, auto const m) {
    return unixtime_t{day + std::chrono::hours{h} + std::chrono::minutes{m}};
  };
  // forward: departures at A from 09:30 to 10:30
  // backward: arrivals at the destination within `arrs`
  auto const check = [&](rt_timetable const* rtt, char const* to,
                         interval<unixtime_t> const arrs,
                         profile_idx_t const prf,
                         std::vector<std::string> const& expected) {
    auto const fwd = raptor_search(
        tt, rtt, "A", to, interval<unixtime_t>{at(9, 30), at(10, 30)},
        direction::kForward, routing::all_clasz_allowed(), false, false, false,
        prf);
    EXPECT_EQ(expected, dep_arr(fwd)) << to << " forward, profile " << prf;
    auto const bwd =
        raptor_search(tt, rtt, to, "A", arrs, direction::kBackward,
                      routing::all_clasz_allowed(), false, false, false, prf);
    EXPECT_EQ(expected, dep_arr(bwd)) << to << " backward, profile " << prf;

    // test::raptor_search only requires McRAPTOR to find a superset on the
    // CPU: check it directly
    for (auto const dir : {direction::kForward, direction::kBackward}) {
      auto q = routing::query{};
      q.start_ = {{tt.find(location_id{"A", {}}).value(), 0min, 0U}};
      q.destination_ = {{tt.find(location_id{to, {}}).value(), 0min, 0U}};
      q.prf_idx_ = prf;
      if (dir == direction::kBackward) {
        q.flip_dir();
      }
      q.start_time_ = dir == direction::kForward
                          ? interval<unixtime_t>{at(9, 30), at(10, 30)}
                          : arrs;
      auto ss = routing::search_state{};
      auto plain = routing::mcraptor_state{};
      EXPECT_EQ(
          expected,
          dep_arr(
              *routing::raptor_search(tt, rtt, ss, plain, q, dir).journeys_))
          << to << " mcraptor " << (dir == direction::kForward ? "fwd" : "bwd");
      auto all =
          routing::mcraptor_non_transit_mode_filter_mode_switches_state{};
      EXPECT_EQ(
          expected,
          dep_arr(*routing::raptor_search(tt, rtt, ss, all, q, dir).journeys_))
          << to << " mcraptor all criteria "
          << (dir == direction::kForward ? "fwd" : "bwd");
    }
  };

  // the timetable is built as intended: without wheelchair, the fast trip and
  // the through run win
  check(nullptr, "D", {at(10, 0), at(11, 0)}, kDefaultProfile, {"10:00 10:30"});
  check(nullptr, "C", {at(10, 30), at(10, 50)}, kDefaultProfile,
        {"10:05 10:40"});

  // route level: FAST_D is not accessible
  check(nullptr, "D", {at(11, 0), at(12, 0)}, kWheelchairProfile,
        {"10:00 11:30"});
  // section level: the through run is accessible to B, not to C
  check(nullptr, "B", {at(10, 0), at(10, 40)}, kWheelchairProfile,
        {"10:05 10:20"});
  check(nullptr, "C", {at(10, 30), at(11, 30)}, kWheelchairProfile,
        {"10:00 11:00"});

  // rt: a delayed FAST_D is an rt transport, still not accessible
  auto rtt = rt::create_rt_timetable(tt, day);
  auto msg = transit_realtime::FeedMessage{};
  auto const hdr = msg.mutable_header();
  hdr->set_gtfs_realtime_version("2.0");
  hdr->set_incrementality(
      transit_realtime::FeedHeader_Incrementality_FULL_DATASET);
  hdr->set_timestamp(std::chrono::duration_cast<std::chrono::seconds>(
                         (day + 9h).time_since_epoch())
                         .count());
  auto const e = msg.add_entity();
  e->set_id("1");
  auto const td = e->mutable_trip_update()->mutable_trip();
  td->set_start_time("10:00:00");
  td->set_start_date("20240619");
  td->set_trip_id("FAST_D");
  auto const stop_update = e->mutable_trip_update()->add_stop_time_update();
  stop_update->set_stop_sequence(1U);
  stop_update->mutable_departure()->set_delay(5 * 60);
  ASSERT_EQ(1U, rt::gtfsrt_update_msg(tt, rtt, source_idx_t{0}, "tag", msg)
                    .total_entities_success_);
  check(&rtt, "D", {at(10, 0), at(11, 0)}, kDefaultProfile, {"10:05 10:35"});
  check(&rtt, "D", {at(11, 0), at(12, 0)}, kWheelchairProfile, {"10:00 11:30"});
}
