#include <string_view>

#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/component_graph.h"
#include "nigiri/routing/limits.h"
#include "nigiri/timetable.h"

#include "../raptor_search.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::routing;

namespace {

// Network (all trips daily 2024-06-08):
//   R1: A -> B -> C        (dep 09:00, B 09:30/09:32, C 10:00; hourly x4)
//   R2: B2 -> D            (dep 09:40, D 10:10; hourly x4)
//   R3: A -> D             (dep 09:05, D 11:05, slow direct; once)
//   footpath B <-> B2 (5 min) -> B, B2 = one component
//   C1 child of C (parent station) -> C, C1 = one component
//   R4: C1 -> E            (dep 10:15, E 10:45; hourly x4)
mem_dir component_files() {
  return mem_dir::read(R"__(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
MTA,MOTIS Transit Authority,https://motis-project.de/,Europe/Berlin

# calendar_dates.txt
service_id,date,exception_type
S,20240608,1

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,1.0,1.0,,0,
B,B,,2.0,2.0,,0,
B2,B2,,2.001,2.001,,0,
C,C,,3.0,3.0,,1,
C1,C1,,3.0,3.0,,0,C
D,D,,4.0,4.0,,0,
E,E,,5.0,5.0,,0,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,MTA,R1,,,2
R2,MTA,R2,,,2
R3,MTA,R3,,,2
R4,MTA,R4,,,2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S,T1a,,
R1,S,T1b,,
R1,S,T1c,,
R1,S,T1d,,
R2,S,T2a,,
R2,S,T2b,,
R2,S,T2c,,
R2,S,T2d,,
R3,S,T3,,
R4,S,T4a,,
R4,S,T4b,,
R4,S,T4c,,
R4,S,T4d,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1a,09:00:00,09:00:00,A,1,0,0
T1a,09:30:00,09:32:00,B,2,0,0
T1a,10:00:00,10:00:00,C1,3,0,0
T1b,10:00:00,10:00:00,A,1,0,0
T1b,10:30:00,10:32:00,B,2,0,0
T1b,11:00:00,11:00:00,C1,3,0,0
T1c,11:00:00,11:00:00,A,1,0,0
T1c,11:30:00,11:32:00,B,2,0,0
T1c,12:00:00,12:00:00,C1,3,0,0
T1d,12:00:00,12:00:00,A,1,0,0
T1d,12:30:00,12:32:00,B,2,0,0
T1d,13:00:00,13:00:00,C1,3,0,0
T2a,09:40:00,09:40:00,B2,1,0,0
T2a,10:10:00,10:10:00,D,2,0,0
T2b,10:40:00,10:40:00,B2,1,0,0
T2b,11:10:00,11:10:00,D,2,0,0
T2c,11:40:00,11:40:00,B2,1,0,0
T2c,12:10:00,12:10:00,D,2,0,0
T2d,12:40:00,12:40:00,B2,1,0,0
T2d,13:10:00,13:10:00,D,2,0,0
T3,09:05:00,09:05:00,A,1,0,0
T3,11:05:00,11:05:00,D,2,0,0
T4a,10:15:00,10:15:00,C1,1,0,0
T4a,10:45:00,10:45:00,E,2,0,0
T4b,11:15:00,11:15:00,C1,1,0,0
T4b,11:45:00,11:45:00,E,2,0,0
T4c,12:15:00,12:15:00,C1,1,0,0
T4c,12:45:00,12:45:00,E,2,0,0
T4d,13:15:00,13:15:00,C1,1,0,0
T4d,13:45:00,13:45:00,E,2,0,0

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
B,B2,2,300
)__");
}

timetable load_tt() {
  auto tt = timetable{};
  tt.date_range_ = {2024_y / June / 7, 2024_y / June / 9};
  register_special_stations(tt);
  gtfs::load_timetable({}, source_idx_t{0}, component_files(), tt);
  finalize(tt);
  return tt;
}

location_idx_t find_loc(timetable const& tt, char const* id) {
  return tt.locations_.location_id_to_idx_.at({id, source_idx_t{0}});
}

}  // namespace

TEST(routing, component_graph_structure) {
  auto const tt = load_tt();
  auto const g = build_component_graph(tt);

  auto const c = [&](char const* id) {
    return g.location_component_[find_loc(tt, id)];
  };

  // footpath-connected + parent/child stops collapse
  EXPECT_EQ(c("B"), c("B2"));
  EXPECT_EQ(c("C"), c("C1"));

  // distinct otherwise
  EXPECT_NE(c("A"), c("B"));
  EXPECT_NE(c("A"), c("D"));
  EXPECT_NE(c("B"), c("C"));
  EXPECT_NE(c("C"), c("E"));

  EXPECT_EQ(tt.n_locations(), g.location_component_.size());
  EXPECT_GT(g.n_components_, 0U);
  EXPECT_EQ(g.seqs_.size(), g.durations_.size());
  for (auto cr = comp_route_idx_t{0U}; cr != g.seqs_.size(); ++cr) {
    ASSERT_GE(g.seqs_[cr].size(), 2U);
    EXPECT_EQ(g.seqs_[cr].size(), g.durations_[cr].size() + 1U);
  }
}

TEST(routing, component_graph_lb_values) {
  auto const tt = load_tt();
  auto const g = build_component_graph(tt);

  auto const c = [&](char const* id) {
    return g.location_component_[find_loc(tt, id)];
  };

  // bounds towards destination D (forward search)
  auto const lb = compute_component_lb(g, direction::kForward,
                                       {{c("D"), std::uint16_t{0U}}},
                                       kMaxTransfers + 1U);

  EXPECT_EQ(0U, lb.tt_[to_idx(c("D"))]);
  EXPECT_EQ(0U, lb.ic_[to_idx(c("D"))]);

  // B/B2 -> D: fastest trip 30 min, one boarding
  EXPECT_EQ(30U, lb.tt_[to_idx(c("B"))]);
  EXPECT_EQ(1U, lb.ic_[to_idx(c("B"))]);

  // A -> D: min(R3 direct 120, R1 A->B 30 + R2 B2->D 30) = 60,
  // but min boardings = 1 (R3 direct exists)
  EXPECT_EQ(60U, lb.tt_[to_idx(c("A"))]);
  EXPECT_EQ(1U, lb.ic_[to_idx(c("A"))]);

  // E: cannot reach D at all
  EXPECT_EQ(component_lb::kUnreachableTt, lb.tt_[to_idx(c("E"))]);
  EXPECT_EQ(component_lb::kUnreachableIc, lb.ic_[to_idx(c("E"))]);
}

TEST(routing, component_graph_lb_admissible) {
  auto const tt = load_tt();
  auto const g = build_component_graph(tt);

  auto const check_dir = [&](direction const dir) {
    for (auto const dest : {"A", "B", "B2", "C1", "D", "E"}) {
      auto const lb = compute_component_lb(
          g, dir,
          {{g.location_component_[find_loc(tt, dest)], std::uint16_t{0U}}},
          kMaxTransfers + 1U);

      for (auto const from : {"A", "B", "B2", "C1", "D", "E"}) {
        if (std::string_view{from} == dest) {
          continue;
        }
        auto const results = test::raptor_search(
            tt, nullptr, dir == direction::kForward ? from : dest,
            dir == direction::kForward ? dest : from,
            "2024-06-08 06:00 Europe/Berlin", direction::kForward);

        auto const l = find_loc(tt, from);
        auto const comp = to_idx(g.location_component_[l]);
        for (auto const& j : results) {
          auto const tt_min =
              std::chrono::duration_cast<std::chrono::minutes>(j.travel_time())
                  .count();
          // travel time lb admissible for every journey
          EXPECT_LE(lb.tt_[comp], tt_min)
              << from << " -> " << dest << " dir=" << static_cast<int>(dir);
          // boarding count lb admissible for every journey
          EXPECT_LE(lb.ic_[comp], j.transfers_ + 1U)
              << from << " -> " << dest << " dir=" << static_cast<int>(dir);
        }
        if (results.size() != 0U) {
          EXPECT_NE(component_lb::kUnreachableTt, lb.tt_[comp])
              << from << " -> " << dest;
        }
      }
    }
  };

  check_dir(direction::kForward);
  check_dir(direction::kBackward);
}
