#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/dijkstra.h"
#include "nigiri/routing/query.h"
#include "nigiri/timetable.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::routing;

namespace {
mem_dir dijkstra_files() {
  return mem_dir::read(R"__(
"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
MTA,MOTIS Transit Authority,https://motis-project.de/,Europe/Berlin

# calendar_dates.txt
service_id,date,exception_type
S,20240608,1

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,A,,,,0,
B,B,B,,,,0,
C,C,C,,,,0,
D,D,D,,,,1,
D1,D1,D1,,,,0,D
D2,D2,D2,,,,0,D

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
AB,MTA,AB,AB,A -> B,0
BC,MTA,BC,BC,B -> C,0
DB,MTA,DB,DB,D -> B,0
DC,MTA,DC,DC,D -> C,0

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
AB,S,AB_TRP,AB_TRP,1
BC,S,BC_TRP,BC_TRP,2
DB,S,DB_TRP,DB_TRP,3
DC,S,DC_TRP,DC_TRP,4

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
AB_TRP,09:00,09:00,A,0,0,0
AB_TRP,09:30,09:30,B,1,0,0
DB_TRP,09:30,09:30,D1,0,0,0
DB_TRP,09:45,09:45,B,1,0,0
BC_TRP,10:00,10:00,B,0,0,0
BC_TRP,11:00,11:00,C,1,0,0
DC_TRP,10:30,10:30,D2,0,0,0
DC_TRP,11:30,11:30,C,1,0,0

)__");
}
}  // namespace

TEST(routing, dijkstra_exact) {
  timetable tt;
  tt.date_range_ = {sys_days{2024_y / June / 7}, sys_days{2024_y / June / 9}};
  register_special_stations(tt);
  auto const src = source_idx_t{0U};
  gtfs::load_timetable({}, src, dijkstra_files(), tt);
  finalize(tt);

  std::cout << "tt:\n" << tt << "\n";

  auto const d1_l = tt.locations_.location_id_to_idx_.at({"D1", src});

  auto const q_d1_c = query{
      .start_time_ = unixtime_t{sys_days{2024_y / June / 8} + 7_hours},
      .start_match_mode_ = location_match_mode::kExact,
      .dest_match_mode_ = location_match_mode::kExact,
      .start_ = {{d1_l, 0_minutes, 0U}},
      .destination_ = {{tt.locations_.location_id_to_idx_.at({"C", src}),
                        0_minutes, 0U}},
  };

  std::cout << "lb_graph:\n";
  for (auto l = 0U; l < tt.n_locations(); ++l) {
    std::cout << std::format("{}: ",
                             tt.locations_.get(location_idx_t{l}).name_);
    for (auto const& fp : tt.fwd_search_lb_graph_[location_idx_t{l}]) {
      std::cout << fp << " ";
    }
    std::cout << "\n";
  }

  auto dists = std::vector<std::uint16_t>{};
  dijkstra(tt, q_d1_c, tt.fwd_search_lb_graph_, dists);

  std::cout << "dists:\n";
  for (auto l = 0U; l < dists.size(); ++l) {
    std::cout << std::format(
        "{}: {}\n", tt.locations_.get(location_idx_t{l}).name_, dists[l]);
  }

  EXPECT_EQ(75U, dists[d1_l.v_]);
}

TEST(routing, dijkstra_equivalent) {
  timetable tt;
  tt.date_range_ = {sys_days{2024_y / June / 7}, sys_days{2024_y / June / 9}};
  register_special_stations(tt);
  auto const src = source_idx_t{0U};
  gtfs::load_timetable({}, src, dijkstra_files(), tt);
  finalize(tt);

  std::cout << "tt:\n" << tt << "\n";

  auto const d1_l = tt.locations_.location_id_to_idx_.at({"D1", src});

  auto const q_d1_c = query{
      .start_time_ = unixtime_t{sys_days{2024_y / June / 8} + 7_hours},
      .start_match_mode_ = location_match_mode::kEquivalent,
      .dest_match_mode_ = location_match_mode::kEquivalent,
      .start_ = {{d1_l, 0_minutes, 0U}},
      .destination_ = {{tt.locations_.location_id_to_idx_.at({"C", src}),
                        0_minutes, 0U}},
  };

  std::cout << "lb_graph:\n";
  for (auto l = 0U; l < tt.n_locations(); ++l) {
    std::cout << std::format("{}: ",
                             tt.locations_.get(location_idx_t{l}).name_);
    for (auto const& fp : tt.fwd_search_lb_graph_[location_idx_t{l}]) {
      std::cout << fp << " ";
    }
    std::cout << "\n";
  }

  auto dists = std::vector<std::uint16_t>{};
  dijkstra(tt, q_d1_c, tt.fwd_search_lb_graph_, dists);

  std::cout << "dists:\n";
  for (auto l = 0U; l < dists.size(); ++l) {
    std::cout << std::format(
        "{}: {}\n", tt.locations_.get(location_idx_t{l}).name_, dists[l]);
  }

  EXPECT_EQ(60U, dists[d1_l.v_]);
}
