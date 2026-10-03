#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/query.h"
#include "nigiri/timetable.h"

#include "../../raptor_search.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace std::chrono_literals;

namespace {

mem_dir stop_group_files() {
  return mem_dir::read(R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
G1,Group 1,,0.0,0.0,,0,
G2,Group 2,,0.0,0.0,,0,
A,Stop A,,48.1,11.5,,0,
B,Stop B,,48.2,11.6,,0,

# stop_group_elements.txt
stop_group_id,stop_id
G1,A
G2,B

# calendar_dates.txt
service_id,date,exception_type
S1,20200101,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,R1,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S1,T1,R1,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,10:00:00,10:00:00,A,1,0,0
T1,10:30:00,10:30:00,B,2,0,0
)");
}

}  // namespace

TEST(gtfs, stop_groups_equivalent_routing) {
  timetable tt;
  register_special_stations(tt);
  tt.date_range_ = {sys_days{2020_y / January / 1},
                    sys_days{2020_y / January / 2}};

  load_timetable({}, source_idx_t{0}, stop_group_files(), tt);
  finalize(tt);

  auto const src = source_idx_t{0};
  auto q = routing::query{
      .start_time_ = interval<unixtime_t>{sys_days{2020_y / January / 1} + 0h,
                                          sys_days{2020_y / January / 1} + 24h},
      .start_match_mode_ = routing::location_match_mode::kEquivalent,
      .dest_match_mode_ = routing::location_match_mode::kEquivalent,
      .use_start_footpaths_ = true,
      .start_ = {{tt.locations_.location_id_to_idx_.at({"G1", src}), 0_minutes,
                  0U}},
      .destination_ = {
          {tt.locations_.location_id_to_idx_.at({"G2", src}), 0_minutes, 0U}}};

  auto const result =
      nigiri::test::search_pong(tt, nullptr, std::move(q), direction::kForward);

  ASSERT_FALSE(result.empty());
}

// A GTFS stop group is a dummy stop at (0,0) that is equivalent to its
// members. Its "walk" to a member in Germany is thousands of kilometers - far
// more minutes than a duration holds - and must not wrap into a footpath.
TEST(gtfs, stop_group_at_null_island_gets_no_footpaths) {
  constexpr auto const kGroupFeed = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
AG,Agency,https://example.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
GRP,Group,,0.0,0.0,,,
M1,M1,,50.0,6.5,,,
M2,M2,,50.0,7.5,,,

# stop_group_elements.txt
stop_group_id,stop_id
GRP,M1
GRP,M2

# calendar_dates.txt
service_id,date,exception_type
X,20190501,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R,AG,R,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R,X,T,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T,10:00:00,10:00:00,M1,1,0,0
T,10:30:00,10:30:00,M2,2,0,0
)";
  for (auto const adjust_footpaths : {true, false}) {
    auto tt = timetable{};
    tt.date_range_ = {sys_days{2019_y / May / 1}, sys_days{2019_y / May / 2}};
    loader::register_special_stations(tt);
    loader::gtfs::load_timetable({}, source_idx_t{0},
                                 loader::mem_dir::read(kGroupFeed), tt);
    loader::finalize(tt, {.adjust_footpaths_ = adjust_footpaths});

    auto const grp =
        tt.locations_.location_id_to_idx_.at({"GRP", source_idx_t{0}});
    ASSERT_FALSE(tt.locations_.equivalences_[grp].empty());
    EXPECT_TRUE(tt.locations_.footpaths_out_[kDefaultProfile][grp].empty());
    EXPECT_TRUE(tt.locations_.footpaths_in_[kDefaultProfile][grp].empty());
  }
}
