#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/raptor/pong.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/routing/search.h"
#include "nigiri/timetable.h"

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

  auto search_state = routing::search_state{};
  auto raptor_state = routing::raptor_state{};

  auto const src = source_idx_t{0};
  auto q = routing::query{
      .start_time_ = sys_days{2020_y / January / 1} + 9h,
      .start_match_mode_ = routing::location_match_mode::kEquivalent,
      .dest_match_mode_ = routing::location_match_mode::kEquivalent,
      .use_start_footpaths_ = true,
      .start_ = {{tt.locations_.location_id_to_idx_.at({"G1", src}), 0_minutes,
                  0U}},
      .destination_ = {
          {tt.locations_.location_id_to_idx_.at({"G2", src}), 0_minutes, 0U}}};

  auto const result =
      routing::pong_search(tt, nullptr, search_state, raptor_state,
                           std::move(q), direction::kForward);

  ASSERT_FALSE(result.journeys_->empty());
}
