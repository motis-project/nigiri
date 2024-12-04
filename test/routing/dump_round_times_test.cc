#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/timetable.h"

#include "../raptor_search.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::routing;
using nigiri::test::raptor_intermodal_search;

namespace {

mem_dir dump_round_times_files() {
  return mem_dir::read(R"__(
"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
MTA,MOTIS Transit Authority,https://motis-project.de/,Europe/Berlin

# calendar_dates.txt
service_id,date,exception_type
D,20240608,1

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A0,A0,A0,,,,,,
A1,A1,A1,,,,,,
B0,B0,B0,,,,,,
B1,B1,B1,,,,,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
A,MTA,A,A,A0 -> A1,0
B,MTA,B,B,B0 -> B1,0


# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
A,D,AWE,AWE,1
B,D,BWE,BWE,2

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
AWE,02:00,02:00,A0,0,0,0
AWE,03:00,03:00,A1,1,0,0
BWE,04:00,04:00,B0,0,0,0
BWE,05:00,05:00,B1,1,0,0

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
A1,B0,2,300
)__");
}

constexpr interval<std::chrono::sys_days> dump_round_times_period() {
  using namespace date;
  constexpr auto const from = (2024_y / June / 7).operator sys_days();
  constexpr auto const to = (2024_y / June / 9).operator sys_days();
  return {from, to};
}

TEST(routing, dump_round_times) {
  constexpr auto const src = source_idx_t{0U};
  auto const config = loader_config{};

  timetable tt;
  tt.date_range_ = dump_round_times_period();
  register_special_stations(tt);
  gtfs::load_timetable(config, src, dump_round_times_files(), tt);
  finalize(tt);

  auto const results = raptor_intermodal_search(
      tt, nullptr,
      {{tt.locations_.location_id_to_idx_.at({.id_ = "A0", .src_ = src}),
        3_minutes, 23U}},
      {{tt.locations_.location_id_to_idx_.at({.id_ = "B1", .src_ = src}),
        10_minutes, 42U}},
      interval{unixtime_t{sys_days{2024_y / June / 7}},
               unixtime_t{sys_days{2024_y / June / 8}}},
      direction::kForward);

  ASSERT_EQ(1U, results.size());

  std::stringstream ss;
  ss << "\n";
  for (auto const& x : results) {
    x.print(ss, tt);
    ss << "\n\n";
  }
}

}  // namespace