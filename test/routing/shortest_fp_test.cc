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

mem_dir shortest_fp_files() {
  return mem_dir::read(R"__(
"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
MTA,MOTIS Transit Authority,https://motis-project.de/,Europe/Berlin

# calendar.txt
service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
D,1,1,1,1,1,1,1,20240101,20241231

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A0,A0,start_offset0,,,,,,
A1,A1,start_offset1,,,,,,
A2,A2,start_offset2,,,,,,
A3,A3,first_transfer_to_B,,,,,,
A4,A4,second_transfer_to_B,,,,,,
A5,A5,final_stop_of_A,,,,,,
B0,B0,first_transfer_from_A,,,,,,
B1,B1,second_transfer_from_A,,,,,,
B2,B2,first_transfer_to_C,,,,,,
B3,B3,second_transfer_to_C,,,,,,
C0,C0,start_no_transfer,,,,,,
C1,C1,first_transfer_from_B,,,,,,
C2,C2,second_transfer_from_B,,,,,,
C3,C3,dest_offset0,,,,,,
C4,C4,dest_offset1,,,,,,
C5,C5,dest_offset2,,,,,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
A,MTA,A,A,A0 -> A5,0
B,MTA,B,B,B0 -> B3,0
C,MTA,C,C,C0 -> C5,0

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
A,D,AWE,AWE,1
B,D,BWE,BWE,2
C,D,CWE,CWE,3

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
AWE,02:00,02:00,A0,0,0,0
AWE,02:01,02:02,A1,1,0,0
AWE,02:06,02:07,A2,2,0,0
AWE,02:15,02:16,A3,3,0,0
AWE,02:20,02:21,A4,4,0,0
AWE,02:25,02:26,A5,5,0,0
BWE,02:58,03:00,B0,0,0,0
BWE,03:20,03:22,B1,1,0,0
BWE,03:30,03:32,B2,2,0,0
BWE,03:40,03:42,B3,3,0,0
CWE,03:58,04:00,C0,0,0,0
CWE,04:08,04:10,C1,1,0,0
CWE,04:18,04:20,C2,2,0,0
CWE,04:30,04:32,C3,3,0,0
CWE,04:40,04:42,C4,4,0,0
CWE,04:50,04:52,C5,5,0,0

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
A3,B0,2,300
A4,B1,2,600
B2,C1,2,600
B3,C2,2,300
)__");
}

constexpr interval<std::chrono::sys_days> shortest_fp_period() {
  using namespace date;
  constexpr auto const from = (2024_y / January / 01).operator sys_days();
  constexpr auto const to = (2024_y / December / 31).operator sys_days();
  return {from, to};
}

TEST(routing, raptor_shortest_fp_forward) {
  constexpr auto const src = source_idx_t{0U};
  auto const config = loader_config{};

  timetable tt;
  tt.date_range_ = shortest_fp_period();
  register_special_stations(tt);
  gtfs::load_timetable(config, src, shortest_fp_files(), tt);
  finalize(tt);

  auto const results = raptor_intermodal_search(
      tt, nullptr,
      {{tt.locations_.location_id_to_idx_.at({.id_ = "A0", .src_ = src}),
        3_minutes, 23U},
       {tt.locations_.location_id_to_idx_.at({.id_ = "A1", .src_ = src}),
        5_minutes, 23U},
       {tt.locations_.location_id_to_idx_.at({.id_ = "A2", .src_ = src}),
        10_minutes, 23U}},
      {{tt.locations_.location_id_to_idx_.at({.id_ = "C3", .src_ = src}),
        30_minutes, 42U},
       {tt.locations_.location_id_to_idx_.at({.id_ = "C4", .src_ = src}),
        20_minutes, 42U},
       {tt.locations_.location_id_to_idx_.at({.id_ = "C5", .src_ = src}),
        10_minutes, 42U}},
      interval{unixtime_t{sys_days{2024_y / June / 8}},
               unixtime_t{sys_days{2024_y / June / 8} + 12_hours}},
      direction::kForward);

  EXPECT_EQ(1U, results.size());

  std::stringstream ss;
  ss << "\n";
  for (auto const& x : results) {
    x.print(ss, tt);
    ss << "\n\n";
  }
  std::cout << ss.str() << "\n";
}

}  // namespace