#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"

#include "nigiri/rt/create_rt_timetable.h"
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
A,50.7677663,6.0913818,06:15-22:30
B,48.841004,10.0965113,"Mo-Fr 07:50-12:00, 12:45-18:10, Sa: 08:50-13:40, 14:25-19:10, So 08:50-13:45, 14:45-21:10"
C,51.7658783,8.9431876,08:15-17:45
D,49.298931,10.5775584,"Mo-Sa 06:50-18:50"
)";

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
R1,DB,RE 1,,,2
R2,DB,RE 2,,,2
R3,DB,RE 1,,,2
R4,DB,RE 2,,,2
R5,DB,RE 1,,,2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S,T1,RE 1,
R2,S,T2,RE 2,
R3,S,T3,RE 3,
R4,S,T4,RE 4,
R5,S,T5,RE 5,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,10:00:00,10:00:00,A,1,0,0
T1,11:00:00,11:00:00,B1,2,0,0
T2,11:30:00,11:30:00,B2,1,0,0
T2,12:00:00,12:00:00,C,2,0,0
T3,12:00:00,12:00:00,B2,1,0,0
T3,12:30:00,12:30:00,C,2,0,0
T4,10:00:00,10:00:00,A,1,0,0
T4,12:00:00,12:00:00,D,2,0,0
T5,13:00:00,13:00:00,D,1,0,0
T5,15:00:00,15:00:00,C,2,0,0
)");
}

// std::string to_string(timetable const& tt,
//                       pareto_set<routing::journey> const& results) {
//   std::stringstream ss;
//   ss << "\n";
//   for (auto const& x : results) {
//     x.print(ss, tt);
//     ss << "\n";
//   }
//   return ss.str();
// }

}  // namespace

TEST(routing, wheelchair_assistance) {
  constexpr auto const kProfile = profile_idx_t{2U};

  auto const assistance = read_assistance(kAssistance);

  timetable tt;
  tt.date_range_ = {date::sys_days{2024_y / June / 18},
                    date::sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, test_files(), tt);
  finalize(tt);

  //  auto const A = tt.locations_.get({"A", {}}).l_;
  //  auto const C = tt.locations_.get({"C", {}}).l_;
  //  auto const B1 = tt.locations_.get({"B1", {}}).l_;
  //  auto const B2 = tt.locations_.get({"B2", {}}).l_;

  auto const results = raptor_search(
      tt, nullptr, "A", "C",
      interval{unixtime_t{sys_days{2020_y / March / 30}} + 5_hours,
               unixtime_t{sys_days{2020_y / March / 30}} + 6_hours},
      direction::kBackward, kProfile);

  CISTA_UNUSED_PARAM(results)
  CISTA_UNUSED_PARAM(kAssistance)
}