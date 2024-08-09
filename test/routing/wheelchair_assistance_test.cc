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
A,0.0,1.0,08:00-22:00
B,2.0,3.0,08:00-22:00
C,4.0,5.0,08:00-22:00
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
route_id,service_id,trip_id,trip_headsign,block_id
R1,S,T1,RE 1,
R2,S,T2,RE 2,

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

std::string to_string(timetable const& tt,
                      pareto_set<routing::journey> const& results) {
  std::stringstream ss;
  ss << "\n";
  for (auto const& x : results) {
    x.print(ss, tt);
    ss << "\n";
  }
  return ss.str();
}

}  // namespace

TEST(routing, wheelchair_assistance) {
  constexpr auto const kProfile = profile_idx_t{0U};

  auto assistance = read_assistance(kAssistance);

  timetable tt;
  tt.date_range_ = {date::sys_days{2024_y / June / 18},
                    date::sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, test_files(), tt, &assistance);
  finalize(tt);

  std::cout << tt << "\n";

  auto const B1 = tt.locations_.get({"B1", {}}).l_;
  auto const B2 = tt.locations_.get({"B2", {}}).l_;
  tt.locations_.footpaths_out_[kProfile].resize(tt.n_locations());
  tt.locations_.footpaths_in_[kProfile].resize(tt.n_locations());
  tt.locations_.footpaths_out_[kProfile][B1].push_back(footpath{B2, 5min});
  tt.locations_.footpaths_in_[kProfile][B2].push_back(footpath{B1, 5min});

  auto const results = raptor_search(
      tt, nullptr, "A", "C",
      interval{unixtime_t{sys_days{2024_y / June / 19} + 5_hours},
               unixtime_t{sys_days{2024_y / June / 19} + 21_hours}},
      direction::kForward, routing::all_clasz_allowed(), false, kProfile);

  std::cout << to_string(tt, results) << "\n";
}