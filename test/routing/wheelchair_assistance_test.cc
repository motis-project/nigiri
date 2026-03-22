#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"

#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
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

}  // namespace

TEST(routing, wheelchair_assistance) {
  auto assistance = read_assistance(kAssistance);

  timetable tt;
  tt.date_range_ = {date::sys_days{2024_y / June / 18},
                    date::sys_days{2024_y / June / 20}};
  register_special_stations(tt);
  load_timetable({}, source_idx_t{0}, test_files(), tt, &assistance);
  finalize(tt);

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
                    routing::all_clasz_allowed(), false, false, 0U);
  ASSERT_FALSE(results_walk.begin() == results_walk.end());
  EXPECT_EQ((unixtime_t{sys_days{2024_y / June / 19} + 5_hours}),
            results_walk.begin()->start_time_);
  EXPECT_EQ((unixtime_t{sys_days{2024_y / June / 19} + 20_hours}),
            std::prev(results_walk.end())->start_time_);

  auto const results_wheelchair =
      raptor_search(tt, nullptr, "A", "C", iv, direction::kForward,
                    routing::all_clasz_allowed(), false, false, 2U);
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
