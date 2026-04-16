#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"
#include "../../include/nigiri/routing/lb/lb_transit_legs.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::routing;

// routes:
// X -> Y -> P | P -> S | S -> T
//                      | S -> B1 | B1 -> T
//                      | S -> C1 | C1 -> C2 | C2 -> T
//                      | S -> D1 | D1 -> D2 | D2 -> D3 | D3 -> T
//
// footpaths:
// F -> S
mem_dir lb_test_tt() {
  return mem_dir::read(R"__(
"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
MTA,MOTIS Transit Authority,https://motis-project.de/,Europe/Berlin

# calendar_dates.txt
service_id,date,exception_type
SID,20260227,1

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
O,O,,,,,,
X,X,,,,,,
Y,Y,,,,,,
P,P,,,,,,
F,F,,,,,,
S,S,,,,,,
B1,B1,,,,,,
C1,C1,,,,,,
C2,C2,,,,,,
D1,D1,,,,,,
D2,D2,,,,,,
D3,D3,,,,,,
T,T,,,,,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
XYP,MTA,XYP,XYP,X -> Y -> P,0
PS,MTA,PS,PS,P -> S,0
A,MTA,A,A,S -> T,0
B1,MTA,B1,B1,S -> B1,0
B2,MTA,B2,B2,B1 -> T,0
C1,MTA,C1,C1,S -> C1,0
C2,MTA,C2,C2,C1 -> C2,0
C3,MTA,C3,C3,C2 -> T,0
D1,MTA,D1,D1,S -> D1,0
D2,MTA,D2,D2,D1 -> D2,0
D3,MTA,D3,D3,D2 -> D3,0
D4,MTA,D4,D4,D3 -> T,0

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
XYP,SID,XYP,XYP,0
PS,SID,PS,PS,1
A,SID,A,A,2
B1,SID,B1,B1,3
B2,SID,B2,B2,4
C1,SID,C1,C1,5
C2,SID,C2,C2,6
C3,SID,C3,C3,7
D1,SID,D1,D1,8
D2,SID,D2,D2,9
D3,SID,D3,D3,10
D4,SID,D4,D4,11

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
XYP,12:00,12:15,X,0,0,0
XYP,13:00,14:15,Y,1,0,0
XYP,15:00,15:15,P,2,0,0
PS,03:00,03:00,P,0,0,0
PS,04:00,04:00,S,1,0,0
A,08:00,08:00,S,0,0,0
A,18:00,18:00,T,1,0,0
B1,09:00,09:00,S,0,0,0
B1,13:00,13:00,B1,1,0,0
B2,14:00,14:00,B1,0,0,0
B2,17:00,17:00,T,1,0,0
C1,10:00,10:00,S,0,0,0
C1,11:00,11:00,C1,1,0,0
C2,12:00,12:00,C1,0,0,0
C2,13:00,13:00,C2,1,0,0
C3,14:00,14:00,C2,0,0,0
C3,15:00,15:00,T,1,0,0
D1,11:00,11:00,S,0,0,0
D1,11:15,11:15,D1,1,0,0
D2,11:30,11:30,D1,0,0,0
D2,11:45,11:45,D2,1,0,0
D3,12:00,12:00,D2,0,0,0
D3,12:15,12:15,D3,1,0,0
D4,12:30,12:30,D3,0,0,0
D4,13:00,13:00,T,0,0,0

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
F,S,2,600
)__");
}

constexpr auto kGtfsDateRange = interval{sys_days{2026_y / February / 27},
                                         sys_days{2026_y / February / 28}};

timetable load_gtfs(auto const& files, interval<sys_days> const date_range) {
  using namespace date;
  timetable tt;
  tt.date_range_ = date_range;
  register_special_stations(tt);
  gtfs::load_timetable({}, source_idx_t{0U}, files(), tt);
  finalize(tt);
  return tt;
}

TEST(routing, lb_transit_legs) {
  auto const tt = load_gtfs(lb_test_tt, kGtfsDateRange);
  auto rtt = rt::create_rt_timetable(tt, sys_days{2026_y / February / 27});
  auto q = query{
      .start_time_ = unixtime_t{sys_days{February / 27 / 2026}},
      .start_ = {{tt.locations_.location_id_to_idx_.at({"P", source_idx_t{0U}}),
                  3_minutes, 0U}},
      .destination_ = {{tt.locations_.location_id_to_idx_.at(
                            {"T", source_idx_t{0U}}),
                        13_minutes, 0U}},
      .td_dest_{{tt.find(location_id{"T", source_idx_t{}}).value(),
                 {{.valid_from_ = sys_days{2026_y / January / 27},
                   .duration_ = 7_minutes,
                   .transport_mode_id_ = 5},
                  {.valid_from_ = sys_days{2026_y / January / 28},
                   .duration_ = footpath::kMaxDuration,
                   .transport_mode_id_ = 5}}}}};

  auto state = lb_transit_legs_state{};
  lb_transit_legs<direction::kForward>(tt, q, state);

  auto const get_lb = [&](auto&& id) {
    return state
        .lb_[tt.locations_.location_id_to_idx_.at({id, source_idx_t{0U}})];
  };

  ASSERT_EQ(state.lb_.size(), tt.n_locations());
  EXPECT_EQ(get_lb("T"), 0U);
  EXPECT_EQ(get_lb("D3"), 1U);
  EXPECT_EQ(get_lb("C2"), 1U);
  EXPECT_EQ(get_lb("B1"), 1U);
  EXPECT_EQ(get_lb("S"), 1U);
  EXPECT_EQ(get_lb("F"), 1U);
  EXPECT_EQ(get_lb("P"), 2U);
  EXPECT_EQ(get_lb("D2"), 2U);
  EXPECT_EQ(get_lb("C1"), 2U);
  EXPECT_EQ(get_lb("D1"), 3U);
  EXPECT_EQ(get_lb("X"), 3U);
  EXPECT_EQ(get_lb("Y"), 3U);

  q.flip_dir();
  lb_transit_legs<direction::kBackward>(tt, q, state);

  ASSERT_EQ(state.lb_.size(), tt.n_locations());
  EXPECT_EQ(get_lb("T"), 2U);
  EXPECT_EQ(get_lb("D3"), 4U);
  EXPECT_EQ(get_lb("C2"), 3U);
  EXPECT_EQ(get_lb("B1"), 2U);
  EXPECT_EQ(get_lb("S"), 1U);
  EXPECT_EQ(get_lb("F"), 1U);
  EXPECT_EQ(get_lb("P"), 0U);
  EXPECT_EQ(get_lb("D2"), 3U);
  EXPECT_EQ(get_lb("C1"), 2U);
  EXPECT_EQ(get_lb("D1"), 2U);
  EXPECT_EQ(get_lb("X"), std::numeric_limits<uint8_t>::max());
  EXPECT_EQ(get_lb("Y"), std::numeric_limits<uint8_t>::max());
}