#include "gtest/gtest.h"

#include <ranges>

#include "utl/enumerate.h"

#include "nigiri/routing/lb_raptor.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

#include "../util.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::routing;

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

constexpr auto kExpLbP = std::array<std::uint16_t, 16U>{
    65535U, 65535U, 578U, 487U, 247U, 142U, 142U, 142U,
    142U,   142U,   142U, 142U, 142U, 142U, 142U, 142U};
constexpr auto kExpLbF = std::array<std::uint16_t, 16U>{/* TODO F should be reached in k = 1 already, since it only takes one transport and a footpath. It follows that lb_raptor needs the search direction to cor*/};
constexpr auto kExpLbS = std::array<std::uint16_t, 16U>{
    65535U, 518U, 427U, 187U, 82U, 82U, 82U, 82U,
    82U,    82U,  82U,  82U,  82U, 82U, 82U, 82U};
constexpr auto kExpLbB1 = std::array<std::uint16_t, 16U>{
    65535U, 187U, 187U, 187U, 187U, 187U, 187U, 187U,
    187U,   187U, 187U, 187U, 187U, 187U, 187U, 187U};
constexpr auto kExpLbC1 = std::array<std::uint16_t, 16U>{
    65535U, 65535U, 127U, 127U, 127U, 127U, 127U, 127U,
    127U,   127U,   127U, 127U, 127U, 127U, 127U, 127U};
constexpr auto kExpLbC2 =
    std::array<std::uint16_t, 16U>{65535U, 67U, 67U, 67U, 67U, 67U, 67U, 67U,
                                   67U,    67U, 67U, 67U, 67U, 67U, 67U, 67U};
constexpr auto kExpLbD1 = std::array<std::uint16_t, 16U>{
    65535U, 65535U, 65535U, 67U, 67U, 67U, 67U, 67U,
    67U,    67U,    67U,    67U, 67U, 67U, 67U, 67U};
constexpr auto kExpLbD2 = std::array<std::uint16_t, 16U>{
    65535U, 65535U, 52U, 52U, 52U, 52U, 52U, 52U,
    52U,    52U,    52U, 52U, 52U, 52U, 52U, 52U};
constexpr auto kExpLbD3 =
    std::array<std::uint16_t, 16U>{65535U, 37U, 37U, 37U, 37U, 37U, 37U, 37U,
                                   37U,    37U, 37U, 37U, 37U, 37U, 37U, 37U};
constexpr auto kExpLbT = std::array<std::uint16_t, 16U>{
    7U, 7U, 7U, 7U, 7U, 7U, 7U, 7U, 7U, 7U, 7U, 7U, 7U, 7U, 7U, 7U};

TEST(routing, lb_raptor) {
  auto const tt = load_gtfs(lb_test_tt, kGtfsDateRange);
  auto rtt = rt::create_rt_timetable(tt, sys_days{2026_y / February / 27});
  auto const T = tt.find(location_id{"T", source_idx_t{}}).value();
  auto const q =
      query{.start_time_ = unixtime_t{sys_days{February / 27 / 2026}},
            .destination_ = {{tt.locations_.location_id_to_idx_.at(
                                  {"T", source_idx_t{0U}}),
                              13_minutes, 0U}},
            .td_dest_{{T,
                       {{.valid_from_ = sys_days{2026_y / January / 27},
                         .duration_ = 7_minutes,
                         .transport_mode_id_ = 5},
                        {.valid_from_ = sys_days{2026_y / January / 28},
                         .duration_ = footpath::kMaxDuration,
                         .transport_mode_id_ = 5}}}}};
  auto state = raptor_state{};
  auto location_round_lb =
      vector_map<location_idx_t,
                 std::array<std::uint16_t, kMaxTransfers + 2U>>{};

  lb_raptor(tt, q, tt.fwd_search_lb_graph_[q.prf_idx_], nullptr, nullptr, state,
            location_round_lb);

  for (auto const [i, round_lb] : utl::enumerate(location_round_lb)) {
    fmt::println("{}: {}", tt.get_default_name(location_idx_t{i}), round_lb);
  }

  ASSERT_LE(kMaxTransfers, std::uint8_t{14U});

  auto const check = [](auto&& exp, auto&& act) {
    for (auto const [e, a] :
         std::views::zip(exp | std::views::take(kMaxTransfers + 2U),
                         act | std::views::take(kMaxTransfers + 2U))) {
      EXPECT_EQ(e, a);
    }
  };

  check(kExpLbT, location_round_lb[T]);
}