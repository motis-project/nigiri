#include "gtest/gtest.h"

#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "nigiri/routing/meat/compact_representation.h"
#include "nigiri/routing/meat/csa/meat_csa.h"
#include "nigiri/routing/meat/expanded_representation.h"

#include "../load_sv_tt.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace std::string_view_literals;

namespace m = routing::meat;
namespace mcsa = m::csa;

namespace {

// Timetable:
// Routs/Connections
// R1: S -> A -> B -> T
// S -> A: 00:00 - 24:02
// A -> B: 24:04 - 24:06
// B -> T: 24:10 - 48:12
constexpr auto const test_tt_multi_day_trips = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S,S,,0.0,0.0,,
A,A,,1.0,0.0,,
B,B,,2.0,0.0,,
T,T,,3.0,0.0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,R1,,,2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S1,TR10,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
TR10,01:00:00,01:00:00,S,0
TR10,25:02:00,25:04:00,A,1
TR10,25:06:00,25:10:00,B,2
TR10,49:12:00,49:12:00,T,3

# calendar_dates.txt
service_id,date,exception_type
S1,20240301,1
S1,20240302,1
S1,20240303,1
S1,20240304,1
)"sv;

constexpr auto const expanded_dot_graph_a_b = R"(digraph decision_graph{
	splines=polyline;rankdir=LR;
	node0[shape=record,tooltip="(A, A)\ntransfer time=00:02.0",label="A|<slot0>2024-03-03 00:04"];
	node1[shape=record,tooltip="(B, B)\ntransfer time=00:02.0",label="B|<slot1>2024-03-03 00:06"];
	node0:slot0 -> node1:slot1 [label="R1",tooltip="probability of use=1\nMEAT=2024-03-03 00:06\n   1: A       A...............................................                               d: 03.03 00:04 [03.03 01:04]  [{name=R1, day=2024-03-02, id=TR10, src=0}]\n   2: B       B............................................... a: 03.03 00:06 [03.03 01:06]\n"];
}
)";

constexpr auto const expanded_dot_graph_s_t = R"(digraph decision_graph{
	splines=polyline;rankdir=LR;
	node0[shape=record,tooltip="(S, S)\ntransfer time=00:02.0",label="S|<slot0>2024-03-03 00:00"];
	node1[shape=record,tooltip="(T, T)\ntransfer time=00:02.0",label="T|<slot1>2024-03-05 00:12"];
	node0:slot0 -> node1:slot1 [label="R1",tooltip="probability of use=1\nMEAT=2024-03-05 00:12\n   0: S       S...............................................                               d: 03.03 00:00 [03.03 01:00]  [{name=R1, day=2024-03-03, id=TR10, src=0}]\n   1: A       A............................................... a: 04.03 00:02 [04.03 01:02]  d: 04.03 00:04 [04.03 01:04]  [{name=R1, day=2024-03-03, id=TR10, src=0}]\n   2: B       B............................................... a: 04.03 00:06 [04.03 01:06]  d: 04.03 00:10 [04.03 01:10]  [{name=R1, day=2024-03-03, id=TR10, src=0}]\n   3: T       T............................................... a: 05.03 00:12 [05.03 01:12]\n"];
}
)";

}  // namespace

TEST(MeatCsa, MultiDayTripsAB) {
  auto tt = test::load_tt(
      test_tt_multi_day_trips,
      {date::sys_days{2024_y / March / 1}, date::sys_days{2024_y / March / 5}});
  auto state = mcsa::meat_csa_state<mcsa::dynamic_profile_set>{};
  auto meat = mcsa::meat_csa<mcsa::dynamic_profile_set>{
      tt, state, day_idx_t{6}, routing::all_clasz_allowed()};
  auto g = m::decision_graph{};

  auto const from = "A";
  auto const to = "B";
  auto const start_time = *tt.date_range_.begin() + 2_days;
  auto const start_location =
      tt.locations_.location_id_to_idx_.at({from, source_idx_t{0}});
  auto const end_location =
      tt.locations_.location_id_to_idx_.at({to, source_idx_t{0}});
  auto const prf_idx = profile_idx_t{0};

  meat.execute(start_time, start_location, end_location, prf_idx, g);

  auto r = m::expanded_representation{g};
  auto ss = std::stringstream{};
  m::write_dot(ss, tt, g, r);
  EXPECT_EQ(ss.str(), expanded_dot_graph_a_b)
      << ss.str() << "\n " << expanded_dot_graph_a_b;
}

TEST(MeatCsa, MultiDayTripsST) {
  auto tt = test::load_tt(
      test_tt_multi_day_trips,
      {date::sys_days{2024_y / March / 1}, date::sys_days{2024_y / March / 5}});
  auto state = mcsa::meat_csa_state<mcsa::dynamic_profile_set>{};
  auto meat = mcsa::meat_csa<mcsa::dynamic_profile_set>{
      tt,  state, day_idx_t{6}, routing::all_clasz_allowed(), 30, 1.0,
      0.0, 0.0,   3_days};
  auto g = m::decision_graph{};

  auto const from = "S";
  auto const to = "T";
  auto const start_time = *tt.date_range_.begin() + 2_days;
  auto const start_location =
      tt.locations_.location_id_to_idx_.at({from, source_idx_t{0}});
  auto const end_location =
      tt.locations_.location_id_to_idx_.at({to, source_idx_t{0}});
  auto const prf_idx = profile_idx_t{0};

  meat.execute(start_time, start_location, end_location, prf_idx, g);

  auto r = m::expanded_representation{g};
  auto ss = std::stringstream{};
  m::write_dot(ss, tt, g, r);
  EXPECT_EQ(ss.str(), expanded_dot_graph_s_t)
      << ss.str() << "\n " << expanded_dot_graph_s_t;
}
