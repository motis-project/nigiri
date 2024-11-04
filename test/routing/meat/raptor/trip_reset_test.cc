#include "gtest/gtest.h"

#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "nigiri/routing/meat/compact_representation.h"
#include "nigiri/routing/meat/expanded_representation.h"
#include "nigiri/routing/meat/raptor/meat_raptor.h"

#include "../load_sv_tt.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace std::string_view_literals;

namespace m = routing::meat;
namespace mraptor = m::raptor;

namespace {

// Timetable:
// Routs/Connections
// R1: A -> S -> C
// A -> S: 00:00 - 00:10
// S -> C: 00:12 - 00:20
//
// R2: B -> C -> T
// B -> C: 00:20 - 24:01
// C -> T: 24:20 - 24:30
//
// B -> C: 01:00 - 25:01
// C -> T: 25:20 - 25:30
constexpr auto const test_tt_trip_reset = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S,S,,0.0,0.0,,
A,A,,1.0,0.0,,
B,B,,2.0,0.0,,
C,C,,3.0,0.0,,
T,T,,4.0,0.0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,R1,,,2
R2,DB,R2,,,2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S1,TR10,,
R2,S1,TR20,,
R2,S1,TR21,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
TR10,01:00:00,01:00:00,A,0
TR10,01:10:00,01:12:00,S,1
TR10,01:20:00,01:20:00,C,2
TR20,01:20:00,01:20:00,B,0
TR20,25:01:00,25:20:00,C,1
TR20,25:30:00,25:30:00,T,2
TR21,02:20:00,02:20:00,B,0
TR21,26:01:00,26:20:00,C,1
TR21,26:30:00,26:30:00,T,2

# calendar_dates.txt
service_id,date,exception_type
S1,20240301,1
S1,20240302,1
S1,20240303,1
S1,20240304,1
)"sv;

constexpr auto const expanded_dot_graph_s_t = R"(digraph decision_graph{
	splines=polyline;rankdir=LR;
	node0[shape=record,tooltip="(S, S)\ntransfer time=00:02.0",label="S|<slot0>2024-03-02 00:12"];
	node1[shape=record,tooltip="(C, C)\ntransfer time=00:02.0",label="C|<slot1>2024-03-02 00:20|<slot2>2024-03-02 01:20"];
	node2[shape=record,tooltip="(T, T)\ntransfer time=00:02.0",label="T|<slot3>2024-03-02 00:30|<slot4>2024-03-02 01:30"];
	node0:slot0 -> node1:slot1 [label="R1",tooltip="probability of use=1\nMEAT=2024-03-02 01:30\n   1: S       S...............................................                               d: 02.03 00:12 [02.03 01:12]  [{name=R1, day=2024-03-02, id=TR10, src=0}]\n   2: C       C............................................... a: 02.03 00:20 [02.03 01:20]\n"];
	node1:slot1 -> node2:slot3 [label="R2",tooltip="probability of use=0\nMEAT=2024-03-02 00:30\n   1: C       C...............................................                               d: 02.03 00:20 [02.03 01:20]  [{name=R2, day=2024-03-01, id=TR20, src=0}]\n   2: T       T............................................... a: 02.03 00:30 [02.03 01:30]\n"];
	node1:slot2 -> node2:slot4 [label="R2",tooltip="probability of use=1\nMEAT=2024-03-02 01:30\n   1: C       C...............................................                               d: 02.03 01:20 [02.03 02:20]  [{name=R2, day=2024-03-01, id=TR21, src=0}]\n   2: T       T............................................... a: 02.03 01:30 [02.03 02:30]\n"];
}
)";

constexpr auto const expanded_dot_graph_s_t2 = R"(digraph decision_graph{
	splines=polyline;rankdir=LR;
	node0[shape=record,tooltip="(S, S)\ntransfer time=00:02.0",label="S|<slot0>2024-03-03 00:12"];
	node1[shape=record,tooltip="(C, C)\ntransfer time=00:02.0",label="C|<slot1>2024-03-03 00:20|<slot2>2024-03-03 01:20"];
	node2[shape=record,tooltip="(T, T)\ntransfer time=00:02.0",label="T|<slot3>2024-03-03 00:30|<slot4>2024-03-03 01:30"];
	node0:slot0 -> node1:slot1 [label="R1",tooltip="probability of use=1\nMEAT=2024-03-03 01:30\n   1: S       S...............................................                               d: 03.03 00:12 [03.03 01:12]  [{name=R1, day=2024-03-03, id=TR10, src=0}]\n   2: C       C............................................... a: 03.03 00:20 [03.03 01:20]\n"];
	node1:slot1 -> node2:slot3 [label="R2",tooltip="probability of use=0\nMEAT=2024-03-03 00:30\n   1: C       C...............................................                               d: 03.03 00:20 [03.03 01:20]  [{name=R2, day=2024-03-02, id=TR20, src=0}]\n   2: T       T............................................... a: 03.03 00:30 [03.03 01:30]\n"];
	node1:slot2 -> node2:slot4 [label="R2",tooltip="probability of use=1\nMEAT=2024-03-03 01:30\n   1: C       C...............................................                               d: 03.03 01:20 [03.03 02:20]  [{name=R2, day=2024-03-02, id=TR21, src=0}]\n   2: T       T............................................... a: 03.03 01:30 [03.03 02:30]\n"];
}
)";

}  // namespace

TEST(MeatRaptor, TripsResetsSimple) {
  auto tt = test::load_tt(
      test_tt_trip_reset,
      {date::sys_days{2024_y / March / 1}, date::sys_days{2024_y / March / 5}});
  auto state = mraptor::meat_raptor_state{};
  auto meat = mraptor::meat_raptor{tt, state, day_idx_t{6},
                                   routing::all_clasz_allowed()};
  auto g = m::decision_graph{};

  auto const from = "S";
  auto const to = "T";
  auto const start_time = *tt.date_range_.begin() + 1_days;
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

TEST(MeatRaptor, TripsResets) {
  auto tt = test::load_tt(
      test_tt_trip_reset,
      {date::sys_days{2024_y / March / 1}, date::sys_days{2024_y / March / 5}});
  auto state = mraptor::meat_raptor_state{};
  auto meat = mraptor::meat_raptor{tt, state, day_idx_t{6},
                                   routing::all_clasz_allowed()};
  auto g = m::decision_graph{};

  auto const from = "S";
  auto const to = "T";
  auto start_time = *tt.date_range_.begin() + 1_days + 1_minutes;
  auto const start_location =
      tt.locations_.location_id_to_idx_.at({from, source_idx_t{0}});
  auto const end_location =
      tt.locations_.location_id_to_idx_.at({to, source_idx_t{0}});
  auto const prf_idx = profile_idx_t{0};

  meat.execute(start_time, start_location, end_location, prf_idx, g);

  meat.next_start_time();
  start_time = *tt.date_range_.begin() + 2_days + 1_minutes;

  meat.execute(start_time, start_location, end_location, prf_idx, g);

  auto r = m::expanded_representation{g};
  auto ss = std::stringstream{};
  m::write_dot(ss, tt, g, r);
  EXPECT_EQ(ss.str(), expanded_dot_graph_s_t2)
      << ss.str() << "\n " << expanded_dot_graph_s_t2;
}
