#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"

#include "nigiri/timetable.h"

#include "nigiri/routing/meat/compact_representation.h"
#include "nigiri/routing/meat/csa/meat_csa.h"
#include "nigiri/routing/meat/expanded_representation.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace std::string_view_literals;

namespace m = routing::meat;
namespace mcsa = m::csa;

// Timetable:
// Connections
// S -> A: 00:00 - 00:01
// A -> T: 00:01 - 00:02
//         00:29 - 00:31
//         00:30 - 00:32
//         00:31 - 00:33
//         00:32 - 00:34
//         00:33 - 00:34
//         00:36 - 00:37
constexpr auto const test_tt = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S,S,,0.0,0.0,,
A,A,,1.0,0.0,,
T,T,,2.0,0.0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
SA,DB,SA,,,2
AT,DB,AT,,,2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
SA,S1,SA0,,
AT,S1,AT0,,
AT,S1,AT1,,
AT,S1,AT2,,
AT,S1,AT3,,
AT,S1,AT4,,
AT,S1,AT5,,
AT,S1,AT6,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
SA0,01:00:00,01:00:00,S,0
SA0,01:01:00,01:01:00,A,1
AT0,01:01:00,01:01:00,A,0
AT0,01:02:00,01:02:00,T,1
AT1,01:29:00,01:29:00,A,0
AT1,01:31:00,01:31:00,T,1
AT2,01:30:00,01:30:00,A,0
AT2,01:32:00,01:32:00,T,1
AT3,01:31:00,01:31:00,A,0
AT3,01:33:00,01:33:00,T,1
AT4,01:32:00,01:32:00,A,0
AT4,01:34:00,01:34:00,T,1
AT5,01:33:00,01:33:00,A,0
AT5,01:35:00,01:35:00,T,1
AT6,01:36:00,01:36:00,A,0
AT6,01:37:00,01:37:00,T,1

# calendar_dates.txt
service_id,date,exception_type
S1,20240301,1
)"sv;

constexpr auto const expanded_dot_graph = R"(digraph decision_graph{
	splines=polyline;rankdir=LR;
	node0[shape=record,tooltip="(S, S)\ntransfer time=00:02.0",label="S|<slot0>2024-03-01 00:00"];
	node1[shape=record,tooltip="(A, A)\ntransfer time=00:02.0",label="A|<slot1>2024-03-01 00:01|<slot2>2024-03-01 00:29|<slot3>2024-03-01 00:30|<slot4>2024-03-01 00:31|<slot5>2024-03-01 00:32|<slot6>2024-03-01 00:33|<slot7>2024-03-01 00:36"];
	node2[shape=record,tooltip="(T, T)\ntransfer time=00:02.0",label="T|<slot8>2024-03-01 00:02|<slot9>2024-03-01 00:31|<slot10>2024-03-01 00:32|<slot11>2024-03-01 00:33|<slot12>2024-03-01 00:34|<slot13>2024-03-01 00:35|<slot14>2024-03-01 00:37"];
	node0:slot0 -> node1:slot1 [label="SA ",tooltip="probability of use=1\nMEAT=2024-03-01 00:31\n   0: S       S...............................................                               d: 01.03 00:00 [01.03 01:00]  [{name=SA , day=2024-03-01, id=SA0, src=0}]\n   1: A       A............................................... a: 01.03 00:01 [01.03 01:01]\n"];
	node1:slot1 -> node2:slot8 [label="AT ",tooltip="probability of use=0\nMEAT=2024-03-01 00:02\n   0: A       A...............................................                               d: 01.03 00:01 [01.03 01:01]  [{name=AT , day=2024-03-01, id=AT0, src=0}]\n   1: T       T............................................... a: 01.03 00:02 [01.03 01:02]\n"];
	node1:slot2 -> node2:slot9 [label="AT ",tooltip="probability of use=0.995392\nMEAT=2024-03-01 00:31\n   0: A       A...............................................                               d: 01.03 00:29 [01.03 01:29]  [{name=AT , day=2024-03-01, id=AT1, src=0}]\n   1: T       T............................................... a: 01.03 00:31 [01.03 01:31]\n"];
	node1:slot3 -> node2:slot10 [label="AT ",tooltip="probability of use=0.00126436\nMEAT=2024-03-01 00:32\n   0: A       A...............................................                               d: 01.03 00:30 [01.03 01:30]  [{name=AT , day=2024-03-01, id=AT2, src=0}]\n   1: T       T............................................... a: 01.03 00:32 [01.03 01:32]\n"];
	node1:slot4 -> node2:slot11 [label="AT ",tooltip="probability of use=0.00118278\nMEAT=2024-03-01 00:33\n   0: A       A...............................................                               d: 01.03 00:31 [01.03 01:31]  [{name=AT , day=2024-03-01, id=AT3, src=0}]\n   1: T       T............................................... a: 01.03 00:33 [01.03 01:33]\n"];
	node1:slot5 -> node2:slot12 [label="AT ",tooltip="probability of use=0.00110886\nMEAT=2024-03-01 00:34\n   0: A       A...............................................                               d: 01.03 00:32 [01.03 01:32]  [{name=AT , day=2024-03-01, id=AT4, src=0}]\n   1: T       T............................................... a: 01.03 00:34 [01.03 01:34]\n"];
	node1:slot6 -> node2:slot13 [label="AT ",tooltip="probability of use=0.00104166\nMEAT=2024-03-01 00:35\n   0: A       A...............................................                               d: 01.03 00:33 [01.03 01:33]  [{name=AT , day=2024-03-01, id=AT5, src=0}]\n   1: T       T............................................... a: 01.03 00:35 [01.03 01:35]\n"];
	node1:slot7 -> node2:slot14 [label="AT ",tooltip="probability of use=1e-05\nMEAT=2024-03-01 00:37\n   0: A       A...............................................                               d: 01.03 00:36 [01.03 01:36]  [{name=AT , day=2024-03-01, id=AT6, src=0}]\n   1: T       T............................................... a: 01.03 00:37 [01.03 01:37]\n"];
}
)";

struct load_tt {
  load_tt(std::string_view tt) {
    src_ = source_idx_t{0};
    tt_.date_range_ = {date::sys_days{2024_y / March / 1},
                       date::sys_days{2024_y / March / 2}};
    register_special_stations(tt_);
    gtfs::load_timetable({}, src_, loader::mem_dir::read(tt), tt_);
    finalize(tt_);
  }
  source_idx_t src_;
  timetable tt_;
};

// Tests if graph is found / indirect if con_end = compute_safe_connection_end()
// is correct
TEST(MeatCsa, FinalConnOfStation) {
  auto ltt = load_tt{test_tt};
  auto state = mcsa::meat_csa_state<mcsa::dynamic_profile_set>{};
  auto meat = mcsa::meat_csa<mcsa::dynamic_profile_set>{
      ltt.tt_, state, day_idx_t{4}, routing::all_clasz_allowed()};
  auto g = m::decision_graph{};

  auto const from = "S";
  auto const to = "T";
  auto const start_time = *ltt.tt_.date_range_.begin();
  auto const start_location =
      ltt.tt_.locations_.location_id_to_idx_.at({from, source_idx_t{0}});
  auto const end_location =
      ltt.tt_.locations_.location_id_to_idx_.at({to, source_idx_t{0}});
  auto const prf_idx = profile_idx_t{0};

  meat.execute(start_time, start_location, end_location, prf_idx, g);

  auto r = m::expanded_representation{g};
  auto ss = std::stringstream{};
  m::write_dot(ss, ltt.tt_, g, r);
  EXPECT_EQ(ss.str(), expanded_dot_graph)
      << ss.str() << "\n " << expanded_dot_graph;
}

// Timetable:
// Connections
// S -> A: 00:00 - 00:01
// A -> T: 00:36 - 00:37
// A -> B: 00:04 - 00:05
// B -> T: 00:07 - 00:08
//         00:10 - 00:11
//         00:41 - 00:42
constexpr auto const test2_tt = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
S,S,,0.0,0.0,,
A,A,,1.0,0.0,,
B,B,,1.0,0.0,,
T,T,,2.0,0.0,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
SA,DB,SA,,,2
AT,DB,AT,,,2
AB,DB,AB,,,2
BT,DB,BT,,,2

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
SA,S1,SA0,,
AT,S1,AT0,,
AB,S1,AB0,,
BT,S1,BT0,,
BT,S1,BT1,,
BT,S1,BT2,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
SA0,01:00:00,01:00:00,S,0
SA0,01:01:00,01:01:00,A,1
AT0,01:36:00,01:36:00,A,0
AT0,01:37:00,01:37:00,T,1
AB0,01:04:00,01:04:00,A,0
AB0,01:05:00,01:05:00,B,1
BT0,01:07:00,01:07:00,B,0
BT0,01:08:00,01:08:00,T,1
BT1,01:10:00,01:10:00,B,0
BT1,01:11:00,01:11:00,T,1
BT2,01:41:00,01:41:00,B,0
BT2,01:42:00,01:42:00,T,1

# calendar_dates.txt
service_id,date,exception_type
S1,20240301,1
)"sv;

constexpr auto const expanded_dot_graph_a_1 = R"(digraph decision_graph{
	splines=polyline;rankdir=LR;
	node0[shape=record,tooltip="(S, S)\ntransfer time=00:02.0",label="S|<slot0>2024-03-01 00:00"];
	node1[shape=record,tooltip="(A, A)\ntransfer time=00:02.0",label="A|<slot1>2024-03-01 00:01|<slot2>2024-03-01 00:36"];
	node2[shape=record,tooltip="(T, T)\ntransfer time=00:02.0",label="T|<slot3>2024-03-01 00:37"];
	node0:slot0 -> node1:slot1 [label="SA ",tooltip="probability of use=1\nMEAT=2024-03-01 00:37\n   0: S       S...............................................                               d: 01.03 00:00 [01.03 01:00]  [{name=SA , day=2024-03-01, id=SA0, src=0}]\n   1: A       A............................................... a: 01.03 00:01 [01.03 01:01]\n"];
	node1:slot2 -> node2:slot3 [label="AT ",tooltip="probability of use=1\nMEAT=2024-03-01 00:37\n   0: A       A...............................................                               d: 01.03 00:36 [01.03 01:36]  [{name=AT , day=2024-03-01, id=AT0, src=0}]\n   1: T       T............................................... a: 01.03 00:37 [01.03 01:37]\n"];
}
)";

constexpr auto const expanded_dot_graph_a_2 = R"(digraph decision_graph{
	splines=polyline;rankdir=LR;
	node0[shape=record,tooltip="(S, S)\ntransfer time=00:02.0",label="S|<slot0>2024-03-01 00:00"];
	node1[shape=record,tooltip="(A, A)\ntransfer time=00:02.0",label="A|<slot1>2024-03-01 00:01|<slot2>2024-03-01 00:04|<slot3>2024-03-01 00:36"];
	node2[shape=record,tooltip="(B, B)\ntransfer time=00:02.0",label="B|<slot4>2024-03-01 00:05|<slot5>2024-03-01 00:07|<slot6>2024-03-01 00:10|<slot7>2024-03-01 00:41"];
	node3[shape=record,tooltip="(T, T)\ntransfer time=00:02.0",label="T|<slot8>2024-03-01 00:08|<slot9>2024-03-01 00:11|<slot10>2024-03-01 00:37|<slot11>2024-03-01 00:42"];
	node0:slot0 -> node1:slot1 [label="SA ",tooltip="probability of use=1\nMEAT=2024-03-01 00:19\n   0: S       S...............................................                               d: 01.03 00:00 [01.03 01:00]  [{name=SA , day=2024-03-01, id=SA0, src=0}]\n   1: A       A............................................... a: 01.03 00:01 [01.03 01:01]\n"];
	node1:slot2 -> node2:slot4 [label="AB ",tooltip="probability of use=0.758326\nMEAT=2024-03-01 00:13\n   0: A       A...............................................                               d: 01.03 00:04 [01.03 01:04]  [{name=AB , day=2024-03-01, id=AB0, src=0}]\n   1: B       B............................................... a: 01.03 00:05 [01.03 01:05]\n"];
	node1:slot3 -> node3:slot10 [label="AT ",tooltip="probability of use=0.241674\nMEAT=2024-03-01 00:37\n   0: A       A...............................................                               d: 01.03 00:36 [01.03 01:36]  [{name=AT , day=2024-03-01, id=AT0, src=0}]\n   1: T       T............................................... a: 01.03 00:37 [01.03 01:37]\n"];
	node2:slot5 -> node3:slot8 [label="BT ",tooltip="probability of use=0.505551\nMEAT=2024-03-01 00:08\n   0: B       B...............................................                               d: 01.03 00:07 [01.03 01:07]  [{name=BT , day=2024-03-01, id=BT0, src=0}]\n   1: T       T............................................... a: 01.03 00:08 [01.03 01:08]\n"];
	node2:slot6 -> node3:slot9 [label="BT ",tooltip="probability of use=0.13902\nMEAT=2024-03-01 00:11\n   0: B       B...............................................                               d: 01.03 00:10 [01.03 01:10]  [{name=BT , day=2024-03-01, id=BT1, src=0}]\n   1: T       T............................................... a: 01.03 00:11 [01.03 01:11]\n"];
	node2:slot7 -> node3:slot11 [label="BT ",tooltip="probability of use=0.113755\nMEAT=2024-03-01 00:42\n   0: B       B...............................................                               d: 01.03 00:41 [01.03 01:41]  [{name=BT , day=2024-03-01, id=BT2, src=0}]\n   1: T       T............................................... a: 01.03 00:42 [01.03 01:42]\n"];
}
)";

TEST(MeatCsa, Alpha) {
  auto ltt = load_tt{test2_tt};
  auto const max_delay = 30;

  auto const from = "S";
  auto const to = "T";
  auto const start_time = *ltt.tt_.date_range_.begin();
  auto const start_location =
      ltt.tt_.locations_.location_id_to_idx_.at({from, source_idx_t{0}});
  auto const end_location =
      ltt.tt_.locations_.location_id_to_idx_.at({to, source_idx_t{0}});
  auto const prf_idx = profile_idx_t{0};

  auto ss = std::stringstream{};
  {
    auto alpha = 1.12;
    auto state = mcsa::meat_csa_state<mcsa::dynamic_profile_set>{};
    auto meat = mcsa::meat_csa<mcsa::dynamic_profile_set>{
        ltt.tt_,   state, day_idx_t{4}, routing::all_clasz_allowed(),
        max_delay, alpha};
    auto g = m::decision_graph{};

    meat.execute(start_time, start_location, end_location, prf_idx, g);

    auto r = m::expanded_representation{g};
    m::write_dot(ss, ltt.tt_, g, r);
    m::write_dot(std::cout, ltt.tt_, g, r);
    EXPECT_EQ(ss.str(), expanded_dot_graph_a_1)
        << ss.str() << "\n " << expanded_dot_graph_a_1;
  }

  {
    auto s = std::stringstream{};
    auto alpha = 1.12;
    auto state = mcsa::meat_csa_state<mcsa::dynamic_profile_set>{};
    auto meat = mcsa::meat_csa<mcsa::dynamic_profile_set>{
        ltt.tt_,   state, day_idx_t{6}, routing::all_clasz_allowed(),
        max_delay, alpha};
    auto g = m::decision_graph{};

    meat.execute(start_time, start_location, end_location, prf_idx, g);

    auto r = m::expanded_representation{g};
    m::write_dot(s, ltt.tt_, g, r);
    m::write_dot(std::cout, ltt.tt_, g, r);
    EXPECT_EQ(s.str(), expanded_dot_graph_a_1)
        << s.str() << "\n " << expanded_dot_graph_a_1;
  }

  auto ss2 = std::stringstream{};
  {
    auto alpha = 1.14;
    auto state = mcsa::meat_csa_state<mcsa::dynamic_profile_set>{};
    auto meat = mcsa::meat_csa<mcsa::dynamic_profile_set>{
        ltt.tt_,   state, day_idx_t{4}, routing::all_clasz_allowed(),
        max_delay, alpha};
    auto g = m::decision_graph{};

    meat.execute(start_time, start_location, end_location, prf_idx, g);

    auto r = m::expanded_representation{g};

    m::write_dot(ss2, ltt.tt_, g, r);
    m::write_dot(std::cout, ltt.tt_, g, r);
    EXPECT_EQ(ss2.str(), expanded_dot_graph_a_2)
        << ss2.str() << "\n " << expanded_dot_graph_a_2;
  }

  auto ss3 = std::stringstream{};
  {
    auto alpha = 1.0;
    auto state = mcsa::meat_csa_state<mcsa::dynamic_profile_set>{};
    auto meat = mcsa::meat_csa<mcsa::dynamic_profile_set>{
        ltt.tt_,   state, day_idx_t{4}, routing::all_clasz_allowed(),
        max_delay, alpha};
    auto g = m::decision_graph{};

    meat.execute(start_time, start_location, end_location, prf_idx, g);

    auto r = m::expanded_representation{g};

    m::write_dot(ss3, ltt.tt_, g, r);
    m::write_dot(std::cout, ltt.tt_, g, r);
    EXPECT_EQ(ss3.str(), expanded_dot_graph_a_1)
        << ss3.str() << "\n " << expanded_dot_graph_a_1;
  }

  auto ss4 = std::stringstream{};
  {
    auto alpha = std::numeric_limits<double>::max();
    auto state = mcsa::meat_csa_state<mcsa::dynamic_profile_set>{};
    auto meat = mcsa::meat_csa<mcsa::dynamic_profile_set>{
        ltt.tt_,   state, day_idx_t{4}, routing::all_clasz_allowed(),
        max_delay, alpha};
    auto g = m::decision_graph{};

    meat.execute(start_time, start_location, end_location, prf_idx, g);

    auto r = m::expanded_representation{g};

    m::write_dot(ss4, ltt.tt_, g, r);
    m::write_dot(std::cout, ltt.tt_, g, r);
    EXPECT_EQ(ss4.str(), expanded_dot_graph_a_2)
        << ss4.str() << "\n " << expanded_dot_graph_a_2;
  }

  auto ss5 = std::stringstream{};
  {
    auto alpha = 1000.0;
    auto state = mcsa::meat_csa_state<mcsa::dynamic_profile_set>{};
    auto meat = mcsa::meat_csa<mcsa::dynamic_profile_set>{
        ltt.tt_,   state, day_idx_t{4}, routing::all_clasz_allowed(),
        max_delay, alpha};
    auto g = m::decision_graph{};

    meat.execute(start_time, start_location, end_location, prf_idx, g);

    auto r = m::expanded_representation{g};

    m::write_dot(ss5, ltt.tt_, g, r);
    m::write_dot(std::cout, ltt.tt_, g, r);
    EXPECT_EQ(ss5.str(), expanded_dot_graph_a_2)
        << ss5.str() << "\n " << expanded_dot_graph_a_2;
  }

  EXPECT_EQ(ss.str(), ss3.str());
  EXPECT_EQ(ss2.str(), ss4.str());
  EXPECT_EQ(ss2.str(), ss5.str());
  EXPECT_NE(ss.str(), ss2.str());
}
