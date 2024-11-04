#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/agency.h"
#include "nigiri/loader/gtfs/calendar.h"
#include "nigiri/loader/gtfs/calendar_date.h"
#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/gtfs/local_to_utc.h"
#include "nigiri/loader/gtfs/noon_offsets.h"
#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"

#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

#include "nigiri/routing/meat/compact_representation.h"
#include "nigiri/routing/meat/expanded_representation.h"
#include "nigiri/routing/meat/raptor/meat_raptor.h"

#include "../../../loader/hrd/hrd_timetable.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::test_data::hrd_timetable;
using namespace std::string_view_literals;

namespace m = routing::meat;
namespace mraptor = m::raptor;

namespace nigiri {
struct timetable;
struct rt_timetable;
}  // namespace nigiri

namespace {

// Timetable:
// Connections
// S -> A: 00:00 - 00:01
// A -> T: 00:01 - 00:02
//         00:11 - 00:12
//         00:21 - 00:22
//         00:31 - 00:32
//         00:41 - 00:42
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

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
SA0,01:00:00,01:00:00,S,0
SA0,01:01:00,01:01:00,A,1
AT0,01:01:00,01:01:00,A,0
AT0,01:02:00,01:02:00,T,1
AT1,01:11:00,01:11:00,A,0
AT1,01:12:00,01:12:00,T,1
AT2,01:21:00,01:21:00,A,0
AT2,01:22:00,01:22:00,T,1
AT3,01:31:00,01:31:00,A,0
AT3,01:32:00,01:32:00,T,1
AT4,01:41:00,01:41:00,A,0
AT4,01:42:00,01:42:00,T,1

# calendar_dates.txt
service_id,date,exception_type
S1,20240301,1
)"sv;

constexpr auto const test_tt_expanded_dot_graph = R"(digraph decision_graph{
	splines=polyline;rankdir=LR;
	node0[shape=record,tooltip="(S, S)\ntransfer time=00:02.0",label="S|<slot0>2024-03-01 00:00"];
	node1[shape=record,tooltip="(A, A)\ntransfer time=00:02.0",label="A|<slot1>2024-03-01 00:01|<slot2>2024-03-01 00:11|<slot3>2024-03-01 00:21|<slot4>2024-03-01 00:31|<slot5>2024-03-01 00:41"];
	node2[shape=record,tooltip="(T, T)\ntransfer time=00:02.0",label="T|<slot6>2024-03-01 00:02|<slot7>2024-03-01 00:12|<slot8>2024-03-01 00:22|<slot9>2024-03-01 00:32|<slot10>2024-03-01 00:42"];
	node0:slot0 -> node1:slot1 [label="SA ",tooltip="probability of use=1\nMEAT=2024-03-01 00:12\n   0: S       S...............................................                               d: 01.03 00:00 [01.03 01:00]  [{name=SA , day=2024-03-01, id=SA0, src=0}]\n   1: A       A............................................... a: 01.03 00:01 [01.03 01:01]\n"];
	node1:slot1 -> node2:slot6 [label="AT ",tooltip="probability of use=0\nMEAT=2024-03-01 00:02\n   0: A       A...............................................                               d: 01.03 00:01 [01.03 01:01]  [{name=AT , day=2024-03-01, id=AT0, src=0}]\n   1: T       T............................................... a: 01.03 00:02 [01.03 01:02]\n"];
	node1:slot2 -> node2:slot7 [label="AT ",tooltip="probability of use=0.933324\nMEAT=2024-03-01 00:12\n   0: A       A...............................................                               d: 01.03 00:11 [01.03 01:11]  [{name=AT , day=2024-03-01, id=AT1, src=0}]\n   1: T       T............................................... a: 01.03 00:12 [01.03 01:12]\n"];
	node1:slot3 -> node2:slot8 [label="AT ",tooltip="probability of use=0.0476186\nMEAT=2024-03-01 00:22\n   0: A       A...............................................                               d: 01.03 00:21 [01.03 01:21]  [{name=AT , day=2024-03-01, id=AT2, src=0}]\n   1: T       T............................................... a: 01.03 00:22 [01.03 01:22]\n"];
	node1:slot4 -> node2:slot9 [label="AT ",tooltip="probability of use=0.0168969\nMEAT=2024-03-01 00:32\n   0: A       A...............................................                               d: 01.03 00:31 [01.03 01:31]  [{name=AT , day=2024-03-01, id=AT3, src=0}]\n   1: T       T............................................... a: 01.03 00:32 [01.03 01:32]\n"];
	node1:slot5 -> node2:slot10 [label="AT ",tooltip="probability of use=0.00216052\nMEAT=2024-03-01 00:42\n   0: A       A...............................................                               d: 01.03 00:41 [01.03 01:41]  [{name=AT , day=2024-03-01, id=AT4, src=0}]\n   1: T       T............................................... a: 01.03 00:42 [01.03 01:42]\n"];
}
)";

constexpr auto const expanded_dot_graph = R"(digraph decision_graph{
	splines=polyline;rankdir=LR;
	node0[shape=record,tooltip="(A, 0000001)\ntransfer time=00:02.0",label="A|<slot0>2020-03-30 05:00"];
	node1[shape=record,tooltip="(B, 0000002)\ntransfer time=00:02.0",label="B|<slot1>2020-03-30 06:00|<slot2>2020-03-30 06:15|<slot3>2020-03-30 06:30|<slot4>2020-03-30 06:45"];
	node2[shape=record,tooltip="(C, 0000003)\ntransfer time=00:02.0",label="C|<slot5>2020-03-30 07:00|<slot6>2020-03-30 07:15|<slot7>2020-03-30 07:30|<slot8>2020-03-30 07:45"];
	node0:slot0 -> node1:slot1 [label="RE 1337",tooltip="probability of use=1\nMEAT=2020-03-30 07:15\n   0: 0000001 A...............................................                               d: 30.03 05:00 [30.03 07:00]  [{name=RE 1337, day=2020-03-30, id=1337/0000001/300/0000002/360/, src=0}]\n   1: 0000002 B............................................... a: 30.03 06:00 [30.03 08:00]\n"];
	node1:slot1 -> node2:slot5 [label="RE 7331",tooltip="probability of use=0\nMEAT=2020-03-30 07:00\n   0: 0000002 B...............................................                               d: 30.03 06:00 [30.03 08:00]  [{name=RE 7331, day=2020-03-30, id=7331/0000002/360/0000003/420/, src=0}]\n   1: 0000003 C............................................... a: 30.03 07:00 [30.03 09:00]\n"];
	node1:slot2 -> node2:slot6 [label="RE 7331",tooltip="probability of use=0.964574\nMEAT=2020-03-30 07:15\n   0: 0000002 B...............................................                               d: 30.03 06:15 [30.03 08:15]  [{name=RE 7331, day=2020-03-30, id=7331/0000002/375/0000003/435/, src=0}]\n   1: 0000003 C............................................... a: 30.03 07:15 [30.03 09:15]\n"];
	node1:slot3 -> node2:slot7 [label="RE 7331",tooltip="probability of use=0.0332658\nMEAT=2020-03-30 07:30\n   0: 0000002 B...............................................                               d: 30.03 06:30 [30.03 08:30]  [{name=RE 7331, day=2020-03-30, id=7331/0000002/390/0000003/450/, src=0}]\n   1: 0000003 C............................................... a: 30.03 07:30 [30.03 09:30]\n"];
	node1:slot4 -> node2:slot8 [label="RE 7331",tooltip="probability of use=0.00216052\nMEAT=2020-03-30 07:45\n   0: 0000002 B...............................................                               d: 30.03 06:45 [30.03 08:45]  [{name=RE 7331, day=2020-03-30, id=7331/0000002/405/0000003/465/, src=0}]\n   1: 0000003 C............................................... a: 30.03 07:45 [30.03 09:45]\n"];
}
)";

struct load_tt {
  load_tt() {
    src_ = source_idx_t{0U};
    tt_.date_range_ = full_period();
    load_timetable(src_, loader::hrd::hrd_5_20_26, files_abc(), tt_);
    finalize(tt_);
  }
  source_idx_t src_;
  timetable tt_;
};

}  // namespace

class MeatRaptorProfileTest : public testing::Test, public load_tt {
protected:
  MeatRaptorProfileTest()
      : state_{},
        meat_raptor_{tt_, state_, day_idx_t{4}, routing::all_clasz_allowed()} {}

  template <typename MEAT>
  void execute(m::decision_graph& g, MEAT& meat) {
    auto const from = "0000001";
    auto const to = "0000003";
    auto const start_time = unixtime_t{sys_days{2020_y / March / 30}} + 5_hours;
    auto const start_location =
        tt_.locations_.location_id_to_idx_.at({from, src_});
    auto const end_location = tt_.locations_.location_id_to_idx_.at({to, src_});
    auto const prf_idx = profile_idx_t{0};
    meat.execute(start_time, start_location, end_location, prf_idx, g);
  }
  m::decision_graph meat_raptor_execute() {
    auto g = m::decision_graph{};
    execute(g, meat_raptor_);
    return g;
  }

  void expect_eq(m::decision_graph const& g1, m::decision_graph const& g2) {
    auto r1 = m::expanded_representation{g1};
    auto r2 = m::expanded_representation{g2};
    auto ss1 = std::stringstream{};
    auto ss2 = std::stringstream{};
    m::write_dot(ss1, tt_, g1, r1);
    m::write_dot(ss2, tt_, g2, r2);
    EXPECT_EQ(ss1.str(), ss2.str()) << ss1.str() << "\n " << ss2.str();
  }
  void expect_correct(m::decision_graph const& g) {
    expect_correct(g, expanded_dot_graph, tt_);
  }
  void expect_correct(m::decision_graph const& g,
                      std::string_view const sv,
                      timetable const& tt) {
    auto r = m::expanded_representation{g};
    auto ss = std::stringstream{};
    m::write_dot(ss, tt, g, r);
    EXPECT_EQ(ss.str(), sv) << ss.str() << "\n " << sv;
  }

  mraptor::meat_raptor_state state_;
  mraptor::meat_raptor meat_raptor_;
};

TEST_F(MeatRaptorProfileTest, CorrectOnSimpleRun) {
  auto g = meat_raptor_execute();

  expect_correct(g);

  auto ea = state_.profile_set_.compute_entry_amount();
  auto ne = state_.profile_set_.n_entry_idxs();
  EXPECT_EQ(ea, 5);
  EXPECT_TRUE(ne > ea);

  meat_raptor_.next_start_time();
  ea = state_.profile_set_.compute_entry_amount();
  ne = state_.profile_set_.n_entry_idxs();
  EXPECT_EQ(ea, 0);
  EXPECT_EQ(ne, 0);
}

TEST_F(MeatRaptorProfileTest, SameOnSeveralRuns) {
  auto g = meat_raptor_execute();
  auto ea = state_.profile_set_.compute_entry_amount();
  auto ne = state_.profile_set_.n_entry_idxs();

  meat_raptor_.next_start_time();
  auto g_2 = meat_raptor_execute();
  auto ea_2 = state_.profile_set_.compute_entry_amount();
  auto ne_2 = state_.profile_set_.n_entry_idxs();

  expect_eq(g, g_2);
  EXPECT_EQ(ea, ea_2);
  EXPECT_EQ(ne, ne_2);

  meat_raptor_.next_start_time();
  auto g_3 = meat_raptor_execute();
  auto ea_3 = state_.profile_set_.compute_entry_amount();
  auto ne_3 = state_.profile_set_.n_entry_idxs();

  expect_eq(g, g_3);
  EXPECT_EQ(ea, ea_3);
  EXPECT_EQ(ne, ne_3);
}

TEST_F(MeatRaptorProfileTest, SameOnNewMeatCsaRouter) {
  auto g = meat_raptor_execute();
  auto ea = state_.profile_set_.compute_entry_amount();
  auto ne = state_.profile_set_.n_entry_idxs();

  auto meat_raptor = mraptor::meat_raptor{tt_, state_, day_idx_t{4},
                                          routing::all_clasz_allowed()};
  EXPECT_EQ(state_.profile_set_.compute_entry_amount(), 0);
  auto g_2 = m::decision_graph{};

  execute(g_2, meat_raptor);
  auto ea_2 = state_.profile_set_.compute_entry_amount();
  auto ne_2 = state_.profile_set_.n_entry_idxs();

  expect_eq(g, g_2);
  EXPECT_EQ(ea, ea_2);
  EXPECT_EQ(ne, ne_2);
}

TEST_F(MeatRaptorProfileTest, SameProfileOnNewTT) {
  auto g = meat_raptor_execute();

  // load new tt
  auto tt = timetable{};
  tt.date_range_ = {date::sys_days{2024_y / March / 1},
                    date::sys_days{2024_y / March / 2}};
  register_special_stations(tt);
  gtfs::load_timetable({}, source_idx_t{0}, loader::mem_dir::read(test_tt), tt);
  finalize(tt);

  auto meat_raptor = mraptor::meat_raptor{tt, state_, day_idx_t{4},
                                          routing::all_clasz_allowed()};
  EXPECT_EQ(state_.profile_set_.compute_entry_amount(), 0);
  auto g_2 = m::decision_graph{};

  auto const from = "S";
  auto const to = "T";
  auto const start_time = *tt.date_range_.begin();
  auto const start_location =
      tt.locations_.location_id_to_idx_.at({from, source_idx_t{0}});
  auto const end_location =
      tt.locations_.location_id_to_idx_.at({to, source_idx_t{0}});
  auto const prf_idx = profile_idx_t{0};

  meat_raptor.execute(start_time, start_location, end_location, prf_idx, g_2);
  // auto ea_2 = state_.profile_set_.compute_entry_amount();

  expect_correct(g_2, test_tt_expanded_dot_graph, tt);
}

// TODO remove
TEST(MeatRaptor, Simple) {
  constexpr auto const src = source_idx_t{0U};

  timetable tt;
  tt.date_range_ = full_period();
  load_timetable(src, loader::hrd::hrd_5_20_26, files_abc(), tt);
  finalize(tt);

  auto from = "0000001";
  auto to = "0000003";
  auto start_time = unixtime_t{sys_days{2020_y / March / 30}} + 5_hours;
  auto start_location = tt.locations_.location_id_to_idx_.at({from, src});
  auto end_location = tt.locations_.location_id_to_idx_.at({to, src});
  auto prf_idx = profile_idx_t{0};
  auto state = mraptor::meat_raptor_state{};
  auto meat = mraptor::meat_raptor(tt, state, day_idx_t{4},
                                   routing::all_clasz_allowed());
  auto g = m::decision_graph{};
  meat.execute(start_time, start_location, end_location, prf_idx, g);
  meat.next_start_time();
  meat.execute(start_time, start_location, end_location, prf_idx, g);
  auto r = m::expanded_representation{g};
  auto r2 = m::compact_representation{g};
  auto ss1 = std::stringstream{};
  m::write_dot(ss1, tt, g, r);
  m::write_dot(std::cout, tt, g, r);
  (void)r2;
  // m::write_dot(std::cout, tt, g, r2);

  EXPECT_EQ(ss1.str(), ss1.str());
}