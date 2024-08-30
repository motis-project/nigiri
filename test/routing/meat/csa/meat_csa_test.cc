#include "gtest/gtest.h"

#include "nigiri/loader/hrd/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/routing/meat/compact_representation.h"
#include "nigiri/routing/meat/csa/meat_csa.h"
#include "nigiri/routing/meat/expanded_representation.h"

#include "../../../loader/hrd/hrd_timetable.h"

using namespace date;
using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::test_data::hrd_timetable;

namespace m = routing::meat;
namespace mcsa = m::csa;

namespace nigiri {
struct timetable;
struct rt_timetable;
}  // namespace nigiri

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

class meat_csa_profiles_test : public testing::Test, public load_tt {
protected:
  meat_csa_profiles_test()
      : state_static_{},
        state_dynamic_growth_{},
        state_dynamic_{},
        meat_static_{tt_, state_static_, day_idx_t{4},
                     routing::all_clasz_allowed()},
        meat_dynamic_growth_{tt_, state_dynamic_growth_, day_idx_t{4},
                             routing::all_clasz_allowed()},
        meat_dynamic_{tt_, state_dynamic_, day_idx_t{4},
                      routing::all_clasz_allowed()} {}

  template <typename MEAT>
  void execute(m::decision_graph& g, MEAT& meat) {
    auto from = "0000001";
    auto to = "0000003";
    auto start_time = unixtime_t{sys_days{2020_y / March / 30}} + 5_hours;
    auto start_location = tt_.locations_.location_id_to_idx_.at({from, src_});
    auto end_location = tt_.locations_.location_id_to_idx_.at({to, src_});
    auto prf_idx = profile_idx_t{0};
    meat.execute(start_time, start_location, end_location, prf_idx, g);
  }
  m::decision_graph meat_static_execute() {
    auto g = m::decision_graph{};
    execute(g, meat_static_);
    return g;
  }
  m::decision_graph meat_dynamic_growth_execute() {
    auto g = m::decision_graph{};
    execute(g, meat_dynamic_growth_);
    return g;
  }
  m::decision_graph meat_dynamic_execute() {
    auto g = m::decision_graph{};
    execute(g, meat_dynamic_);
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
    auto r = m::expanded_representation{g};
    auto ss = std::stringstream{};
    m::write_dot(ss, tt_, g, r);
    EXPECT_EQ(ss.str(), expanded_dot_graph)
        << ss.str() << "\n " << expanded_dot_graph;
  }

  mcsa::meat_csa_state<mcsa::static_profile_set> state_static_;
  mcsa::meat_csa_state<mcsa::dynamic_growth_profile_set> state_dynamic_growth_;
  mcsa::meat_csa_state<mcsa::dynamic_profile_set> state_dynamic_;
  mcsa::meat_csa<mcsa::static_profile_set> meat_static_;
  mcsa::meat_csa<mcsa::dynamic_growth_profile_set> meat_dynamic_growth_;
  mcsa::meat_csa<mcsa::dynamic_profile_set> meat_dynamic_;
};

TEST_F(meat_csa_profiles_test, same_and_correct_on_simple_run) {
  auto g_s = meat_static_execute();
  auto g_dg = meat_dynamic_growth_execute();
  auto g_d = meat_dynamic_execute();

  expect_correct(g_s);
  expect_correct(g_dg);
  expect_correct(g_d);
  expect_eq(g_s, g_dg);
  expect_eq(g_d, g_dg);
}

TEST_F(meat_csa_profiles_test, same_on_several_runs) {
  // test static_profile_set
  auto g_s = meat_static_execute();
  auto ea_s = state_static_.profile_set_.compute_entry_amount();
  auto ne_s = state_static_.profile_set_.n_entry_idxs();

  meat_static_.next_start_time();
  auto g_s2 = meat_static_execute();
  auto ea_s2 = state_static_.profile_set_.compute_entry_amount();
  auto ne_s2 = state_static_.profile_set_.n_entry_idxs();

  expect_eq(g_s, g_s2);
  EXPECT_EQ(ea_s, ea_s2);
  EXPECT_EQ(ne_s, ne_s2);

  meat_static_.next_start_time();
  auto g_s3 = meat_static_execute();
  auto ea_s3 = state_static_.profile_set_.compute_entry_amount();
  auto ne_s3 = state_static_.profile_set_.n_entry_idxs();

  expect_eq(g_s, g_s3);
  EXPECT_EQ(ea_s, ea_s3);
  EXPECT_EQ(ne_s, ne_s3);

  // test dynamic_growth_profile_set
  auto g_dg = meat_dynamic_growth_execute();
  auto ea_dg = state_dynamic_growth_.profile_set_.compute_entry_amount();
  auto ne_dg = state_dynamic_growth_.profile_set_.n_entry_idxs();

  meat_dynamic_growth_.next_start_time();
  auto g_dg2 = meat_dynamic_growth_execute();
  auto ea_dg2 = state_dynamic_growth_.profile_set_.compute_entry_amount();
  auto ne_dg2 = state_dynamic_growth_.profile_set_.n_entry_idxs();

  expect_eq(g_dg, g_dg2);
  EXPECT_EQ(ea_dg, ea_dg2);
  EXPECT_EQ(ne_dg, ne_dg2);

  meat_dynamic_growth_.next_start_time();
  auto g_dg3 = meat_dynamic_growth_execute();
  auto ea_dg3 = state_dynamic_growth_.profile_set_.compute_entry_amount();
  auto ne_dg3 = state_dynamic_growth_.profile_set_.n_entry_idxs();

  expect_eq(g_dg, g_dg3);
  EXPECT_EQ(ea_dg, ea_dg3);
  EXPECT_EQ(ne_dg, ne_dg3);

  // test dynamic_profile_set
  auto g_d = meat_dynamic_execute();
  auto ea_d = state_dynamic_.profile_set_.compute_entry_amount();
  auto ne_d = state_dynamic_.profile_set_.n_entry_idxs();

  meat_dynamic_.next_start_time();
  auto g_d2 = meat_dynamic_execute();
  auto ea_d2 = state_dynamic_.profile_set_.compute_entry_amount();
  auto ne_d2 = state_dynamic_.profile_set_.n_entry_idxs();

  expect_eq(g_d, g_d2);
  EXPECT_EQ(ea_d, ea_d2);
  EXPECT_EQ(ne_d, ne_d2);

  meat_dynamic_.next_start_time();
  auto g_d3 = meat_dynamic_execute();
  auto ea_d3 = state_dynamic_.profile_set_.compute_entry_amount();
  auto ne_d3 = state_dynamic_.profile_set_.n_entry_idxs();

  expect_eq(g_d, g_d3);
  EXPECT_EQ(ea_d, ea_d3);
  EXPECT_EQ(ne_d, ne_d3);
}

TEST_F(meat_csa_profiles_test, same_on_new_meat_csa_router) {
  auto g_s = meat_static_execute();
  auto ea_s = state_static_.profile_set_.compute_entry_amount();
  auto ne_s = state_static_.profile_set_.n_entry_idxs();
  auto g_dg = meat_dynamic_growth_execute();
  auto ea_dg = state_dynamic_growth_.profile_set_.compute_entry_amount();
  auto ne_dg = state_dynamic_growth_.profile_set_.n_entry_idxs();
  auto g_d = meat_dynamic_execute();
   auto ea_d = state_dynamic_.profile_set_.compute_entry_amount();
  auto ne_d = state_dynamic_.profile_set_.n_entry_idxs();

  auto meat_static = mcsa::meat_csa<mcsa::static_profile_set>{
      tt_, state_static_, day_idx_t{4}, routing::all_clasz_allowed()};
  auto meat_dynamic_growth = mcsa::meat_csa<mcsa::dynamic_growth_profile_set>{
      tt_, state_dynamic_growth_, day_idx_t{4}, routing::all_clasz_allowed()};
  auto meat_dynamic = mcsa::meat_csa<mcsa::dynamic_profile_set>{
      tt_, state_dynamic_, day_idx_t{4}, routing::all_clasz_allowed()};
  auto g_s2 = m::decision_graph{};
  auto g_dg2 = m::decision_graph{};
  auto g_d2 = m::decision_graph{};

  execute(g_s2, meat_static);
  auto ea_s2 = state_static_.profile_set_.compute_entry_amount();
  auto ne_s2 = state_static_.profile_set_.n_entry_idxs();
  execute(g_dg2, meat_dynamic_growth);
  auto ea_dg2 = state_dynamic_growth_.profile_set_.compute_entry_amount();
  auto ne_dg2 = state_dynamic_growth_.profile_set_.n_entry_idxs();
  execute(g_d2, meat_dynamic);
  auto ea_d2 = state_dynamic_.profile_set_.compute_entry_amount();
  auto ne_d2 = state_dynamic_.profile_set_.n_entry_idxs();

  expect_eq(g_s, g_s2);
  EXPECT_EQ(ea_s, ea_s2);
  EXPECT_EQ(ne_s, ne_s2);
  expect_eq(g_dg, g_dg2);
  EXPECT_EQ(ea_dg, ea_dg2);
  EXPECT_EQ(ne_dg, ne_dg2);
  expect_eq(g_d, g_d2);
  EXPECT_EQ(ea_d, ea_d2);
  EXPECT_EQ(ne_d, ne_d2);
}

TEST(meat_csa, simple) {
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
  auto state = mcsa::meat_csa_state<mcsa::dynamic_growth_profile_set>{};
  auto meat = mcsa::meat_csa<mcsa::dynamic_growth_profile_set>(
      tt, state, day_idx_t{4}, routing::all_clasz_allowed());
  auto g = m::decision_graph{};
  meat.execute(start_time, start_location, end_location, prf_idx, g);
  meat.next_start_time();
  meat.execute(start_time, start_location, end_location, prf_idx, g);
  auto r = m::expanded_representation{g};
  auto r2 = m::compact_representation{g};
  auto ss1 = std::stringstream{};
  m::write_dot(ss1, tt, g, r);
  m::write_dot(std::cout, tt, g, r);
  m::write_dot(std::cout, tt, g, r2);

  auto state2 = mcsa::meat_csa_state<mcsa::static_profile_set>{};
  auto meat2 = mcsa::meat_csa<mcsa::static_profile_set>(
      tt, state2, day_idx_t{4}, routing::all_clasz_allowed());
  auto g2 = m::decision_graph();
  meat2.execute(start_time, start_location, end_location, prf_idx, g2);
  meat2.next_start_time();
  meat2.execute(start_time, start_location, end_location, prf_idx, g2);
  auto r22 = m::expanded_representation{g2};
  auto ss2 = std::stringstream{};
  m::write_dot(ss2, tt, g2, r22);

  auto state3 = mcsa::meat_csa_state<mcsa::dynamic_profile_set>{};
  auto meat3 = mcsa::meat_csa<mcsa::dynamic_profile_set>(
      tt, state3, day_idx_t{4}, routing::all_clasz_allowed());
  auto g3 = m::decision_graph();
  meat3.execute(start_time, start_location, end_location, prf_idx, g3);
  meat3.next_start_time();
  meat3.execute(start_time, start_location, end_location, prf_idx, g3);
  auto r33 = m::expanded_representation{g3};
  auto ss3 = std::stringstream{};
  m::write_dot(ss3, tt, g3, r33);

  EXPECT_EQ(ss1.str(), ss2.str());
  EXPECT_EQ(ss3.str(), ss2.str());
  assert(ss1.str() == ss2.str());
  assert(ss3.str() == ss2.str());
}