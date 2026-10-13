#include "gtest/gtest.h"

#include <optional>
#include <string_view>

#include "nigiri/timetable.h"

#include "../transfer_rules_util.h"

// Replacing the walks of the default profile (osr_footpath: a street router
// knows better than the loader's beelines) changes nothing a rule states.

using namespace nigiri;
using nigiri::test::arrival;
using nigiri::test::at;
using nigiri::test::g_from_f;
using nigiri::test::gl_from_f;
using nigiri::test::hl_from_f;
using nigiri::test::rebuild_default_profile;

// test::network(): F arrives S1 10:30. G leaves S2 10:40 (fallback GL 11:10)
// to B, H leaves S1 10:31 (fallback HL 11:01) to C. Plus K, which leaves T
// 10:45 to E. T is 450 m from S: the same feed, not the same station, so the
// loader connects the two with nothing.
test::network_rows const kRows{
    .stops_ = {{"T", 50.0045, 6.5}, {"E", 50.0, 8.5}},
    .trips_ = {{"K", "RK", {{"T", "10:45"}, {"E", "11:05"}}}}};

// The arrival from A via K. A function, not a constant: at static init time
// the time zone database is not loaded yet.
unixtime_t via_k() { return at("11:05"); }

TEST(rebuild_default_profile, walk_replaces_beeline) {
  auto tt = test::load_network("", kRows);
  EXPECT_EQ(g_from_f(),
            arrival(tt, nullptr, {"A", "B"}));  // beeline S1 -> S2: 2 min
  rebuild_default_profile(tt, {{"S1", "S2", 12}});
  EXPECT_EQ(gl_from_f(), arrival(tt, nullptr, {"A", "B"}));
  rebuild_default_profile(tt, {{"S1", "S2", 9}});
  EXPECT_EQ(g_from_f(), arrival(tt, nullptr, {"A", "B"}));
}

TEST(rebuild_default_profile, walk_the_router_did_not_find) {
  auto tt = test::load_network("", kRows);
  rebuild_default_profile(tt, {});
  EXPECT_EQ(std::nullopt, arrival(tt, nullptr, {"A", "B"}));
  EXPECT_EQ(hl_from_f(),
            arrival(tt, nullptr, {"A", "C"}));  // changing at S1 is no walk
}

// A rule beats the walk between its stops, slower or faster than the rule -
// a ban of one trip pair as well, and a rule for a whole station, which applies
// to all pairs of its stops: the walk underneath has to go, the routing would
// take the faster one.
TEST(rebuild_default_profile, rule_beats_walk) {
  struct test_case {
    char const* rows_;
    int walk_;
    unixtime_t (*a_to_b_)();
  };
  for (auto const& c : std::initializer_list<test_case>{
           {"S1,S2,2,180,,,,", 12, g_from_f},
           {"S1,S2,2,900,,,,", 1, gl_from_f},
           {"S1,S2,2,120,,,,\nS1,S2,3,,,,F,G", 1, gl_from_f},
           {"S,S,2,900,,,,", 1, gl_from_f}}) {
    SCOPED_TRACE(c.rows_);
    auto tt = test::load_network(c.rows_, kRows);
    EXPECT_EQ(c.a_to_b_(), arrival(tt, nullptr, {"A", "B"}));
    EXPECT_EQ(hl_from_f(), arrival(tt, nullptr, {"A", "C"}));
    rebuild_default_profile(tt, {{"S1", "S2", c.walk_}});
    EXPECT_EQ(c.a_to_b_(), arrival(tt, nullptr, {"A", "B"}));
    EXPECT_EQ(hl_from_f(), arrival(tt, nullptr, {"A", "C"}));
  }
}

// F2 leaves S1 6 min after F arrives, the F -> F2 rule asks for 30 min: F
// stops at a location of its own below S1. It owns no walks: it gets those of
// S1, so they have to follow the new walks.
TEST(rebuild_default_profile, virtual_location_walks_like_its_stop) {
  auto tt = test::load_network("S1,S1,2,120,,,,\nS1,S1,2,1800,,,F,F2", kRows);
  ASSERT_EQ(std::nullopt, arrival(tt, nullptr, {"A", "D"}));
  EXPECT_EQ(hl_from_f(), arrival(tt, nullptr, {"A", "C"}));  // H: 1 min
  EXPECT_EQ(g_from_f(), arrival(tt, nullptr, {"A", "B"}));  // beeline
  rebuild_default_profile(tt, {{"S1", "S2", 12}});
  EXPECT_EQ(gl_from_f(), arrival(tt, nullptr, {"A", "B"}));
  EXPECT_EQ(hl_from_f(), arrival(tt, nullptr, {"A", "C"}));
  rebuild_default_profile(tt, {{"S1", "S2", 3}, {"S1", "T", 6}});
  EXPECT_EQ(g_from_f(), arrival(tt, nullptr, {"A", "B"}));
  EXPECT_EQ(via_k(), arrival(tt, nullptr, {"A", "E"}));
}

// A second rebuild changes nothing: the rule footpaths are built again from
// the rules in the timetable, with the same result.
TEST(rebuild_default_profile, is_repeatable) {
  auto tt = test::load_network("S,S,2,120,,,,\nS,S,2,1800,,,F,F2", kRows);
  ASSERT_EQ(std::nullopt, arrival(tt, nullptr, {"A", "D"}));

  rebuild_default_profile(tt, {{"S1", "T", 6}});
  auto const n_fps = tt.locations_.footpaths_out_[kDefaultProfile].data_.size();
  rebuild_default_profile(tt, {{"S1", "T", 6}});

  EXPECT_EQ(n_fps, tt.locations_.footpaths_out_[kDefaultProfile].data_.size());
  EXPECT_EQ(g_from_f(),
            arrival(tt, nullptr, {"A", "B"}));  // the station rule: 2 min
  EXPECT_EQ(hl_from_f(), arrival(tt, nullptr, {"A", "C"}));
  EXPECT_EQ(via_k(), arrival(tt, nullptr, {"A", "E"}));
}

// The timed F -> G transfer (0 min) matters only once the walk S1 -> S2 is
// longer than the 10 min between F and G: it is decided on the routed walk,
// not on the beeline (2 min).
TEST(rebuild_default_profile, rules_apply_to_routed_walks) {
  constexpr auto const kRule = "S1,S2,1,,,,F,G";
  ASSERT_EQ(0U, test::n_virts(test::load_network(kRule, kRows)));
  auto const tt =
      test::load_network_with_walks(kRule, kRows, {{"S1", "S2", 12}});
  EXPECT_NE(0U, test::n_virts(tt));
  EXPECT_EQ(g_from_f(), arrival(tt, nullptr, {"A", "B"}));
  EXPECT_EQ(hl_from_f(), arrival(tt, nullptr, {"A", "C"}));
  // what finalize computed covers the routes and locations added
  EXPECT_EQ(tt.n_routes(), tt.route_traffic_days_.size());
  EXPECT_EQ(tt.n_locations(), tt.locations_.location_importance_.size());
  // motis' flex routing indexes it with every location near a coordinate
  EXPECT_EQ(tt.n_locations(), tt.location_location_groups_.size());
}
