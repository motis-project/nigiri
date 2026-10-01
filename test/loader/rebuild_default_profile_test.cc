#include "gtest/gtest.h"

#include <optional>
#include <string_view>

#include "nigiri/timetable.h"

#include "../transfer_rules_util.h"

// Replacing the walks of the default profile (osr_footpath: a street router
// knows better than the loader's beelines) changes nothing a rule states.

using namespace nigiri;
using nigiri::test::arrival;
using nigiri::test::n_virts;
using nigiri::test::rebuild_default_profile;
using nigiri::test::t;

// test::kNetwork: F arrives S1 10:30. G leaves S2 10:40 (fallback GL 11:10)
// to B, H leaves S1 10:31 (fallback HL 11:01) to C. Plus K, which leaves T
// 10:45 to E. T is 450 m from S: the same feed, not the same station, so the
// loader connects the two with nothing.
constexpr auto const kRows =
    test::network_rows{.stops_ =
                           "T,T,,50.0045,6.5,,,\n"
                           "E,E,,50.0,8.5,,,\n",
                       .routes_ = "RK,AG,RK,,,3\n",
                       .trips_ = "RK,X,K,,\n",
                       .stop_times_ =
                           "K,10:45:00,10:45:00,T,1,0,0\n"
                           "K,11:05:00,11:05:00,E,2,0,0\n"};

// Arrivals from A via G, GL, HL and K. Functions, not constants: at static
// init time the time zone database is not loaded yet.
unixtime_t via_g() { return t("2019-05-01 11:00 Europe/Berlin"); }
unixtime_t via_gl() { return t("2019-05-01 11:30 Europe/Berlin"); }
unixtime_t via_hl() { return t("2019-05-01 11:30 Europe/Berlin"); }
unixtime_t via_k() { return t("2019-05-01 11:05 Europe/Berlin"); }

TEST(rebuild_default_profile, walk_replaces_beeline) {
  auto tt = test::load_network("", kRows);
  EXPECT_EQ(via_g(),
            arrival(tt, nullptr, {"A", "B"}));  // beeline S1 -> S2: 2 min
  rebuild_default_profile(tt, {{"S1", "S2", 12}});
  EXPECT_EQ(via_gl(), arrival(tt, nullptr, {"A", "B"}));
  rebuild_default_profile(tt, {{"S1", "S2", 9}});
  EXPECT_EQ(via_g(), arrival(tt, nullptr, {"A", "B"}));
}

TEST(rebuild_default_profile, walk_the_router_did_not_find) {
  auto tt = test::load_network("", kRows);
  rebuild_default_profile(tt, {});
  EXPECT_EQ(std::nullopt, arrival(tt, nullptr, {"A", "B"}));
  EXPECT_EQ(via_hl(),
            arrival(tt, nullptr, {"A", "C"}));  // changing at S1 is no walk
}

TEST(rebuild_default_profile, rule_beats_slower_walk) {
  auto tt = test::load_network("S1,S2,2,180,,,,", kRows);
  rebuild_default_profile(tt, {{"S1", "S2", 12}});
  EXPECT_EQ(via_g(), arrival(tt, nullptr, {"A", "B"}));
}

TEST(rebuild_default_profile, rule_beats_faster_walk) {
  auto tt = test::load_network("S1,S2,2,900,,,,", kRows);
  EXPECT_EQ(via_gl(), arrival(tt, nullptr, {"A", "B"}));
  rebuild_default_profile(tt, {{"S1", "S2", 1}});
  EXPECT_EQ(via_gl(), arrival(tt, nullptr, {"A", "B"}));
}

TEST(rebuild_default_profile, forbidden_trip_pair_stays_forbidden) {
  auto tt = test::load_network("S1,S2,2,120,,,,\nS1,S2,3,,,,F,G", kRows);
  ASSERT_NE(0U, n_virts(tt));
  EXPECT_EQ(via_gl(), arrival(tt, nullptr, {"A", "B"}));
  rebuild_default_profile(tt, {{"S1", "S2", 1}});
  EXPECT_EQ(via_gl(), arrival(tt, nullptr, {"A", "B"}));
}

// A rule for a whole station is a hub, which never overrides: the routing takes
// the minimum of hub and footpath, so the walk underneath has to go.
TEST(rebuild_default_profile, station_rule_beats_faster_walk) {
  auto tt = test::load_network("S,S,2,900,,,,", kRows);
  EXPECT_EQ(via_gl(), arrival(tt, nullptr, {"A", "B"}));
  EXPECT_EQ(via_hl(), arrival(tt, nullptr, {"A", "C"}));
  rebuild_default_profile(tt, {{"S1", "S2", 1}});
  EXPECT_EQ(via_gl(), arrival(tt, nullptr, {"A", "B"}));
  EXPECT_EQ(via_hl(), arrival(tt, nullptr, {"A", "C"}));
}

// F stops at a virtual location below S1 (the F -> H rule). It owns no walks:
// the walk hubs hand it those of S1, so they have to follow the new walks.
TEST(rebuild_default_profile, virtual_location_walks_like_its_stop) {
  auto tt = test::load_network("S1,S1,2,120,,,,\nS1,S1,2,1800,,,F,H", kRows);
  ASSERT_NE(0U, n_virts(tt));
  EXPECT_EQ(via_hl(), arrival(tt, nullptr, {"A", "C"}));  // the rule: 30 min
  EXPECT_EQ(via_g(), arrival(tt, nullptr, {"A", "B"}));  // beeline
  rebuild_default_profile(tt, {{"S1", "S2", 12}});
  EXPECT_EQ(via_gl(), arrival(tt, nullptr, {"A", "B"}));
  EXPECT_EQ(via_hl(), arrival(tt, nullptr, {"A", "C"}));
  rebuild_default_profile(tt, {{"S1", "S2", 3}, {"S1", "T", 6}});
  EXPECT_EQ(via_g(), arrival(tt, nullptr, {"A", "B"}));
  EXPECT_EQ(via_k(), arrival(tt, nullptr, {"A", "E"}));
}

// A second rebuild changes nothing: rule and stop hubs are built again from
// the rules in the timetable, with the same result.
TEST(rebuild_default_profile, is_repeatable) {
  auto tt = test::load_network("S,S,2,120,,,,\nS,S,2,1800,,,F,H", kRows);
  ASSERT_NE(0U, n_virts(tt));
  ASSERT_NE(0U, tt.locations_.hub_in_[kDefaultProfile].size());

  rebuild_default_profile(tt, {{"S1", "T", 6}});
  auto const n_hubs = tt.locations_.hub_in_[kDefaultProfile].size();
  auto const n_fps = tt.locations_.footpaths_out_[kDefaultProfile].data_.size();
  rebuild_default_profile(tt, {{"S1", "T", 6}});

  EXPECT_EQ(n_hubs, tt.locations_.hub_in_[kDefaultProfile].size());
  EXPECT_EQ(n_fps, tt.locations_.footpaths_out_[kDefaultProfile].data_.size());
  EXPECT_EQ(via_g(),
            arrival(tt, nullptr, {"A", "B"}));  // the station rule: 2 min
  EXPECT_EQ(via_hl(), arrival(tt, nullptr, {"A", "C"}));
  EXPECT_EQ(via_k(), arrival(tt, nullptr, {"A", "E"}));
}
