#include "gtest/gtest.h"

#include "nigiri/routing/query.h"
#include "nigiri/timetable.h"

#include "../../raptor_search.h"
#include "../../transfer_rules_util.h"

// Folding the pair defaults (the GTFS loader's majority fold) must not change
// what the rules state. The first tests run on test::network()
// (transfer_rules_util.h): F arrives S1 10:30, G leaves S2 10:40 (fallback GL
// 11:10), H leaves S1 10:31 (fallback HL 11:01).

using namespace nigiri;
using nigiri::test::arrival;
using nigiri::test::arrival_at;
using nigiri::test::at;
using nigiri::test::feed;
using nigiri::test::lidx;
using nigiri::test::load_feeds;
using nigiri::test::load_network;
using nigiri::test::n_virts;
using nigiri::test::raptor_search;
using nigiri::test::search_at;

// A trip rule that restates the default (2 min) is not redundant if a less
// specific rule states something else (route pair: 0 min). F -> H has 1 min,
// so the trip rule sends F's passengers to HL.
TEST(gtfs, transfer_rules_fold_keeps_exception_to_faster_rule) {
  auto const tt =
      load_network("S1,S1,2,120,,,,\nS1,S1,2,0,RF,RH,,\nS1,S1,2,120,,,F,H");
  EXPECT_EQ(at("11:30"), arrival(tt, nullptr, {"A", "C"}));
}

// ... same with a ban: RF may not transfer from S1 to S2, but the trip F may,
// at the default time.
TEST(gtfs, transfer_rules_fold_keeps_exception_to_ban) {
  auto const tt =
      load_network("S1,S2,2,120,,,,\nS1,S2,3,,RF,,,\nS1,S2,2,120,,,F,");
  EXPECT_EQ(at("11:00"), arrival(tt, nullptr, {"A", "B"}));
}

// Rules stated for a station: their majority (15 min) is the default for the
// pairs of its child stops as well, not only for the station itself.
TEST(gtfs, transfer_rules_fold_station_majority_reaches_base_locations) {
  auto const tt = load_network("S,S,2,900,RF,RG,,\nS,S,2,900,RF,RH,,");
  EXPECT_EQ(at("11:30"), arrival(tt, nullptr, {"A", "B"}));
}

// ... and an exception to that majority survives: F -> G is quick.
TEST(gtfs, transfer_rules_fold_station_majority_with_exception) {
  auto const tt =
      load_network("S,S,2,900,RF,RG,,\nS,S,2,900,RF,RH,,\nS,S,2,60,,,F,G");
  EXPECT_EQ(at("11:00"), arrival(tt, nullptr, {"A", "B"}));
}

// ===========================================================================
// The fold and the station hierarchy, on feeds of their own.
// ===========================================================================

// DA -> DB is named by a station level trip rule (2 min), the stop pair
// T1 -> T2 carries an RD1 -> RD2 route rule (8 min) and a plain 5 min row. The
// trip rule is the most specific one for DA -> DB: the 3 min gap is enough.
TEST(
    gtfs,
    transfer_rules_fold_keeps_station_trip_rule_over_route_rule_of_base_locations) {
  auto const tt =
      load_feeds({feed({{"T", 54.0, 12.0, "", true},
                        {"T1", 54.0, 12.0, "T"},
                        {"T2", 54.0003, 12.0, "T"},
                        {"J", 54.1, 12.0},
                        {"K", 54.2, 12.0}},
                       {{"DA", "RD1", {{"J", "16:00"}, {"T1", "16:30"}}},
                        {"DB", "RD2", {{"T2", "16:33"}, {"K", "17:00"}}},
                        {"DB2", "RD2", {{"T2", "16:45"}, {"K", "17:15"}}}},
                       "T,T,2,120,,,DA,DB\n"
                       "T1,T2,2,300,,,,\n"
                       "T1,T2,2,480,RD1,RD2,,\n")});
  EXPECT_EQ(at("17:00"), arrival_at(tt, "J", "K", "16:00"));
}

// SJ,SJ,2,600 states the default for every pair of SJ's stops. The one
// route rule RJ1 -> RJ2 on J1 -> J2 must not replace it for other routes:
// RJ1 -> RJ3 (5 min gap) still needs 10 min.
TEST(gtfs, transfer_rules_fold_keeps_explicit_station_default) {
  auto const tt =
      load_feeds({feed({{"SJ", 59.0, 17.0, "", true},
                        {"J1", 59.0, 17.0, "SJ"},
                        {"J2", 59.0003, 17.0, "SJ"},
                        {"JA", 59.1, 17.0},
                        {"JB", 59.2, 17.0}},
                       {{"JT1", "RJ1", {{"JA", "10:00"}, {"J1", "10:30"}}},
                        {"JC", "RJ3", {{"J2", "10:35"}, {"JB", "11:00"}}},
                        {"JC2", "RJ3", {{"J2", "10:45"}, {"JB", "11:15"}}}},
                       "SJ,SJ,2,600,,,,\n"
                       "J1,J2,2,180,RJ1,RJ2,,\n",
                       {"RJ2"})});
  EXPECT_EQ(at("11:15"), arrival_at(tt, "JA", "JB", "10:00"));
}

// The station states its default (LS,LS: 1 min), so the qualified rows at
// stop L1 are exceptions to it and nothing is folded: every other trip pair
// at L1 has the station's value - for a trip at a virtual location of L1 (LT5,
// by its 0 min trip rule) just like for a plain one (LT9).
TEST(gtfs, transfer_rules_fold_station_default_holds_at_virtual_locations) {
  auto const tt =
      load_feeds({feed({{"LS", 61.0, 19.0, "", true},
                        {"L1", 61.0, 19.0, "LS"},
                        {"L2", 61.0003, 19.0, "LS"},
                        {"LO", 61.1, 19.0},
                        {"LO2", 61.1, 19.3},
                        {"LD", 61.2, 19.0},
                        {"LD2", 61.2, 19.2}},
                       {{"LT5", "RL5", {{"LO", "10:00"}, {"L1", "10:30"}}},
                        {"LT6", "RL6", {{"L1", "10:40"}, {"LD", "11:00"}}},
                        {"LT7", "RL7", {{"L1", "10:32"}, {"LD2", "11:00"}}},
                        {"LT8", "RL7", {{"L1", "10:36"}, {"LD2", "11:05"}}},
                        {"LT9", "RL9", {{"LO2", "10:00"}, {"L1", "10:30"}}}},
                       "LS,LS,2,60,,,,\n"
                       "L1,L1,2,300,RL1,RL2,,\n"
                       "L1,L1,2,300,RL3,RL4,,\n"
                       "L1,L1,2,0,,,LT5,LT6\n",
                       {"RL1", "RL2", "RL3", "RL4"})});

  // A plain arrival at L1 makes LT7 (2 min later).
  EXPECT_EQ(at("11:00"), arrival_at(tt, "LO2", "LD2", "10:00"));

  // LT5 arrives at its virtual location below L1: the same.
  EXPECT_EQ(at("11:00"), arrival_at(tt, "LO", "LD2", "10:00"));
}

// One guarantee (KT1 -> KT2) next to three 5 min trip pairs at K: the 5 min
// rows are the majority and become K's transfer time. The guarantee names other
// trips, so it applies to none of their pairs: the three rows are redundant
// and only KT1/KT2 get virtual locations. KT3 -> KT4 (4 min) still misses.
TEST(gtfs,
     transfer_rules_fold_drops_default_rows_next_to_rules_for_other_trips) {
  auto const tt = load_feeds(
      {feed({{"K", 62.0, 20.0}, {"KA", 62.1, 20.0}, {"KB", 62.2, 20.0}},
            {{"KT1", "RK1", {{"KA", "08:00"}, {"K", "08:30"}}},
             {"KT2", "RK2", {{"K", "08:30"}, {"KB", "09:00"}}},
             {"KT3", "RK1", {{"KA", "09:30"}, {"K", "10:00"}}},
             {"KT4", "RK2", {{"K", "10:04"}, {"KB", "10:30"}}},
             {"KT5", "RK1", {{"KA", "10:30"}, {"K", "11:00"}}},
             {"KT6", "RK2", {{"K", "11:10"}, {"KB", "11:40"}}},
             {"KT7", "RK1", {{"KA", "11:30"}, {"K", "12:00"}}},
             {"KT8", "RK2", {{"K", "12:10"}, {"KB", "12:40"}}},
             {"KT9", "RK2", {{"K", "10:10"}, {"KB", "10:40"}}}},
            "K,K,1,,,,KT1,KT2\n"
            "K,K,2,300,,,KT3,KT4\n"
            "K,K,2,300,,,KT5,KT6\n"
            "K,K,2,300,,,KT7,KT8\n")});
  EXPECT_EQ(5, tt.locations_.transfer_time_[lidx(tt, "K")].count());
  EXPECT_EQ(2U, n_virts(tt));

  EXPECT_EQ(at("09:00"), arrival_at(tt, "KA", "KB", "08:00"));

  EXPECT_EQ(at("10:40"), arrival_at(tt, "KA", "KB", "09:30"));
}
