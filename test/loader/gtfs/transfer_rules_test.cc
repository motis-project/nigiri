#include "gtest/gtest.h"

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "fmt/format.h"

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/loader/register.h"
#include "nigiri/loader/transfer_rules.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/transfers.h"
#include "nigiri/timetable.h"

#include "../../raptor_search.h"
#include "../../transfer_rules_util.h"

// What transfers.txt rules mean, through the GTFS loader. Every test states
// its feed (test::feed) on 2019-05-01, Europe/Berlin; without a rule, a change
// at a stop takes 2 min.

using namespace nigiri;
using namespace std::string_view_literals;
using nigiri::test::add_empty_profile;
using nigiri::test::arrival_at;
using nigiri::test::at;
using nigiri::test::feed;
using nigiri::test::lidx;
using nigiri::test::load_feeds;
using nigiri::test::n_transit_legs;
using nigiri::test::n_virts;
using nigiri::test::raptor_search;
using nigiri::test::search_at;
using nigiri::test::t;

// A station-level rule cascades to the station's stops: XS,XS type=2 600s.
// Every change within XS takes 10 min, X1 -> X2 as well as a change at X1
// itself: the 10:35 departures are out of reach, the 10:42 one is not.
//     T1: A 10:00 -> X1 10:30
//     T2: X2 10:35 -> B 11:00 (5 min gap < 10 -> not reachable)
//     T3: X2 10:42 -> B 11:10 (12 min gap -> ok)
//     T4: X1 10:35 -> B 11:05 (5 min at X1 itself < 10 -> not reachable)
TEST(gtfs, transfer_rules_station_level_min_time) {
  auto const tt =
      load_feeds({feed({{"XS", 50.0, 6.5, "", true},
                        {"X1", 50.0001, 6.5, "XS"},
                        {"X2", 50.0002, 6.5, "XS"},
                        {"A", 50.0, 6.0},
                        {"B", 50.0, 7.0}},
                       {{"T1", "R1", {{"A", "10:00"}, {"X1", "10:30"}}},
                        {"T2", "R2", {{"X2", "10:35"}, {"B", "11:00"}}},
                        {"T3", "R2", {{"X2", "10:42"}, {"B", "11:10"}}},
                        {"T4", "R2", {{"X1", "10:35"}, {"B", "11:05"}}}},
                       "XS,XS,2,600,,,,\n")});
  EXPECT_EQ(at("11:10"), arrival_at(tt, "A", "B", "10:00"));
}

// Directed forbidden transfer: F1,F2 type=3, two stops of the station FS
// within walking distance.
//     V1 (R20): A3 14:00 -> F1 14:30
//     V2 (R21): F2 14:40 -> E 15:00 (would be reachable by foot -> banned)
//     V2L (R21): F2 23:30 -> E 23:50 (late enough that a banned pair stored
//                as a very long footpath would become usable - it must not be)
TEST(gtfs, transfer_rules_forbidden) {
  auto const tt =
      load_feeds({feed({{"FS", 52.0, 6.5, "", true},
                        {"F1", 52.0001, 6.5, "FS"},
                        {"F2", 52.0002, 6.5, "FS"},
                        {"A3", 52.0, 6.0},
                        {"E", 52.0, 7.0}},
                       {{"V1", "R20", {{"A3", "14:00"}, {"F1", "14:30"}}},
                        {"V2", "R21", {{"F2", "14:40"}, {"E", "15:00"}}},
                        {"V2L", "R21", {{"F2", "23:30"}, {"E", "23:50"}}}},
                       "F1,F2,3,,,,,\n")});

  EXPECT_EQ(0U, search_at(tt, "A3", "E", "14:00").size());

  // A ban is not a very long transfer. Stored as one, the pair stays walkable
  // and any departure far enough out turns it into a journey: V2L leaves F2 at
  // 23:30, nine hours after V1 gets in, so a banned pair written at
  // footpath::kBan would be reachable here.
  auto const late =
      raptor_search(tt, nullptr, "A3", "E",
                    interval{at("14:00"), t("2019-05-02 02:00 Europe/Berlin")});
  EXPECT_EQ(0U, late.size());

  // Nothing may connect the two stops in the footpath layer either.
  auto const f2 = lidx(tt, "F2");
  for (auto const fp :
       tt.locations_.footpaths_out_[kDefaultProfile][lidx(tt, "F1")]) {
    EXPECT_NE(fp.target(), f2) << "the banned pair is still a footpath";
  }
}

// A trip-qualified rule beats a route-qualified one: Z,Z type=2 0s
// from_route=RZ1 to_route=RZ2 (allows tight changes), Z,Z type=3
// from_trip=W1 to_trip=W2 (bans this one pair), plus the unqualified Z,Z
// type=2 120s row (pair default).
//     W1 (RZ1): A4 16:00 -> Z 16:30
//     W2 (RZ2): Z 16:31 -> BZ 17:00 (banned by the trip rule)
//     W3 (RZ2): Z 16:31 -> CZ 17:10 (allowed by the route rule)
TEST(gtfs, transfer_rules_trip_beats_route) {
  auto const tt =
      load_feeds({feed({{"A4", 53.0, 6.0},
                        {"Z", 53.0, 6.5},
                        {"BZ", 53.0, 7.0},
                        {"CZ", 53.0, 7.5}},
                       {{"W1", "RZ1", {{"A4", "16:00"}, {"Z", "16:30"}}},
                        {"W2", "RZ2", {{"Z", "16:31"}, {"BZ", "17:00"}}},
                        {"W3", "RZ2", {{"Z", "16:31"}, {"CZ", "17:10"}}}},
                       "Z,Z,2,120,,,,\n"
                       "Z,Z,2,0,RZ1,RZ2,,\n"
                       "Z,Z,3,,,,W1,W2\n")});
  ASSERT_LT(0U, n_virts(tt));

  EXPECT_EQ(0U, search_at(tt, "A4", "BZ", "16:00").size());

  EXPECT_EQ(at("17:10"), arrival_at(tt, "A4", "CZ", "16:00"));
}

// Guaranteed connection (MetroNorth pattern): M,M type=1 for the trip pair
// M1 -> M2. The vehicle waits, so that pair costs 0 min, whatever
// min_transfer_time states (5 min) - but a guarantee says nothing about how
// long a change at M takes, so it does not become the stop's default.
//     M1 (R30): A5 18:00 -> M 18:30
//     M2 (R31): M 18:30 -> G 19:00 (0 min -> reachable)
//     M3 (R32): M 18:30 -> H 19:00 (unnamed, default 2 min -> not reachable)
TEST(gtfs, transfer_rules_guaranteed_connection) {
  auto const tt = load_feeds({feed(
      {{"A5", 54.0, 6.0}, {"M", 54.0, 6.5}, {"G", 54.0, 7.0}, {"H", 54.0, 7.5}},
      {{"M1", "R30", {{"A5", "18:00"}, {"M", "18:30"}}},
       {"M2", "R31", {{"M", "18:30"}, {"G", "19:00"}}},
       {"M3", "R32", {{"M", "18:30"}, {"H", "19:00"}}}},
      "M,M,1,300,,,M1,M2\n")});
  ASSERT_LT(0U, n_virts(tt));

  EXPECT_EQ(duration_t{2}, tt.locations_.transfer_time_[lidx(tt, "M")]);

  EXPECT_EQ(at("19:00"), arrival_at(tt, "A5", "G", "18:00"));

  EXPECT_EQ(0U, search_at(tt, "A5", "H", "18:00").size());
}

// Self-pair rule (both sides match the same trips): P,P type=2 600s
// from_route=RP to_route=RP. All RP trips share one virtual location, and the
// value becomes its transfer time; the explicit P,P type=2 120s row keeps the
// rule an exception (majority fold).
//     P1 (RP): A8 22:00 -> P 22:30
//     P2 (RP): P 22:35 -> K 23:00 (5 min < 10 -> not reachable)
//     P3 (RQ): P 22:35 -> L 23:00 (unnamed, pair default 2 min -> ok)
TEST(gtfs, transfer_rules_self_pair) {
  auto const tt = load_feeds({feed(
      {{"A8", 56.0, 6.0}, {"P", 56.0, 6.5}, {"K", 56.0, 7.0}, {"L", 56.0, 7.5}},
      {{"P1", "RP", {{"A8", "22:00"}, {"P", "22:30"}}},
       {"P2", "RP", {{"P", "22:35"}, {"K", "23:00"}}},
       {"P3", "RQ", {{"P", "22:35"}, {"L", "23:00"}}}},
      "P,P,2,120,,,,\n"
      "P,P,2,600,RP,RP,,\n")});
  ASSERT_LT(0U, n_virts(tt));

  EXPECT_EQ(0U, search_at(tt, "A8", "K", "22:00").size());

  EXPECT_EQ(at("23:00"), arrival_at(tt, "A8", "L", "22:00"));
}

// Trip pair rule: Q,Q type=2 600s for Q1 -> Q2, plus the unqualified Q,Q
// type=2 120s row.
//     Q1 (RQ1): A9 08:00 -> Q 08:30
//     Q2 (RQ2): Q 08:32 -> QA 09:00 (2 min gap < 10 -> not reachable)
//     Q2L (RQ2): Q 08:41 -> QA 09:10 (11 min gap -> ok)
//     Q3 (RQ3): Q 08:32 -> QB 09:00 (unnamed, pair default 2 min -> ok)
TEST(gtfs, transfer_rules_trip_pair) {
  auto const tt =
      load_feeds({feed({{"A9", 57.0, 6.0},
                        {"Q", 57.0, 6.5},
                        {"QA", 57.0, 7.0},
                        {"QB", 57.0, 7.5}},
                       {{"Q1", "RQ1", {{"A9", "08:00"}, {"Q", "08:30"}}},
                        {"Q2", "RQ2", {{"Q", "08:32"}, {"QA", "09:00"}}},
                        {"Q2L", "RQ2", {{"Q", "08:41"}, {"QA", "09:10"}}},
                        {"Q3", "RQ3", {{"Q", "08:32"}, {"QB", "09:00"}}}},
                       "Q,Q,2,120,,,,\n"
                       "Q,Q,2,600,,,Q1,Q2\n")});
  ASSERT_LT(0U, n_virts(tt));

  EXPECT_EQ(at("09:10"), arrival_at(tt, "A9", "QA", "08:00"));

  // The unnamed trip still departs from the stop Q itself at the default.
  EXPECT_EQ(at("09:00"), arrival_at(tt, "A9", "QB", "08:00"));
}

// A rule that lets one route change without waiting must not become usable
// for another route at the same stop. GA and GB state the same slow rule, GC
// the fast one: GA and GB may share a virtual location, but it must not get
// the fast rule of GC, which would make GB's impossible connection (GB and GC
// both arrive 1 min before GX leaves) possible.
TEST(gtfs, transfer_rules_fast_rule_stays_on_its_route) {
  auto const tt =
      load_feeds({feed({{"GO1", 59.0, 6.0},
                        {"GO2", 59.0, 6.1},
                        {"GO3", 59.0, 6.2},
                        {"GU", 59.0, 6.5},
                        {"GUD", 59.0, 7.0}},
                       {{"GA", "RGA", {{"GO1", "08:00"}, {"GU", "09:00"}}},
                        {"GB", "RGB", {{"GO2", "09:30"}, {"GU", "10:00"}}},
                        {"GC", "RGC", {{"GO3", "09:40"}, {"GU", "10:00"}}},
                        {"GX", "RGX", {{"GU", "10:01"}, {"GUD", "10:30"}}}},
                       "GU,GU,2,180,RGA,RGX,,\n"
                       "GU,GU,2,180,RGB,RGX,,\n"
                       "GU,GU,2,0,RGC,RGX,,\n")});
  ASSERT_LT(0U, n_virts(tt));

  // GC is the route the 0 min rule names: 10:00 + 0 catches the 10:01.
  EXPECT_EQ(at("10:30"), arrival_at(tt, "GO3", "GUD", "09:30"));

  // GB arrives 10:00 and owes 3 min, so 10:01 is out of reach - it may not
  // borrow the 0 min rule from GC.
  EXPECT_EQ(0U, search_at(tt, "GO2", "GUD", "09:00").size());

  // GA states the same 3 min as GB and has the slack for it.
  EXPECT_EQ(at("10:30"), arrival_at(tt, "GO1", "GUD", "07:30"));
}

// Unqualified same-stop ban: T6N,T6N type=3 says no change is possible at
// T6N, for anyone. It becomes the stop's own transfer time, and a ban is not a
// very long time: neither the 15 min nor the 3 h connection may exist.
// Getting off at T6N as the destination is still fine. No rule is qualified,
// so there is no virtual location, and every profile has the stop's transfer
// time: a profile that ignores transfers.txt finds no connection either.
//     T6X1 (R76): T6A 14:30 -> T6N 15:00
//     T6Y1 (R77): T6N 15:15 -> T6J 15:30
//     T6Y2 (R77): T6N 18:00 -> T6J 18:20
TEST(gtfs, transfer_rules_forbidden_same_stop_unqualified) {
  constexpr auto const kProfile = profile_idx_t{1U};
  auto tt = load_feeds(
      {feed({{"T6A", 60.0, 6.0}, {"T6N", 60.0, 6.5}, {"T6J", 60.0, 7.0}},
            {{"T6X1", "R76", {{"T6A", "14:30"}, {"T6N", "15:00"}}},
             {"T6Y1", "R77", {{"T6N", "15:15"}, {"T6J", "15:30"}}},
             {"T6Y2", "R77", {{"T6N", "18:00"}, {"T6J", "18:20"}}}},
            "T6N,T6N,3,,,,,\n")});
  add_empty_profile(tt, kProfile);

  EXPECT_EQ(1U, search_at(tt, "T6A", "T6N", "14:30").size());
  for (auto const prf : {kDefaultProfile, kProfile}) {
    SCOPED_TRACE(prf);
    EXPECT_EQ(0U, raptor_search(
                      tt, nullptr,
                      routing::query{
                          .start_time_ = interval{at("14:30"), at("22:00")},
                          .start_ = {{lidx(tt, "T6A"), 0_minutes, 0U}},
                          .destination_ = {{lidx(tt, "T6J"), 0_minutes, 0U}},
                          .prf_idx_ = prf},
                      direction::kForward)
                      .size());
  }
}

// Route-qualified same-stop ban: T7Q,T7Q type=3 from_route=R78 to_route=R78
// bans changing between trips of R78 at T7Q; the unqualified T7Q,T7Q 120s row
// is the default for everyone else. The ban becomes the transfer time of the
// virtual location R78 gets at T7Q. Stored as a duration, it would wrap in the
// 8 bit field and a departure 260 min later would be reachable - it must not
// be.
//     T7X1 (R78): T7A 14:30 -> T7Q 15:00
//     T7X2 (R78): T7Q 15:15 -> T7K 15:30 (banned)
//     T7X3 (R78): T7Q 19:20 -> T7K 19:40 (banned, 260 min later)
//     T7Y1 (R79): T7Q 15:15 -> T7L 15:30 (default 2 min -> reachable)
TEST(gtfs, transfer_rules_forbidden_same_stop_route) {
  auto const tt =
      load_feeds({feed({{"T7A", 61.0, 6.0},
                        {"T7Q", 61.0, 6.5},
                        {"T7K", 61.0, 7.0},
                        {"T7L", 61.0, 7.5}},
                       {{"T7X1", "R78", {{"T7A", "14:30"}, {"T7Q", "15:00"}}},
                        {"T7X2", "R78", {{"T7Q", "15:15"}, {"T7K", "15:30"}}},
                        {"T7X3", "R78", {{"T7Q", "19:20"}, {"T7K", "19:40"}}},
                        {"T7Y1", "R79", {{"T7Q", "15:15"}, {"T7L", "15:30"}}}},
                       "T7Q,T7Q,2,120,,,,\n"
                       "T7Q,T7Q,3,,R78,R78,,\n")});
  ASSERT_LT(0U, n_virts(tt));

  EXPECT_EQ(at("15:30"), arrival_at(tt, "T7A", "T7L", "14:30"));

  EXPECT_EQ(0U, raptor_search(tt, nullptr, "T7A", "T7K",
                              interval{at("14:30"), at("22:00")})
                    .size());
}

// Unqualified same-stop ban with a qualified exception: T8Q,T8Q type=3 bans
// every change at T8Q, T8Q,T8Q type=2 120s from_route=R80 to_route=R81
// allows R80 -> R81. R80 gets a virtual location at T8Q, which inherits the
// ban as its own transfer time - as the ban, not as a transfer time of 255
// minutes that a long enough wait satisfies. The ban forbids changing
// vehicles at T8Q, not leaving it: the beeline walk to T8W in the second feed
// keeps its normal duration (link_nearby_stations links stops of different
// feeds only, T8W is 55 m from T8Q).
//     T8X1 (R80): T8A 09:30 -> T8Q 10:00
//     T8Y1 (R81): T8Q 10:05 -> T8K 10:25 (exception, 2 min -> reachable)
//     T8X2 (R80): T8Q 15:00 -> T8L 15:20 (same virtual location, banned)
//     T8Z1 (R82, second feed): T8W 10:15 -> T8V 10:35, walk to T8M 10:37
timetable load_banned_stop_with_exception() {
  return load_feeds(
      {feed({{"T8A", 62.0, 6.0},
             {"T8Q", 62.0, 6.5},
             {"T8K", 62.0, 7.0},
             {"T8L", 62.0, 7.2},
             {"T8M", 62.0005, 7.5}},
            {{"T8X1", "R80", {{"T8A", "09:30"}, {"T8Q", "10:00"}}},
             {"T8X2", "R80", {{"T8Q", "15:00"}, {"T8L", "15:20"}}},
             {"T8Y1", "R81", {{"T8Q", "10:05"}, {"T8K", "10:25"}}}},
            "T8Q,T8Q,3,,,,,\n"
            "T8Q,T8Q,2,120,R80,R81,,\n"),
       feed({{"T8W", 62.0005, 6.5}, {"T8V", 62.0, 7.5}},
            {{"T8Z1", "R82", {{"T8W", "10:15"}, {"T8V", "10:35"}}}}, "")});
}

TEST(gtfs, transfer_rules_forbidden_same_stop_with_exception) {
  auto const tt = load_banned_stop_with_exception();
  ASSERT_LT(0U, n_virts(tt));

  EXPECT_EQ(at("10:25"), arrival_at(tt, "T8A", "T8K", "09:30"));

  EXPECT_EQ(0U, raptor_search(tt, nullptr, "T8A", "T8L",
                              interval{at("09:30"), at("16:00")})
                    .size());

  EXPECT_EQ(at("10:37"), arrival_at(tt, "T8A", "T8M", "09:30"));
}

// The same journey searched backwards (arrive by 12:00). The walk into T8M
// ends the journey: it starts when the ride before it ends, at 10:35, and not
// as late as the search start allows.
TEST(gtfs, transfer_rules_walk_at_end_of_backward_search) {
  auto const tt = load_banned_stop_with_exception();

  auto q = routing::query{};
  q.use_start_footpaths_ = true;  // The search starts at T8M, which has no
                                  // trips: it has to walk out of it.
  auto const res =
      raptor_search(tt, nullptr, std::move(q), "T8M", "T8A",
                    "2019-05-01 12:00 Europe/Berlin", direction::kBackward);
  ASSERT_EQ(1U, res.size());
  auto const& j = *begin(res);
  ASSERT_EQ(4U, j.legs_.size());
  EXPECT_EQ(at("09:30"), j.legs_.front().dep_time_);
  EXPECT_EQ(at("10:35"), j.legs_.back().dep_time_);
  EXPECT_EQ(at("10:37"), j.legs_.back().arr_time_);
}

// Forbidden cross product: T9S,T9T type=3 from_route=R90 to_route=R91 bans
// R90 -> R91 between the two stations (walking distance). R90 arrives at two
// stops, R91 departs from three, so the six banned pairs are a rectangle a
// one footpath per pair would cover - but a ban has no weight: written as a
// footpath of the maximum duration, the pairs stay walkable and become
// reachable after 511 minutes.
//     T9X1 (R90): T9A 09:00 -> T9S1 09:30, T9X2 (R90): T9A 09:05 -> T9S2 09:35
//     T9Y1..3 (R91): T9T1 09:40 / T9T2 09:45 / T9T3 09:50 -> T9B (banned)
//     T9Y4 (R91): T9T1 19:00 -> T9B 19:20 (banned, 570 min later)
//     T9Z1 (R92): T9T1 09:40 -> T9C 10:00 (no rule -> walk, reachable)
TEST(gtfs, transfer_rules_forbidden_cross_product) {
  auto const tt =
      load_feeds({feed({{"T9S", 63.0, 6.5, "", true},
                        {"T9S1", 63.0001, 6.5, "T9S"},
                        {"T9S2", 63.0002, 6.5, "T9S"},
                        {"T9T", 63.0006, 6.5, "", true},
                        {"T9T1", 63.0003, 6.5, "T9T"},
                        {"T9T2", 63.0004, 6.5, "T9T"},
                        {"T9T3", 63.0005, 6.5, "T9T"},
                        {"T9A", 63.0, 6.0},
                        {"T9B", 63.0, 7.0},
                        {"T9C", 63.0, 7.5}},
                       {{"T9X1", "R90", {{"T9A", "09:00"}, {"T9S1", "09:30"}}},
                        {"T9X2", "R90", {{"T9A", "09:05"}, {"T9S2", "09:35"}}},
                        {"T9Y1", "R91", {{"T9T1", "09:40"}, {"T9B", "10:00"}}},
                        {"T9Y2", "R91", {{"T9T2", "09:45"}, {"T9B", "10:05"}}},
                        {"T9Y3", "R91", {{"T9T3", "09:50"}, {"T9B", "10:10"}}},
                        {"T9Y4", "R91", {{"T9T1", "19:00"}, {"T9B", "19:20"}}},
                        {"T9Z1", "R92", {{"T9T1", "09:40"}, {"T9C", "10:00"}}}},
                       "T9S,T9T,3,,R90,R91,,\n")});
  ASSERT_LT(0U, n_virts(tt));

  EXPECT_EQ(0U, raptor_search(tt, nullptr, "T9A", "T9B",
                              interval{at("09:00"), at("20:00")})
                    .size());

  EXPECT_EQ(at("10:00"), arrival_at(tt, "T9A", "T9C", "09:00"));
}

// A guaranteed arrival shadowed by an earlier one: T10P1 reaches T10S first
// (10:00, no guarantee), T10P2 a minute later with a guarantee onto T10P3
// (10:01). Only the later arrival may board. A search that kept one arrival
// per stop and checked the guarantee when boarding would hold T10P1's label
// and lose the connection; the virtual location of the guaranteed pair keeps
// it.
//     T10P1 (R98): T10O 09:30 -> T10S 10:00
//     T10P2 (R99): T10O 09:29 -> T10S 10:01 (guaranteed onto T10P3)
//     T10P3 (R100): T10S 10:01 -> T10K 10:30
TEST(gtfs, transfer_rules_guarantee_not_shadowed) {
  auto const tt = load_feeds(
      {feed({{"T10O", 65.0, 6.0}, {"T10S", 65.0, 6.5}, {"T10K", 65.0, 7.0}},
            {{"T10P1", "R98", {{"T10O", "09:30"}, {"T10S", "10:00"}}},
             {"T10P2", "R99", {{"T10O", "09:29"}, {"T10S", "10:01"}}},
             {"T10P3", "R100", {{"T10S", "10:01"}, {"T10K", "10:30"}}}},
            "T10S,T10S,1,,,,T10P2,T10P3\n")});
  ASSERT_LT(0U, n_virts(tt));

  auto const res = raptor_search(tt, nullptr, "T10O", "T10K",
                                 interval{at("09:29"), at("09:31")});
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("09:29"), begin(res)->start_time_);
  EXPECT_EQ(at("10:30"), begin(res)->dest_time_);
}

// Trip a (route BR) ends at BS and the vehicle goes on as b (route BQ). At
// BS, the rule for leaving route BR (5 min) forbids the change to c, which
// leaves 3 min after a arrives. The vehicle still runs through BS: b leaves
// when a arrives, too early for a change.
TEST(gtfs, transfer_rules_block_runs_through_rule_stop) {
  auto const tt =
      load_feeds({feed({{"BA", 50.0, 6.0},
                        {"BS", 50.0, 6.5},
                        {"BB", 50.0, 7.0},
                        {"BX", 50.0, 8.0}},
                       {{"a", "BR", {{"BA", "10:00"}, {"BS", "10:30"}}, "blk"},
                        {"b", "BQ", {{"BS", "10:30"}, {"BB", "11:00"}}, "blk"},
                        {"c", "BC", {{"BS", "10:33"}, {"BX", "11:00"}}}},
                       "BS,BS,2,120,,,,\n"
                       "BS,BS,2,300,BR,,,\n")});
  ASSERT_LT(0U, n_virts(tt));
  ASSERT_EQ(std::nullopt, arrival_at(tt, "BA", "BX", "09:50"));

  auto const res = search_at(tt, "BA", "BB", "09:50");
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(0U, begin(res)->transfers_);
  EXPECT_EQ(at("11:00"), begin(res)->dest_time_);
}

// A trip at a location of its own must not get walks of its own into other
// feeds: it leaves through its stop. On test::network(), S1 takes 5 min to
// change, the RF trips among themselves 0 min, and F3 (RF) leaves 3 min after
// F arrives. The other feed's TZ1 leaves Z (50 m away) 4 min after F arrives:
// out of reach, like from S1 itself - TZ2 is the next.
TEST(gtfs, transfer_rules_virtual_location_not_linked_to_other_feeds) {
  auto const tt = load_feeds(
      {test::network(
           "S1,S1,2,300,,,,\nS1,S1,2,0,RF,RF,,",
           {.stops_ = {{"D3", 50.0, 9.0}},
            .trips_ = {{"F3", "RF", {{"S1", "10:33"}, {"D3", "11:00"}}}}}),
       feed({{"Z", 50.0005, 6.5}, {"Y", 50.5, 6.5}},
            {{"TZ1", "RZ", {{"Z", "10:34"}, {"Y", "11:00"}}},
             {"TZ2", "RZ", {{"Z", "10:40"}, {"Y", "11:10"}}}},
            "")});
  ASSERT_LT(0U, n_virts(tt));
  ASSERT_EQ(at("11:00"), arrival_at(tt, "A", "D3", "10:00"));
  EXPECT_EQ(
      at("11:10"),
      test::arrival(raptor_search(
          tt, nullptr,
          routing::query{.start_time_ = at("10:00"),
                         .start_ = {{lidx(tt, "A"), 0_minutes, 0U}},
                         .destination_ = {{lidx(tt, "Y", source_idx_t{1U}),
                                           0_minutes, 0U}}})));
}

// ===========================================================================
// A virtual location's own transfer time: a rule qualified on one side only
// ("arriving on RB, whatever departs") also covers two RB trips that share
// the virtual location.
// ===========================================================================

TEST(gtfs, transfer_rules_one_sided_rule_applies_between_trips_of_its_route) {
  auto const tt =
      load_feeds({feed({{"Z", 52.0, 10.0},
                        {"E", 52.1, 10.0},
                        {"F", 52.2, 10.0},
                        {"E2", 52.1, 10.2},
                        {"F2", 52.2, 10.2}},
                       {{"TB1", "RB", {{"E", "12:00"}, {"Z", "12:30"}}},
                        {"TB2", "RB", {{"Z", "12:35"}, {"F", "13:00"}}},
                        {"TB3", "RB", {{"Z", "12:45"}, {"F", "13:15"}}},
                        {"TB4", "RB", {{"E2", "12:00"}, {"Z", "12:30"}}},
                        {"TB5", "RB2", {{"Z", "12:35"}, {"F2", "13:00"}}},
                        {"TB6", "RB2", {{"Z", "12:45"}, {"F2", "13:15"}}}},
                       "Z,Z,2,120,,,,\n"
                       "Z,Z,2,600,RB,,,\n")});
  ASSERT_LT(0U, n_virts(tt));

  // Control: RB -> RB2 respects the 10 min.
  EXPECT_EQ(at("13:15"), arrival_at(tt, "E2", "F2", "12:00"));

  // RB -> RB: the same rule, so TB2 (5 min) cannot be reached either.
  EXPECT_EQ(at("13:15"), arrival_at(tt, "E", "F", "12:00"));
}

// ===========================================================================
// A shared location has to keep the rules' specificity: CA is named by a trip
// rule (5 min), CA2 only by the route rules. For CA2 -> CC2 the most specific
// rule is RC -> RC2 (10 min), not CA's trip rule.
// ===========================================================================

// CA and CA2 (route RC) arrive at S, the RC2 trips leave S2: CC 6 min after
// CA, CC2 6 min after CA2, CC3 15 min after CA2. CA is named by a trip rule,
// RC by a route rule (both 5 min), RC -> RC2 by a route pair rule (10 min).
TEST(gtfs, transfer_rules_shared_location_keeps_specificity) {
  auto const tt =
      load_feeds({feed({{"S", 53.0, 11.0},
                        {"S2", 53.0005, 11.0},
                        {"G", 53.1, 11.0},
                        {"H", 53.2, 11.0}},
                       {{"CA", "RC", {{"G", "13:00"}, {"S", "13:30"}}},
                        {"CA2", "RC", {{"G", "14:00"}, {"S", "14:30"}}},
                        {"CC", "RC2", {{"S2", "13:36"}, {"H", "14:00"}}},
                        {"CC2", "RC2", {{"S2", "14:36"}, {"H", "15:00"}}},
                        {"CC3", "RC2", {{"S2", "14:45"}, {"H", "15:10"}}}},
                       "S,S2,2,180,,,,\n"
                       "S,S2,2,300,,,CA,\n"
                       "S,S2,2,300,RC,,,\n"
                       "S,S2,2,600,RC,RC2,,\n")});
  ASSERT_LT(0U, n_virts(tt));

  // Control: CA -> CC, the trip rule (one trip beats both routes) gives 5 min.
  EXPECT_EQ(at("14:00"), arrival_at(tt, "G", "H", "13:00"));

  // CA2 -> CC2 (6 min) needs the route pair's 10 min: CC3.
  EXPECT_EQ(at("15:10"), arrival_at(tt, "G", "H", "14:00"));
}

// The specificity also has to keep the order against rules outside the
// signature. e (forbidden, any trip at KS -> trip KT, spec 12) sits
// between a (trip KTA -> route RY, 5 min, spec 16) and c (route RK -> route
// RY, 5 min, spec 8); three 7 min rows make 7 min the default, so a and c are
// kept. a beats e, e beats c: KTA and KTR must not share a virtual location.
TEST(gtfs, transfer_rules_keeps_order_against_stop_rules) {
  auto const tt =
      load_feeds({feed({{"KS", 54.0, 11.0},
                        {"KY", 54.0005, 11.0},
                        {"KA", 54.1, 11.0},
                        {"KB", 54.1, 11.1},
                        {"KD", 54.2, 11.0}},
                       {{"KTA", "RA", {{"KA", "10:00"}, {"KS", "10:30"}}},
                        {"KTR", "RK", {{"KB", "10:00"}, {"KS", "10:30"}}},
                        {"KT", "RY", {{"KY", "10:40"}, {"KD", "11:00"}}},
                        {"KT2", "RY", {{"KY", "11:00"}, {"KD", "11:20"}}}},
                       "KS,KY,2,300,,RY,KTA,\n"
                       "KS,KY,2,300,RK,RY,,\n"
                       "KS,KY,3,,,,,KT\n"
                       "KS,KY,2,420,RO1,RY,,\n"
                       "KS,KY,2,420,RO2,RY,,\n"
                       "KS,KY,2,420,RO3,RY,,\n",
                       {"RO1", "RO2", "RO3"})});
  ASSERT_LT(0U, n_virts(tt));

  // KTA -> KT: a (trip -> route) beats the ban e (stop -> trip).
  EXPECT_EQ(at("11:00"), arrival_at(tt, "KA", "KD", "10:00"));

  // KTR -> KT: e beats c (route -> route), KTR has to wait for KT2.
  EXPECT_EQ(at("11:20"), arrival_at(tt, "KB", "KD", "10:00"));
}

// ===========================================================================
// Block through-services and stay-seated chains.
// ===========================================================================

// ST1 -> ST2 is a stay-seated chain (type 4) at SS. The rule SX -> ST2 needs
// 10 min, SX arrives 4 min before ST2 leaves: a passenger of SX cannot make
// ST2. The chain must not drop the rule of ST2's first stop.
TEST(gtfs, transfer_rules_stay_seated_chain_keeps_rule_of_second_trip) {
  auto const tt =
      load_feeds({feed({{"SS", 58.3, 16.3},
                        {"SA", 58.4, 16.3},
                        {"SB", 58.5, 16.3},
                        {"SX0", 58.4, 16.6}},
                       {{"ST1", "RS1", {{"SA", "10:00"}, {"SS", "10:30"}}},
                        {"ST2", "RS2", {{"SS", "10:30"}, {"SB", "11:00"}}},
                        {"SX", "RSX", {{"SX0", "09:00"}, {"SS", "10:26"}}}},
                       "SS,SS,4,,,,ST1,ST2\n"
                       "SS,SS,2,120,,,,\n"
                       "SS,SS,2,600,,,SX,ST2\n")});
  ASSERT_LT(0U, n_virts(tt));

  // Control: the chain itself works.
  EXPECT_EQ(at("11:00"), arrival_at(tt, "SA", "SB", "10:00"));

  EXPECT_EQ(0U, search_at(tt, "SX0", "SB", "09:00").size());
}

// As in feed ca-qc_STTR (stop 596): JT1 (route RJA) ends at JS and its
// vehicle goes on as JT2 (route RJB, same block). RJA -> RJD is a 0 s change,
// RJB -> RJD a 60 s one (both route pairs, the stop's default is 120 s). A
// passenger who gets off JT1 at 06:42 makes JT3 (RJD, 06:42) - the rule of
// the trip they leave is RJA -> RJD. The handover stop's virtual location must
// not let RJB's rule decide: nobody gets off JT2 at its first stop.
TEST(gtfs, transfer_rules_handover_stop_rule_of_arriving_trip_decides) {
  auto const make = [](std::string_view const block) {
    return feed({{"JS", 58.6, 16.6},
                 {"JA", 58.7, 16.6},
                 {"JB", 58.8, 16.6},
                 {"JC", 58.9, 16.6}},
                {{"JT1", "RJA", {{"JA", "06:17"}, {"JS", "06:42"}}, block},
                 {"JT2", "RJB", {{"JS", "06:45"}, {"JC", "07:09"}}, block},
                 {"JT3", "RJD", {{"JS", "06:42"}, {"JB", "07:10"}}},
                 {"JT4", "RJD", {{"JS", "07:00"}, {"JB", "07:28"}}}},
                "JS,JS,2,120,RX1,RY1,,\n"
                "JS,JS,2,120,RX2,RY2,,\n"
                "JS,JS,2,120,RX3,RY3,,\n"
                "JS,JS,2,0,RJA,RJD,,\n"
                "JS,JS,2,60,RJB,RJD,,\n",
                {"RX1", "RY1", "RX2", "RY2", "RX3", "RY3"});
  };

  // Control: without the block, JT1's stop only carries RJA's rules.
  auto const plain = load_feeds({make("")});
  EXPECT_EQ(at("07:10"), arrival_at(plain, "JA", "JB", "06:17"));

  auto const joined = load_feeds({make("BLJ")});
  ASSERT_LT(0U, n_virts(joined));
  EXPECT_EQ(at("07:10"), arrival_at(joined, "JA", "JB", "06:17"));

  // The vehicle still runs through JS.
  auto const through = search_at(joined, "JA", "JC", "06:17");
  ASSERT_EQ(1U, through.size());
  EXPECT_EQ(at("07:09"), begin(through)->dest_time_);
  EXPECT_EQ(1U, n_transit_legs(*begin(through)));
}

// ===========================================================================
// Changes across days. Times are UTC (agency_timezone UTC): a trip runs on
// 2019-05-01 + its day, the timetable spans four days. A -R1-> S -R2-> B, S
// changes in 5 min.
// ===========================================================================

namespace {

struct dated_trip {
  std::string_view id_;
  std::string_view route_;
  unsigned day_;
  std::vector<std::pair<std::string_view, std::string_view>> times_;
};

timetable load_utc_feed(std::vector<dated_trip> const& trips,
                        std::string_view const transfers) {
  auto f = std::string{
      "# agency.txt\n"
      "agency_id,agency_name,agency_url,agency_timezone\n"
      "AG,Agency,https://example.com,UTC\n\n"
      "# calendar_dates.txt\n"
      "service_id,date,exception_type\n"
      "D0,20190501,1\n"
      "D1,20190502,1\n"
      "D2,20190503,1\n\n"
      "# stops.txt\n"
      "stop_id,stop_name,stop_lat,stop_lon\n"
      "A,A,50.0,6.0\n"
      "S,S,50.0,6.5\n"
      "B,B,50.0,7.0\n\n"
      "# routes.txt\n"
      "route_id,agency_id,route_short_name,route_long_name,route_type\n"
      "R1,AG,R1,,3\n"
      "R2,AG,R2,,3\n\n"
      "# trips.txt\n"
      "route_id,service_id,trip_id\n"};
  for (auto const& x : trips) {
    f += fmt::format("{},D{},{}\n", x.route_, x.day_, x.id_);
  }
  f += "\n# stop_times.txt\n"
       "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n";
  for (auto const& x : trips) {
    auto seq = 1U;
    for (auto const& [stop, hhmm] : x.times_) {
      f +=
          fmt::format("{},{}:00,{}:00,{},{}\n", x.id_, hhmm, hhmm, stop, seq++);
    }
  }
  f += "\n# transfers.txt\n"
       "from_stop_id,to_stop_id,transfer_type,min_transfer_time,"
       "from_route_id,to_route_id,from_trip_id,to_trip_id\n"
       "S,S,2,300,,,,\n";
  f += transfers;

  auto tt = timetable{};
  tt.date_range_ = {test::kDay, test::kDay + date::days{4}};
  loader::register_special_stations(tt);
  loader::gtfs::load_timetable({}, source_idx_t{0U}, loader::mem_dir::read(f),
                               tt);
  loader::finalize(tt);
  return tt;
}

std::optional<unixtime_t> utc_arrival(timetable const& tt,
                                      std::string_view const time) {
  return test::arrival(
      raptor_search(tt, nullptr, "A", "B", fmt::format("{} UTC", time)));
}

}  // namespace

// T2 leaves S 3 min after T1 arrives, on the next day - or two days after T1
// started, for a T1 that arrives at S at 49:58: only the guarantee makes the
// change.
TEST(gtfs, transfer_rules_guarantee_across_midnight) {
  struct test_case {
    std::vector<dated_trip> trips_;
    char const* start_;
    char const* without_;
    char const* with_;
  };
  for (auto const& c : std::vector<test_case>{
           {{{"T1", "R1", 0U, {{"A", "23:30"}, {"S", "23:58"}}},
             {"T2", "R2", 1U, {{"S", "00:01"}, {"B", "00:30"}}},
             {"T2L", "R2", 1U, {{"S", "00:20"}, {"B", "00:50"}}}},
            "2019-05-01 23:00",
            "2019-05-02 00:50 UTC",
            "2019-05-02 00:30 UTC"},
           {{{"T1", "R1", 0U, {{"A", "22:00"}, {"S", "49:58"}}},
             {"T2", "R2", 2U, {{"S", "02:01"}, {"B", "02:30"}}},
             {"T2L", "R2", 2U, {{"S", "02:20"}, {"B", "02:50"}}}},
            "2019-05-01 21:00",
            "2019-05-03 02:50 UTC",
            "2019-05-03 02:30 UTC"}}) {
    SCOPED_TRACE(c.start_);
    EXPECT_EQ(t(c.without_),
              utc_arrival(load_utc_feed(c.trips_, ""), c.start_));
    auto const with = load_utc_feed(c.trips_, "S,S,1,,,,T1,T2\n");
    ASSERT_LT(0U, n_virts(with));
    EXPECT_EQ(t(c.with_), utc_arrival(with, c.start_));
  }
}

// T1G and T1 run on R1 one minute apart. Only T1G has a guarantee onto T2
// (next day, 4 min). T1 must not share T1G's virtual location: its change to
// T2 (3 min, across midnight) would get the guarantee.
TEST(gtfs, transfer_rules_change_across_midnight_separates_trips) {
  auto const tt =
      load_utc_feed({{"T1G", "R1", 0U, {{"A", "23:27"}, {"S", "23:57"}}},
                     {"T1", "R1", 0U, {{"A", "23:28"}, {"S", "23:58"}}},
                     {"T2", "R2", 1U, {{"S", "00:01"}, {"B", "00:30"}}},
                     {"T2L", "R2", 1U, {{"S", "00:20"}, {"B", "00:50"}}}},
                    "S,S,1,,,,T1G,T2\n");
  ASSERT_LT(0U, n_virts(tt));

  EXPECT_EQ(t("2019-05-02 00:30 UTC"), utc_arrival(tt, "2019-05-01 23:27"));
  EXPECT_EQ(t("2019-05-02 00:50 UTC"), utc_arrival(tt, "2019-05-01 23:28"));
}

// ===========================================================================
// GTFS transfers.txt semantics.
// ===========================================================================

// transfer_type 5: no in-seat transfer between K5T1 and K5T2 although they
// share a block - the rider has to alight, and a 0 min change is not enough.
TEST(gtfs, transfer_rules_type5_prevents_block_through_service) {
  auto const tt = load_feeds(
      {feed({{"K5S", 62.0, 21.0}, {"K5A", 62.1, 21.0}, {"K5B", 62.2, 21.0}},
            {{"K5T1", "RK1", {{"K5A", "10:00"}, {"K5S", "10:30"}}, "K5"},
             {"K5T2", "RK2", {{"K5S", "10:30"}, {"K5B", "11:00"}}, "K5"}},
            "K5S,K5S,5,,,,K5T1,K5T2\n")});
  EXPECT_EQ(0U, search_at(tt, "K5A", "K5B", "10:00").size());
}

// A profile keeps the stop's plain change time (2 min) for trips no rule names:
// the 0 s row for the trip pair Q1 -> Q2 says nothing about Q3 -> Q4 (1 min
// gap).
TEST(gtfs,
     transfer_rules_qualified_row_does_not_set_stop_time_for_other_profiles) {
  constexpr auto const kProfile = profile_idx_t{1U};
  auto tt =
      load_feeds({feed({{"QS", 63.0, 22.0},
                        {"QA", 63.1, 22.0},
                        {"QB", 63.2, 22.0},
                        {"QC", 63.1, 22.3},
                        {"QD", 63.2, 22.3}},
                       {{"Q1", "RQ1", {{"QA", "10:00"}, {"QS", "10:30"}}},
                        {"Q2", "RQ2", {{"QS", "10:40"}, {"QB", "11:00"}}},
                        {"Q3", "RQ3", {{"QC", "10:00"}, {"QS", "10:30"}}},
                        {"Q4", "RQ4", {{"QS", "10:31"}, {"QD", "11:00"}}},
                        {"Q4L", "RQ4", {{"QS", "10:40"}, {"QD", "11:10"}}}},
                       "QS,QS,2,120,,,,\n"
                       "QS,QS,2,0,,,Q1,Q2\n")});
  add_empty_profile(tt, kProfile);

  auto const res = raptor_search(
      tt, nullptr,
      routing::query{.start_time_ = at("10:00"),
                     .start_ = {{lidx(tt, "QC"), 0_minutes, 0U}},
                     .destination_ = {{lidx(tt, "QD"), 0_minutes, 0U}},
                     .prf_idx_ = kProfile});
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("11:10"), begin(res)->dest_time_);
}

// min_transfer_time is a non-negative number of seconds. A negative value is
// invalid input: at most it means "no time", never "no transfer". 36000 s =
// 600 min is valid (if long): it exceeds what a footpath can hold (511 min),
// so it is capped - but it is no ban, and neither wraps a huge value. NT2
// leaves 10 min, NT2L 9 h after NT1 arrives.
TEST(gtfs, transfer_rules_min_transfer_time_is_not_a_ban) {
  for (auto const& [seconds, arr] :
       {std::tuple{"-120", "11:00"}, std::tuple{"36000", "19:50"},
        std::tuple{"2000000", "19:50"}}) {
    SCOPED_TRACE(seconds);
    auto const tt =
        load_feeds({feed({{"NA1", 64.0, 23.0},
                          {"NA2", 64.0003, 23.0},
                          {"N0", 64.1, 23.0},
                          {"N9", 64.2, 23.0}},
                         {{"NT1", "RN1", {{"N0", "10:00"}, {"NA1", "10:30"}}},
                          {"NT2", "RN2", {{"NA2", "10:40"}, {"N9", "11:00"}}},
                          {"NT2L", "RN2", {{"NA2", "19:30"}, {"N9", "19:50"}}}},
                         fmt::format("NA1,NA2,2,{},,,,\n", seconds))});
    EXPECT_EQ(at(arr), arrival_at(tt, "N0", "N9", "10:00"));
  }
}

// min_transfer_time only matters for types 0 and 2: an invalid one does not
// cost a ban (type 3) or a guarantee (type 1) its row.
TEST(gtfs, transfer_rules_min_transfer_time_of_other_types_is_ignored) {
  auto const load = [](std::string_view const rows) {
    return load_feeds(
        {feed({{"K", 50.0, 6.5}, {"O", 50.0, 6.0}, {"D", 50.0, 7.0}},
              {{"F", "RF", {{"O", "10:00"}, {"K", "10:30"}}},
               {"G", "RG", {{"K", "10:31"}, {"D", "11:00"}}},
               {"GL", "RG", {{"K", "10:45"}, {"D", "11:10"}}}},
              rows)});
  };
  EXPECT_EQ(std::nullopt,
            arrival_at(load("K,K,3,-60,,,F,G\nK,K,3,-60,,,F,GL\n"), "O", "D",
                       "10:00"));
  EXPECT_EQ(at("11:00"),
            arrival_at(load("K,K,1,-60,,,F,G\n"), "O", "D", "10:00"));
}

// ===========================================================================
// Walks of virtual locations and rule footpaths: a slower rule keeps the pair
// out of the walks, the pair gets no walk footpath.
// ===========================================================================

// A rule from station BS (route RB) to the platform BY1 of station BY states
// 10 min: 2 virtual locations at BP1 x 3 locations at BY1. BS -> BY1 is a
// mixed pair, so the fold gives no default for BP1 -> BY1, and the 2 min walk
// BP1 -> BY1 stays. It must not take BP1's virtual locations along at 2 min:
// BT1 (10:30) misses BU1 (10:35) and takes BU3.
TEST(gtfs, transfer_rules_virt_walk_respects_rule_pair) {
  auto const tt =
      load_feeds({feed({{"BS", 55.0, 11.0, "", true},
                        {"BP1", 55.0, 11.0, "BS"},
                        {"BY", 55.0005, 11.0, "", true},
                        {"BY1", 55.0005, 11.0, "BY"},
                        {"BA", 55.1, 11.0},
                        {"BD", 55.2, 11.0}},
                       {{"BT1", "RB", {{"BA", "10:00"}, {"BP1", "10:30"}}},
                        {"BT2", "RB", {{"BA", "10:01"}, {"BP1", "10:31"}}},
                        {"BU1", "RU1", {{"BY1", "10:35"}, {"BD", "11:00"}}},
                        {"BU2", "RU2", {{"BY1", "11:30"}, {"BD", "12:00"}}},
                        {"BU3", "RU3", {{"BY1", "10:45"}, {"BD", "11:10"}}}},
                       "BS,BY1,2,600,RB,,,\n"
                       "BP1,BP1,2,120,,,,\n"
                       "BP1,BP1,2,300,,,BT1,\n"
                       "BY1,BY1,2,120,,,,\n"
                       "BY1,BY1,2,300,,,,BU1\n"
                       "BY1,BY1,2,400,,,,BU2\n")});
  ASSERT_LT(0U, n_virts(tt));

  EXPECT_EQ(at("11:10"), arrival_at(tt, "BA", "BD", "10:00"));
}

// A rule qualified by a route binds only that route's trips: BS -> BY1 (RB,
// 10 min) leaves the walk BP1 -> BY1 (55 m: 2 min) of the unnamed trip X
// alone, and a qualified ban between two stops 2 km apart creates no walk
// between them for anyone else.
TEST(gtfs, transfer_rules_qualified_rule_keeps_walks_of_other_trips) {
  auto const tt =
      load_feeds({feed({{"BS", 55.0, 11.0, "", true},
                        {"BP1", 55.0, 11.0, "BS"},
                        {"BY", 55.0005, 11.0, "", true},
                        {"BY1", 55.0005, 11.0, "BY"},
                        {"BA", 55.1, 11.0},
                        {"BA2", 55.1, 11.2},
                        {"BD", 55.2, 11.0}},
                       {{"BT", "RB", {{"BA2", "10:00"}, {"BP1", "10:30"}}},
                        {"X", "RX", {{"BA", "10:00"}, {"BP1", "10:30"}}},
                        {"Y", "RY", {{"BY1", "10:35"}, {"BD", "11:00"}}},
                        {"YL", "RY", {{"BY1", "10:45"}, {"BD", "11:10"}}}},
                       "BS,BY1,2,600,RB,,,\n")});
  ASSERT_LT(0U, n_virts(tt));
  EXPECT_EQ(at("11:10"), arrival_at(tt, "BA2", "BD", "10:00"));
  EXPECT_EQ(at("11:00"), arrival_at(tt, "BA", "BD", "10:00"));

  auto const distant =
      load_feeds({feed({{"P", 56.0, 12.0},
                        {"Q", 56.018, 12.0},
                        {"PA", 56.1, 12.0},
                        {"QB", 56.2, 12.0}},
                       {{"TP", "RP", {{"PA", "10:00"}, {"P", "10:30"}}},
                        {"TQ", "RQ", {{"Q", "11:30"}, {"QB", "12:00"}}}},
                       "P,Q,3,,RX1,RX2,,\n", {"RX1", "RX2"})});
  EXPECT_EQ(std::nullopt, arrival_at(distant, "PA", "QB", "10:00"));
}

// A route that visits S twice: the rule R1 -> R2 (2 min) binds both visits.
// At the first visit the change works (3 min gap), at the second it does not
// (1 min gap), although both are the same route at the same stop.
//     T1 (R1): A 09:50 -> S 10:00 -> B 10:10 -> S 10:20 -> C 10:30
//     T2 (R2): S 10:03 -> D 10:30 (3 min after the first visit -> ok)
//     T3 (R2): S 10:21 -> D 10:48 (1 min after the second visit < 2 -> not)
//     T4 (R2): S 10:40 -> D 11:07
TEST(gtfs, transfer_rules_route_visits_stop_twice) {
  auto const tt =
      load_feeds({feed({{"A", 53.0, 9.0},
                        {"S", 53.1, 9.0},
                        {"B", 53.2, 9.0},
                        {"C", 53.3, 9.0},
                        {"D", 53.1, 9.3}},
                       {{"T1",
                         "R1",
                         {{"A", "09:50"},
                          {"S", "10:00"},
                          {"B", "10:10"},
                          {"S", "10:20"},
                          {"C", "10:30"}}},
                        {"T2", "R2", {{"S", "10:03"}, {"D", "10:30"}}},
                        {"T3", "R2", {{"S", "10:21"}, {"D", "10:48"}}},
                        {"T4", "R2", {{"S", "10:40"}, {"D", "11:07"}}}},
                       "S,S,2,300,,,,\n"
                       "S,S,2,120,R1,R2,,\n")});
  ASSERT_LT(0U, n_virts(tt));

  EXPECT_EQ(at("10:30"), arrival_at(tt, "A", "D", "09:50"));
  EXPECT_EQ(at("11:07"), arrival_at(tt, "B", "D", "10:10"));
}

// The same, but the rule names only trip T1 of R1: T1 and T5 have the same
// stops, so they share a route, but their signatures at S differ. Each visit
// is its own route node, the first visit splits R1, and T1's two visits end up
// in different locations (first: 2 min, second: S's 5 min).
//     T1 (R1): A 09:50 -> S 10:00 -> B 10:10 -> S 10:20 -> C 10:30
//     T5 (R1): A 10:50 -> S 11:00 -> B 11:10 -> S 11:20 -> C 11:30
//     T2/T3/T4 (R2): S 10:03/10:21/10:40 -> D 10:30/10:48/11:07
//     T6/T7/T8 (R2): S 11:03/11:21/11:40 -> D 11:30/11:48/12:07
TEST(gtfs, transfer_rules_route_visits_stop_twice_trips_differ) {
  auto const tt =
      load_feeds({feed({{"A", 53.0, 9.0},
                        {"S", 53.1, 9.0},
                        {"B", 53.2, 9.0},
                        {"C", 53.3, 9.0},
                        {"D", 53.1, 9.3}},
                       {{"T1",
                         "R1",
                         {{"A", "09:50"},
                          {"S", "10:00"},
                          {"B", "10:10"},
                          {"S", "10:20"},
                          {"C", "10:30"}}},
                        {"T5",
                         "R1",
                         {{"A", "10:50"},
                          {"S", "11:00"},
                          {"B", "11:10"},
                          {"S", "11:20"},
                          {"C", "11:30"}}},
                        {"T2", "R2", {{"S", "10:03"}, {"D", "10:30"}}},
                        {"T3", "R2", {{"S", "10:21"}, {"D", "10:48"}}},
                        {"T4", "R2", {{"S", "10:40"}, {"D", "11:07"}}},
                        {"T6", "R2", {{"S", "11:03"}, {"D", "11:30"}}},
                        {"T7", "R2", {{"S", "11:21"}, {"D", "11:48"}}},
                        {"T8", "R2", {{"S", "11:40"}, {"D", "12:07"}}}},
                       "S,S,2,300,,,,\n"
                       "S,S,2,120,,R2,T1,\n")});
  ASSERT_LT(0U, n_virts(tt));

  EXPECT_EQ(at("10:30"), arrival_at(tt, "A", "D", "09:50"));
  EXPECT_EQ(at("11:07"), arrival_at(tt, "B", "D", "10:10"));
  EXPECT_EQ(at("11:48"), arrival_at(tt, "A", "D", "10:50"));
  EXPECT_EQ(at("12:07"), arrival_at(tt, "B", "D", "11:10"));
}
