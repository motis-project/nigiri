#include "gtest/gtest.h"

#include <optional>
#include <string>

#include "nigiri/loader/register.h"
#include "nigiri/loader/transfer_rules.h"
#include "nigiri/routing/for_each_hub_source.h"
#include "nigiri/routing/query.h"
#include "nigiri/timetable.h"

#include "../../raptor_search.h"
#include "../../transfer_rules_util.h"

// What transfers.txt rules mean, through the GTFS loader. Every test states
// its feed (test::feed) on 2019-05-01, Europe/Berlin; without a rule, a change
// at a stop takes 2 min.

using namespace nigiri;
using namespace std::string_view_literals;
using nigiri::test::add_empty_profile;
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
  auto const res = search_at(tt, "A", "B", "10:00");
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("11:10"), begin(res)->dest_time_);
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
  // footpath::kMaxDuration would be reachable here.
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

  EXPECT_EQ(0U, search_at(tt, "A4", "BZ", "16:00").size());

  auto const res_cz = search_at(tt, "A4", "CZ", "16:00");
  ASSERT_EQ(1U, res_cz.size());
  EXPECT_EQ(at("17:10"), begin(res_cz)->dest_time_);
}

// Guaranteed connection (MetroNorth pattern): M,M type=1 for the trip pair
// M1 -> M2, no min_transfer_time. The vehicle waits, so that pair costs
// 0 min - but a guarantee says nothing about how long a change at M takes,
// so it does not become the stop's default.
//     M1 (R30): A5 18:00 -> M 18:30
//     M2 (R31): M 18:30 -> G 19:00 (0 min -> reachable)
//     M3 (R32): M 18:30 -> H 19:00 (unnamed, default 2 min -> not reachable)
TEST(gtfs, transfer_rules_guaranteed_connection) {
  auto const tt = load_feeds({feed(
      {{"A5", 54.0, 6.0}, {"M", 54.0, 6.5}, {"G", 54.0, 7.0}, {"H", 54.0, 7.5}},
      {{"M1", "R30", {{"A5", "18:00"}, {"M", "18:30"}}},
       {"M2", "R31", {{"M", "18:30"}, {"G", "19:00"}}},
       {"M3", "R32", {{"M", "18:30"}, {"H", "19:00"}}}},
      "M,M,1,,,,M1,M2\n")});

  EXPECT_EQ(duration_t{2}, tt.locations_.transfer_time_[lidx(tt, "M")]);

  auto const res_g = search_at(tt, "A5", "G", "18:00");
  ASSERT_EQ(1U, res_g.size());
  EXPECT_EQ(at("19:00"), begin(res_g)->dest_time_);

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

  EXPECT_EQ(0U, search_at(tt, "A8", "K", "22:00").size());

  auto const res_l = search_at(tt, "A8", "L", "22:00");
  ASSERT_EQ(1U, res_l.size());
  EXPECT_EQ(at("23:00"), begin(res_l)->dest_time_);
}

// Mutually restricting trip pairs: Q,Q type=2 600s for Q1 -> Q2 and for
// Q2 -> Q1, plus the unqualified Q,Q type=2 120s row. Both virtual locations
// are the source AND the target of a slower-than-default transfer, so neither
// is unrestricted.
//     Q1 (RQ1): A9 08:00 -> Q 08:30
//     Q2 (RQ2): Q 08:32 -> QA 09:00 (2 min gap < 10 -> not reachable)
//     Q2L (RQ2): Q 08:41 -> QA 09:10 (11 min gap -> ok)
//     Q3 (RQ3): Q 08:32 -> QB 09:00 (unnamed, pair default 2 min -> ok)
TEST(gtfs, transfer_rules_mutual_restriction) {
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
                       "Q,Q,2,600,,,Q1,Q2\n"
                       "Q,Q,2,600,,,Q2,Q1\n")});

  auto const res_qa = search_at(tt, "A9", "QA", "08:00");
  ASSERT_EQ(1U, res_qa.size());
  EXPECT_EQ(at("09:10"), begin(res_qa)->dest_time_);

  // The unnamed trip still departs from the stop Q itself at the default.
  auto const res_qb = search_at(tt, "A9", "QB", "08:00");
  ASSERT_EQ(1U, res_qb.size());
  EXPECT_EQ(at("09:00"), begin(res_qb)->dest_time_);
}

// A slow member together with a restricted source: SS,SS type=2 600s
// from_route=RSS to_route=RSS makes the shared RSS virtual location slow - its
// own transfer time is 10 min, so it feeds no hub. SS,SS type=2 600s
// TS1 -> TS2 makes TS1 a restricted source. Plus the unqualified SS,SS type=2
// 120s row.
//     TS0 (RSS): A10 08:50 -> SS 09:20
//     TS1 (RSX): A10 09:00 -> SS 09:30
//     TS2 (RSY): SS 09:32 -> SSA 10:00 (2 min gap < 10 -> not reachable)
//     TS3 (RSS): SS 09:33 -> SSB 10:00 (TS1 -> the slow member at the 2 min
//                default: nothing slow leads to it, so it stays reachable)
//     TS4 (RSY): SS 09:23 -> SSC 10:00 (leaving the slow member at the
//                default)
TEST(gtfs, transfer_rules_slow_member) {
  auto const tt =
      load_feeds({feed({{"A10", 58.0, 6.0},
                        {"SS", 58.0, 6.5},
                        {"SSA", 58.0, 7.0},
                        {"SSB", 58.0, 7.5},
                        {"SSC", 58.0, 8.0}},
                       {{"TS0", "RSS", {{"A10", "08:50"}, {"SS", "09:20"}}},
                        {"TS1", "RSX", {{"A10", "09:00"}, {"SS", "09:30"}}},
                        {"TS2", "RSY", {{"SS", "09:32"}, {"SSA", "10:00"}}},
                        {"TS3", "RSS", {{"SS", "09:33"}, {"SSB", "10:00"}}},
                        {"TS4", "RSY", {{"SS", "09:23"}, {"SSC", "10:00"}}}},
                       "SS,SS,2,120,,,,\n"
                       "SS,SS,2,600,RSS,RSS,,\n"
                       "SS,SS,2,600,,,TS1,TS2\n")});

  // TS1 -> TS2 is the slow pair: a 2 min gap is not enough for 10 min.
  EXPECT_EQ(0U, search_at(tt, "A10", "SSA", "09:00").size());

  // The same restricted source still reaches the slow member at the pair
  // default: nothing slow leads there, so it stays in the restricted hub.
  auto const res_b = search_at(tt, "A10", "SSB", "09:00");
  ASSERT_EQ(1U, res_b.size());
  EXPECT_EQ(at("10:00"), begin(res_b)->dest_time_);

  // Leaving the slow member: it feeds no hub at all, so even this plain
  // 2 min transfer has to be written explicitly.
  auto const res_c = search_at(tt, "A10", "SSC", "08:45");
  ASSERT_EQ(1U, res_c.size());
  EXPECT_EQ(at("10:00"), begin(res_c)->dest_time_);
}

// A rule that lets one route change without waiting must not become usable
// for another route at the same stop. GA and GB state the same slow rule
// footpath, GC the fast one: GA and GB may share a virtual location, but it
// must not get the fast rule footpath of GC, which would make GB's impossible
// connection possible.
TEST(gtfs, transfer_rules_fast_rule_stays_on_its_route) {
  auto const tt =
      load_feeds({feed({{"GO1", 59.0, 6.0},
                        {"GO2", 59.0, 6.1},
                        {"GO3", 59.0, 6.2},
                        {"GU", 59.0, 6.5},
                        {"GUD", 59.0, 7.0}},
                       {{"GA", "RGA", {{"GO1", "08:00"}, {"GU", "09:00"}}},
                        {"GB", "RGB", {{"GO2", "09:30"}, {"GU", "10:00"}}},
                        {"GC", "RGC", {{"GO3", "09:40"}, {"GU", "09:55"}}},
                        {"GX", "RGX", {{"GU", "10:01"}, {"GUD", "10:30"}}}},
                       "GU,GU,2,180,RGA,RGX,,\n"
                       "GU,GU,2,180,RGB,RGX,,\n"
                       "GU,GU,2,0,RGC,RGX,,\n")});

  // GC is the route the 0 min rule names: 09:55 + 0 catches the 10:01.
  auto const res_c = search_at(tt, "GO3", "GUD", "09:30");
  ASSERT_EQ(1U, res_c.size());
  EXPECT_EQ(at("10:30"), begin(res_c)->dest_time_);

  // GB arrives 10:00 and owes 3 min, so 10:01 is out of reach - it may not
  // borrow the 0 min rule from GC.
  EXPECT_EQ(0U, search_at(tt, "GO2", "GUD", "09:00").size());

  // GA states the same 3 min as GB and has the slack for it.
  auto const res_a = search_at(tt, "GO1", "GUD", "07:30");
  ASSERT_EQ(1U, res_a.size());
  EXPECT_EQ(at("10:30"), begin(res_a)->dest_time_);
}

// Unqualified same-stop ban: T6N,T6N type=3 says no change is possible at
// T6N, for anyone. It becomes the stop's own transfer time, and a ban is not a
// very long time: neither the 15 min nor the 3 h connection may exist.
// Getting off at T6N as the destination is still fine.
//     T6X1 (R76): T6A 14:30 -> T6N 15:00
//     T6Y1 (R77): T6N 15:15 -> T6J 15:30
//     T6Y2 (R77): T6N 18:00 -> T6J 18:20
TEST(gtfs, transfer_rules_forbidden_same_stop_unqualified) {
  auto const tt = load_feeds(
      {feed({{"T6A", 60.0, 6.0}, {"T6N", 60.0, 6.5}, {"T6J", 60.0, 7.0}},
            {{"T6X1", "R76", {{"T6A", "14:30"}, {"T6N", "15:00"}}},
             {"T6Y1", "R77", {{"T6N", "15:15"}, {"T6J", "15:30"}}},
             {"T6Y2", "R77", {{"T6N", "18:00"}, {"T6J", "18:20"}}}},
            "T6N,T6N,3,,,,,\n")});

  EXPECT_EQ(1U, search_at(tt, "T6A", "T6N", "14:30").size());
  EXPECT_EQ(0U, raptor_search(tt, nullptr, "T6A", "T6J",
                              interval{at("14:30"), at("22:00")})
                    .size());
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

  auto const allowed = search_at(tt, "T7A", "T7L", "14:30");
  ASSERT_EQ(1U, allowed.size());
  EXPECT_EQ(at("15:30"), begin(allowed)->dest_time_);

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

  auto const allowed = search_at(tt, "T8A", "T8K", "09:30");
  ASSERT_EQ(1U, allowed.size());
  EXPECT_EQ(at("10:25"), begin(allowed)->dest_time_);

  EXPECT_EQ(0U, raptor_search(tt, nullptr, "T8A", "T8L",
                              interval{at("09:30"), at("16:00")})
                    .size());

  auto const walked = search_at(tt, "T8A", "T8M", "09:30");
  ASSERT_EQ(1U, walked.size());
  EXPECT_EQ(at("10:37"), begin(walked)->dest_time_);
}

// The same journey searched backwards (arrive by 12:00). The walk into T8M
// ends the journey: it starts when the ride before it ends, at 10:35, and not
// as late as the search start allows. CPU and GPU search have to agree.
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
// hub would cover - but a hub gives each of its pairs its weight, and a ban is
// none: written as a hub, the pairs stay walkable and become reachable after
// 511 minutes.
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

  EXPECT_EQ(0U, raptor_search(tt, nullptr, "T9A", "T9B",
                              interval{at("09:00"), at("20:00")})
                    .size());

  auto const walked = search_at(tt, "T9A", "T9C", "09:00");
  ASSERT_EQ(1U, walked.size());
  EXPECT_EQ(at("10:00"), begin(walked)->dest_time_);
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

  auto const res = raptor_search(tt, nullptr, "T10O", "T10K",
                                 interval{at("09:29"), at("09:31")});
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("09:29"), begin(res)->start_time_);
  EXPECT_EQ(at("10:30"), begin(res)->dest_time_);
}

// A block where only the arriving trip's last stop gets a virtual location:
// trip a (route BR) ends at BS, where a rule for boarding route BR applies,
// and the vehicle goes on as b (route BQ). Nobody boards a or gets off b at
// the handover stop, so no rule applies there and the vehicle runs through: b
// leaves when a arrives, too early for a change.
TEST(gtfs, transfer_rules_block_handover_stop_without_rules) {
  constexpr auto const kBlockFeed = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
AG,Agency,https://example.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,location_type,parent_station
BA,BA,,50.0,6.0,,
BS,BS,,50.0,6.5,,
BB,BB,,50.0,7.0,,
BX,BX,,50.0,8.0,,

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
BR,AG,BR,,,3
BQ,AG,BQ,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
BR,S1,a,,blk
BQ,S1,b,,blk

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
a,10:00:00,10:00:00,BA,1
a,10:30:00,10:30:00,BS,2
b,10:30:00,10:30:00,BS,1
b,11:00:00,11:00:00,BB,2

# transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time,from_route_id,to_route_id
BX,BS,2,120,,
BX,BS,2,300,,BR
)"sv;

  auto const tt = load_feeds({std::string{kBlockFeed}});
  ASSERT_EQ(1U, n_virts(tt))
      << "precondition: a's last stop is a virtual location";

  auto const res = search_at(tt, "BA", "BB", "09:50");
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(0U, begin(res)->transfers_);
  EXPECT_EQ(at("11:00"), begin(res)->dest_time_);
}

// A virtual location must not get walks of its own into other feeds: it
// leaves through its stop. On test::kNetwork, S1 takes 5 min to change, the
// RF trips among themselves 0 min - their virtual location must not reach the
// other feed's stop Z (50 m away) faster than S1 does.
TEST(gtfs, transfer_rules_virtual_location_not_linked_to_other_feeds) {
  constexpr auto const kOtherFeed = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
AG2,Agency 2,https://example.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
Z,Z,,50.0005,6.5,,,
Y,Y,,50.5,6.5,,,

# calendar_dates.txt
service_id,date,exception_type
X,20190501,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
RZ,AG2,RZ,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
RZ,X,TZ,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
TZ,12:00:00,12:00:00,Z,1,0,0
TZ,12:30:00,12:30:00,Y,2,0,0
)";
  auto const tt =
      load_feeds({test::network("S1,S1,2,300,,,,\nS1,S1,2,0,RF,RF,,"),
                  std::string{kOtherFeed}});

  auto const s1 = lidx(tt, "S1");
  auto const z = lidx(tt, "Z", source_idx_t{1});
  // The whole transfer relation: footpaths and what the hubs cover.
  auto const walk = [&](location_idx_t const from) {
    auto d = std::optional<duration_t>{};
    auto const take = [&](footpath const fp) {
      if (fp.target() == z && (!d.has_value() || fp.duration() < *d)) {
        d = fp.duration();
      }
      return true;
    };
    for (auto const fp : tt.locations_.footpaths_out_[kDefaultProfile][from]) {
      take(fp);
    }
    routing::for_each_hub_source<direction::kBackward>(tt, kDefaultProfile,
                                                       from, take);
    return d;
  };
  ASSERT_TRUE(walk(s1).has_value());
  auto n_virts = 0U;
  for (auto const c : tt.locations_.children_[s1]) {
    if (tt.locations_.types_[c] == location_type::kVirt) {
      ++n_virts;
      EXPECT_TRUE(!walk(c).has_value() || *walk(c) >= *walk(s1))
          << "virtual location reaches Z in " << walk(c)->count()
          << " min, its stop in " << walk(s1)->count();
    }
  }
  EXPECT_NE(0U, n_virts);
}

// ===========================================================================
// GTFS: a rule naming a station applies to all its child stops - also to a
// change that stays at one child stop.
// ===========================================================================

// YS,YS,3: no change between routes anywhere in station YS - also not at the
// stop Y1 itself.
TEST(gtfs, transfer_rules_station_ban_applies_at_one_base_location) {
  auto const tt =
      load_feeds({feed({{"YS", 50.5, 8.5, "", true},
                        {"Y1", 50.5, 8.5, "YS"},
                        {"Y2", 50.5003, 8.5, "YS"},
                        {"C", 50.6, 8.5},
                        {"D", 50.7, 8.5}},
                       {{"TE1", "RE1", {{"C", "10:00"}, {"Y1", "10:30"}}},
                        {"TE2", "RE2", {{"Y1", "10:40"}, {"D", "11:00"}}}},
                       "YS,YS,3,,,,,\n")});
  EXPECT_EQ(0U, search_at(tt, "C", "D", "10:00").size());
}

// ===========================================================================
// The fold's majority must not override an explicit row. MS1 -> MP2 only has
// a trip-qualified row (TA -> TB, 10 min), from which the fold takes a
// 10 min default for MS1 -> MP2. At MP1 -> MP2 that default would be as
// specific as the explicit row MP1 -> MS2 (3 min) and win the tie, so the
// unnamed trips TX -> TY would get 10 min instead of the stated 3 min.
// ===========================================================================

TEST(gtfs, transfer_rules_majority_does_not_override_explicit_row) {
  auto const tt =
      load_feeds({feed({{"MS1", 49.5, 9.5, "", true},
                        {"MP1", 49.5, 9.5, "MS1"},
                        {"MS2", 49.5003, 9.5, "", true},
                        {"MP2", 49.5003, 9.5, "MS2"},
                        {"MC", 49.6, 9.5},
                        {"MD", 49.7, 9.5},
                        {"MC2", 49.6, 9.7},
                        {"MD2", 49.7, 9.7}},
                       {{"TX", "RX", {{"MC", "10:00"}, {"MP1", "10:30"}}},
                        {"TY", "RY", {{"MP2", "10:35"}, {"MD", "11:00"}}},
                        {"TA", "RA", {{"MC2", "10:00"}, {"MP1", "10:30"}}},
                        {"TB", "RB", {{"MP2", "10:35"}, {"MD2", "11:00"}}}},
                       "MP1,MS2,2,180,,,,\n"
                       "MS1,MP2,2,600,,,TA,TB\n")});

  // Unnamed trips: only the explicit 3 min apply, the 5 min change works.
  auto const unnamed = search_at(tt, "MC", "MD", "10:00");
  ASSERT_EQ(1U, unnamed.size());
  EXPECT_EQ(at("11:00"), begin(unnamed)->dest_time_);

  // TA -> TB: the trip-qualified 10 min are more specific, 5 min fail.
  EXPECT_EQ(0U, search_at(tt, "MC2", "MD2", "10:00").size());
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

  // Control: RB -> RB2 respects the 10 min.
  auto const other_route = search_at(tt, "E2", "F2", "12:00");
  ASSERT_EQ(1U, other_route.size());
  EXPECT_EQ(at("13:15"), begin(other_route)->dest_time_);

  // RB -> RB: the same rule, so TB2 (5 min) cannot be reached either.
  auto const same_route = search_at(tt, "E", "F", "12:00");
  ASSERT_EQ(1U, same_route.size());
  EXPECT_EQ(at("13:15"), begin(same_route)->dest_time_);
}

// ===========================================================================
// The virtual location key has to keep the rule's specificity: CA is named by
// a trip rule (5 min), CA2 only by the route rules. For CA2 -> CC2 the most
// specific rule is RC -> RC2 (10 min), not CA's trip rule.
// ===========================================================================

TEST(gtfs, transfer_rules_virtual_location_key_keeps_specificity) {
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

  // Control: CA -> CC, the trip rule (one trip beats both routes) gives 5 min.
  auto const ca = search_at(tt, "G", "H", "13:00");
  ASSERT_EQ(1U, ca.size());
  EXPECT_EQ(at("14:00"), begin(ca)->dest_time_);

  // CA2 -> CC2 (6 min) needs the route pair's 10 min: CC3.
  auto const ca2 = search_at(tt, "G", "H", "14:00");
  ASSERT_EQ(1U, ca2.size());
  EXPECT_EQ(at("15:10"), begin(ca2)->dest_time_);
}

// Without a rule of another duration in between, the specificity does not
// decide: CA ("trip CA -> anything: 5 min") and CA2 ("route RC -> anything: 5
// min") get 5 min for every partner either way, so they keep sharing one
// virtual location - and CA2 makes CC2 (6 min).
TEST(
    gtfs,
    transfer_rules_virtual_location_key_merges_equal_values_without_competition) {
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
                       "S,S2,2,300,RC,,,\n")});
  EXPECT_EQ(1U, n_virts(tt));
  auto const ca2 = search_at(tt, "G", "H", "14:00");
  ASSERT_EQ(1U, ca2.size());
  EXPECT_EQ(at("15:00"), begin(ca2)->dest_time_);
}

// ===========================================================================
// Block through-services and stay-seated chains.
// ===========================================================================

// IT1 and IT2 form one block (through ride at BS). A guarantee from another
// trip into IT2 gives IT2's first stop a virtual location - the
// through ride has to stay, exactly like the control block CT1 + CT2.
TEST(gtfs,
     transfer_rules_block_through_service_survives_rule_at_handover_stop) {
  auto const tt = load_feeds(
      {feed({{"BS", 58.0, 16.0},
             {"BA", 58.1, 16.0},
             {"BB", 58.2, 16.0},
             {"BZ", 58.1, 16.3},
             {"CS", 58.5, 16.5},
             {"CA", 58.6, 16.5},
             {"CB", 58.7, 16.5}},
            {{"IT1", "RI1", {{"BA", "10:00"}, {"BS", "10:30"}}, "K1"},
             {"IT2", "RI2", {{"BS", "10:30"}, {"BB", "11:00"}}, "K1"},
             {"IT0", "RI3", {{"BZ", "09:00"}, {"BS", "10:20"}}},
             {"CT1", "RC1", {{"CA", "10:00"}, {"CS", "10:30"}}, "K2"},
             {"CT2", "RC9", {{"CS", "10:30"}, {"CB", "11:00"}}, "K2"}},
            "BS,BS,1,,,,IT0,IT2\n")});

  auto const control = search_at(tt, "CA", "CB", "10:00");
  ASSERT_EQ(1U, control.size());
  EXPECT_EQ(at("11:00"), begin(control)->dest_time_);
  EXPECT_EQ(1U, n_transit_legs(*begin(control)));

  auto const res = search_at(tt, "BA", "BB", "10:00");
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("11:00"), begin(res)->dest_time_);
  EXPECT_EQ(1U, n_transit_legs(*begin(res)));
}

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

  // Control: the chain itself works.
  auto const chain = search_at(tt, "SA", "SB", "10:00");
  ASSERT_EQ(1U, chain.size());
  EXPECT_EQ(at("11:00"), begin(chain)->dest_time_);

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
  auto const control = search_at(plain, "JA", "JB", "06:17");
  ASSERT_EQ(1U, control.size());
  EXPECT_EQ(at("07:10"), begin(control)->dest_time_);

  auto const joined = load_feeds({make("BLJ")});
  auto const res = search_at(joined, "JA", "JB", "06:17");
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("07:10"), begin(res)->dest_time_);
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

// transfer_type 1: the departing vehicle waits and leaves sufficient time,
// so TT1 -> TT2 is guaranteed although the scheduled gap (2 min) is shorter
// than the stated min_transfer_time (5 min).
TEST(gtfs, transfer_rules_timed_transfer_with_min_time_is_guaranteed) {
  auto const tt = load_feeds(
      {feed({{"TTS", 62.5, 21.5}, {"TTA", 62.6, 21.5}, {"TTB", 62.7, 21.5}},
            {{"TT1", "RT1", {{"TTA", "10:00"}, {"TTS", "10:30"}}},
             {"TT2", "RT2", {{"TTS", "10:32"}, {"TTB", "11:00"}}},
             {"TT2L", "RT2", {{"TTS", "10:50"}, {"TTB", "11:20"}}}},
            "TTS,TTS,2,120,,,,\n"
            "TTS,TTS,1,300,,,TT1,TT2\n")});
  auto const res = search_at(tt, "TTA", "TTB", "10:00");
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("11:00"), begin(res)->dest_time_);
}

// A profile that ignores the qualified rules keeps the stop's plain change
// time (2 min). The 0 s row for the trip pair Q1 -> Q2 says nothing about
// Q3 -> Q4 (1 min gap).
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
// invalid input: at most it means "no time", never "no transfer".
TEST(gtfs, transfer_rules_negative_min_transfer_time_is_not_a_ban) {
  auto const tt =
      load_feeds({feed({{"NA1", 64.0, 23.0},
                        {"NA2", 64.0003, 23.0},
                        {"N0", 64.1, 23.0},
                        {"N9", 64.2, 23.0}},
                       {{"NT1", "RN1", {{"N0", "10:00"}, {"NA1", "10:30"}}},
                        {"NT2", "RN2", {{"NA2", "10:40"}, {"N9", "11:00"}}}},
                       "NA1,NA2,2,-120,,,,\n")});
  auto const res = search_at(tt, "N0", "N9", "10:00");
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("11:00"), begin(res)->dest_time_);
}

// 36000 s = 600 min is a valid (if long) min_transfer_time. It exceeds
// what a footpath can hold (511 min), but it is no ban: the 21:00 departure
// 10.5 h after the arrival is reachable.
TEST(gtfs, transfer_rules_very_large_min_transfer_time_is_not_a_ban) {
  auto const tt =
      load_feeds({feed({{"NB1", 64.5, 23.5},
                        {"NB2", 64.5003, 23.5},
                        {"NB0", 64.6, 23.5},
                        {"NB9", 64.7, 23.5}},
                       {{"NBT1", "RNB1", {{"NB0", "10:00"}, {"NB1", "10:30"}}},
                        {"NBT2", "RNB2", {{"NB2", "21:00"}, {"NB9", "21:30"}}}},
                       "NB1,NB2,2,36000,,,,\n")});
  auto const res = search_at(tt, "NB0", "NB9", "10:00");
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("21:30"), begin(res)->dest_time_);
}

// ===========================================================================
// Walk hubs and rule hubs. get_walk_hubs sees rule footpaths, not the pairs
// of a rule hub, so a rule hub is only safe where a rule speaks for its two
// stops as well - for GTFS, the fold's default does that. add_rule_hubs
// asserts it; this test builds rules without the GTFS loader to break it.
// ===========================================================================

// A rule states 10 min from A's virtual locations to everything at B: a cross
// product of 2 x 3 pairs, so it becomes a rule hub instead of rule footpaths.
// No rule speaks for A -> B itself, so a 3 min walk A -> B would go into a walk
// hub that takes the virtual locations along and undercuts the rule.
TEST(transfer_rules_DeathTest, rule_hub_without_stop_pair_rule) {
  auto tt = timetable{};
  loader::register_special_stations(tt);
  auto const add = [&](std::string_view const id, geo::latlng const pos,
                       location_type const type, location_idx_t const parent) {
    auto l = loader::location{};
    l.src_ = source_idx_t{0U};
    l.id_ = id;
    l.pos_ = pos;
    l.type_ = type;
    l.parent_ = parent;
    l.transfer_time_ = duration_t{2};
    auto const idx = loader::register_location(tt, l);
    if (parent != location_idx_t::invalid()) {
      tt.locations_.children_[parent].push_back(idx);
    }
    return idx;
  };
  auto const a =
      add("A", {50.0, 8.0}, location_type::kStation, location_idx_t::invalid());
  auto const b = add("B", {50.01, 8.0}, location_type::kStation,
                     location_idx_t::invalid());
  auto const va1 = add("", {50.0, 8.0}, location_type::kVirt, a);
  auto const va2 = add("", {50.0, 8.0}, location_type::kVirt, a);
  auto const vb1 = add("", {50.01, 8.0}, location_type::kVirt, b);
  auto const vb2 = add("", {50.01, 8.0}, location_type::kVirt, b);

  tt.transfer_rules_.rules_.push_back(transfer_rule{
      .from_stop_ = a, .to_stop_ = b, .duration_ = duration_t{10}});
  auto const rule = transfer_rule_idx_t{0U};
  auto most_specific = hash_map<loader::transfer_pair, transfer_rule_idx_t>{};
  for (auto const x : {va1, va2}) {
    for (auto const y : {b, vb1, vb2}) {
      most_specific[loader::transfer_pair{x, y}] = rule;
    }
  }
  auto transfers = loader::rule_transfers{};
  EXPECT_DEBUG_DEATH(loader::add_rule_hubs(tt, most_specific, transfers),
                     "most_specific\\.contains");
}
