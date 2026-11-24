#include "gtest/gtest.h"

#include <string_view>
#include <vector>

#include "nigiri/loader/load.h"
#include "nigiri/common/interval.h"
#include "nigiri/common/parse_time.h"
#include "nigiri/footpath.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/reconstruct.h"
#include "nigiri/special_stations.h"
#include "nigiri/types.h"

#include "../transfer_rules_util.h"

using namespace nigiri;
using namespace date;
using namespace std::chrono_literals;
using namespace std::string_view_literals;

namespace {

std::string print_journey(timetable const& tt,
                          routing::journey const& journey) {
  std::stringstream ss;
  ss << "\n";
  journey.print(ss, tt);
  return ss.str();
}

timetable load_timetable(std::string_view s) {
  auto const tt = loader::load(loader::mem_dir::read(s),
                               {2019_y / May / 1, 2019_y / May / 2});
  return tt;
}

location_idx_t loc_idx(timetable const& tt, std::string_view const id) {
  return tt.find(location_id{id, source_idx_t{0}}).value();
}

constexpr auto const test_files_1 = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

#stops.txt
stop_id,stop_name,stop_desc,stop_lat,stop_lon,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,
C,C,,4.0,5.0,,
D,D,,6.0,7.0,,
E,E,,8.0,9.0,,

#routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R0,DB,0,,,3
R1,DB,1,,,3
R2,DB,2,,,3

#trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R0,S1,T0,,
R1,S1,T1,,

#stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence
T0,10:10:00,10:10:00,A,0
T0,10:12:00,10:12:00,B,1
T0,10:15:00,10:15:00,C,2
T1,10:17:00,10:17:00,C,0
T1,10:33:00,10:33:00,D,1
T1,10:40:00,10:40:00,E,2

#calendar_dates.txt
service_id,date,exception_type
S1,20190501,1

#transfers.txt
from_stop_id,to_stop_id,transfer_type,min_transfer_time
)"sv;

}  // namespace

TEST(routing, optimize_initial_departure_test) {
  constexpr auto const expected_optimize_initial_departure = R"(
[2019-05-01 08:00, 2019-05-01 08:15]
TRANSFERS: 0
     FROM: (START, START) [2019-05-01 08:00]
       TO: (C, C) [2019-05-01 08:15]
leg 0: (START, START) [2019-05-01 08:00] -> (A, A) [2019-05-01 08:10]
  MUMO (payload=0, duration=10)
leg 1: (A, A) [2019-05-01 08:10] -> (C, C) [2019-05-01 08:15]
   0: A       A...............................................                               d: 01.05 08:10 [01.05 10:10]  [{name=0, day=2019-05-01, id=T0, src=0}]
   1: B       B............................................... a: 01.05 08:12 [01.05 10:12]  d: 01.05 08:12 [01.05 10:12]  [{name=0, day=2019-05-01, id=T0, src=0}]
   2: C       C............................................... a: 01.05 08:15 [01.05 10:15]
)"sv;

  auto const tt = load_timetable(test_files_1);

  auto const A = loc_idx(tt, "A");
  auto const B = loc_idx(tt, "B");
  auto const C = loc_idx(tt, "C");

  auto const q = routing::query{.start_ = {{B, 8_minutes, 0U}},
                                .via_stops_ = {{loc_idx(tt, "A"), 0_minutes}}};

  auto journey = routing::journey{};
  journey.transfers_ = 0U;
  journey.start_time_ = test::at("10:00");
  journey.dest_time_ = test::at("10:15");

  // Start -> A
  journey.add(routing::journey::leg{
      direction::kForward, get_special_station(special_station::kStart), A,
      test::at("10:00"), test::at("10:10"),
      routing::offset{A, 10_minutes, 0U}});

  // A -> B -> C
  journey.add(routing::journey::leg{
      direction::kForward, A, C, test::at("10:10"), test::at("10:15"),
      routing::journey::run_enter_exit{
          {
              .t_ = {transport_idx_t{0U}, day_idx_t{5U}},
              .stop_range_ = interval<stop_idx_t>{0U, 3U},
          },
          0,
          2}});

  optimize_footpaths(tt, nullptr, q, journey);
  EXPECT_EQ(expected_optimize_initial_departure, print_journey(tt, journey));
}

TEST(routing, optimize_last_arrival_test) {
  constexpr auto const expected_optimize_last_arrival = R"(
[2019-05-01 08:10, 2019-05-01 08:25]
TRANSFERS: 0
     FROM: (A, A) [2019-05-01 08:10]
       TO: (END, END) [2019-05-01 08:25]
leg 0: (A, A) [2019-05-01 08:10] -> (C, C) [2019-05-01 08:15]
   0: A       A...............................................                               d: 01.05 08:10 [01.05 10:10]  [{name=0, day=2019-05-01, id=T0, src=0}]
   1: B       B............................................... a: 01.05 08:12 [01.05 10:12]  d: 01.05 08:12 [01.05 10:12]  [{name=0, day=2019-05-01, id=T0, src=0}]
   2: C       C............................................... a: 01.05 08:15 [01.05 10:15]
leg 1: (C, C) [2019-05-01 08:15] -> (END, END) [2019-05-01 08:25]
  MUMO (payload=0, duration=10)
)"sv;

  auto const tt = load_timetable(test_files_1);

  auto const A = loc_idx(tt, "A");
  auto const B = loc_idx(tt, "B");
  auto const C = loc_idx(tt, "C");

  auto const q = routing::query{.destination_ = {{B, 8_minutes, 0U}},
                                .via_stops_ = {{loc_idx(tt, "C"), 0_minutes}}};

  auto journey = routing::journey{};
  journey.transfers_ = 0U;
  journey.start_time_ = test::at("10:10");
  journey.dest_time_ = test::at("10:25");

  // A -> B -> C
  journey.add(routing::journey::leg{
      direction::kForward, A, C, test::at("10:10"), test::at("10:15"),
      routing::journey::run_enter_exit{
          {
              .t_ = {transport_idx_t{0U}, day_idx_t{5U}},
              .stop_range_ = interval<stop_idx_t>{0U, 3U},
          },
          0,
          2}});

  // C -> END
  journey.add(routing::journey::leg{
      direction::kForward, C, get_special_station(special_station::kEnd),
      test::at("10:15"), test::at("10:25"),
      routing::offset{get_special_station(special_station::kEnd), 10_minutes,
                      0U}});

  optimize_footpaths(tt, nullptr, q, journey);
  EXPECT_EQ(expected_optimize_last_arrival, print_journey(tt, journey));
}

TEST(routing, optimize_transfers_test) {
  constexpr auto const expected_optimize_transfers = R"(
[2019-05-01 08:10, 2019-05-01 08:40]
TRANSFERS: 1
     FROM: (A, A) [2019-05-01 08:10]
       TO: (E, E) [2019-05-01 08:40]
leg 0: (A, A) [2019-05-01 08:10] -> (C, C) [2019-05-01 08:15]
   0: A       A...............................................                               d: 01.05 08:10 [01.05 10:10]  [{name=0, day=2019-05-01, id=T0, src=0}]
   1: B       B............................................... a: 01.05 08:12 [01.05 10:12]  d: 01.05 08:12 [01.05 10:12]  [{name=0, day=2019-05-01, id=T0, src=0}]
   2: C       C............................................... a: 01.05 08:15 [01.05 10:15]
leg 1: (C, C) [2019-05-01 08:15] -> (C, C) [2019-05-01 08:17]
  FOOTPATH (duration=2)
leg 2: (C, C) [2019-05-01 08:17] -> (E, E) [2019-05-01 08:40]
   0: C       C...............................................                               d: 01.05 08:17 [01.05 10:17]  [{name=1, day=2019-05-01, id=T1, src=0}]
   1: D       D............................................... a: 01.05 08:33 [01.05 10:33]  d: 01.05 08:33 [01.05 10:33]  [{name=1, day=2019-05-01, id=T1, src=0}]
   2: E       E............................................... a: 01.05 08:40 [01.05 10:40]
)"sv;

  auto tt = load_timetable(test_files_1);

  auto const A = loc_idx(tt, "A");
  auto const B = loc_idx(tt, "B");
  auto const C = loc_idx(tt, "C");
  auto const D = loc_idx(tt, "D");
  auto const E = loc_idx(tt, "E");

  tt.locations_.footpaths_out_[0U][B].push_back(footpath{D, 1_minutes});

  auto const q = routing::query{.prf_idx_ = 0U,
                                .via_stops_ = {{loc_idx(tt, "A"), 0_minutes},
                                               {loc_idx(tt, "C"), 0_minutes}}};

  auto journey = routing::journey{};
  journey.transfers_ = 1U;
  journey.start_time_ = test::at("10:10");
  journey.dest_time_ = test::at("10:40");

  // A -> B -> C
  journey.add(routing::journey::leg{
      direction::kForward, A, C, test::at("10:10"), test::at("10:15"),
      routing::journey::run_enter_exit{
          {
              .t_ = {transport_idx_t{0U}, day_idx_t{5U}},
              .stop_range_ = interval<stop_idx_t>{0U, 3U},
          },
          0,
          2}});

  // Transfer C
  journey.add(routing::journey::leg{direction::kForward, C, C,
                                    test::at("10:15"), test::at("10:17"),
                                    footpath{C, 2_minutes}});

  // C -> D -> E
  journey.add(routing::journey::leg{
      direction::kForward, C, E, test::at("10:17"), test::at("10:40"),
      routing::journey::run_enter_exit{
          {
              .t_ = {transport_idx_t{1U}, day_idx_t{5U}},
              .stop_range_ = interval<stop_idx_t>{0U, 3U},
          },
          0,
          2}});

  optimize_footpaths(tt, nullptr, q, journey);
  EXPECT_EQ(expected_optimize_transfers, print_journey(tt, journey));
}

// Stops far apart: no walks between them, only the walks added below.
std::string test_files_2() {
  return test::feed(
      {{"A", 0.0, 0.0},
       {"B", 1.0, 0.0},
       {"C", 2.0, 0.0},
       {"G", 3.0, 0.0},
       {"D", 4.0, 0.0},
       {"E", 5.0, 0.0},
       {"F", 6.0, 0.0},
       {"S", 7.0, 0.0},
       {"Z", 8.0, 0.0}},
      {{"T0",
        "R0",
        {{"A", "10:00"}, {"B", "10:10"}, {"C", "10:20"}, {"G", "10:30"}}},
       {"T1", "R1", {{"D", "10:15"}, {"E", "10:25"}, {"F", "10:40"}}}},
      "C,E,2,300,,,,\n");
}

// A 3 min walk u <-> v.
timetable load_with_walk(std::string_view const u, std::string_view const v) {
  return test::load_feeds_with_walks({test_files_2()}, {{u, v, 3}});
}

routing::journey::run_enter_exit ride(std::uint32_t const t,
                                      stop_idx_t const from,
                                      stop_idx_t const to) {
  return {{.t_ = {transport_idx_t{t}, day_idx_t{5U}},
           .stop_range_ = interval<stop_idx_t>{from, to}},
          from,
          static_cast<stop_idx_t>(to - 1U)};
}

TEST(routing, optimize_initial_start_footpath_walks_to_earlier_stop) {
  auto const tt = load_with_walk("S", "B");
  auto const S = loc_idx(tt, "S");
  auto const B = loc_idx(tt, "B");
  auto const C = loc_idx(tt, "C");
  auto const G = loc_idx(tt, "G");

  auto j = routing::journey{};
  j.start_time_ = test::at("10:00");
  j.dest_time_ = test::at("10:30");
  j.add(routing::journey::leg{direction::kForward, S, C, test::at("10:00"),
                              test::at("10:20"), footpath{C, 20_minutes}});
  j.add(routing::journey::leg{direction::kForward, C, G, test::at("10:20"),
                              test::at("10:30"), ride(0U, 2U, 4U)});

  optimize_footpaths(tt, nullptr, routing::query{}, j);

  ASSERT_EQ(2U, j.legs_.size());
  EXPECT_EQ(B, j.legs_[0].to_);
  EXPECT_EQ(3_minutes, get<footpath>(j.legs_[0].uses_).duration());
  EXPECT_EQ(test::at("10:07"), j.legs_[0].dep_time_);
}

TEST(routing, optimize_final_egress_footpath_walks_from_later_stop) {
  auto const tt = load_with_walk("B", "Z");
  auto const A = loc_idx(tt, "A");
  auto const B = loc_idx(tt, "B");
  auto const G = loc_idx(tt, "G");
  auto const Z = loc_idx(tt, "Z");

  auto j = routing::journey{};
  j.start_time_ = test::at("10:00");
  j.dest_time_ = test::at("10:50");
  j.add(routing::journey::leg{direction::kForward, A, G, test::at("10:00"),
                              test::at("10:30"), ride(0U, 0U, 4U)});
  j.add(routing::journey::leg{direction::kForward, G, Z, test::at("10:30"),
                              test::at("10:50"), footpath{Z, 20_minutes}});

  optimize_footpaths(tt, nullptr, routing::query{}, j);

  ASSERT_EQ(2U, j.legs_.size());
  EXPECT_EQ(B, j.legs_[1].from_);
  EXPECT_EQ(3_minutes, get<footpath>(j.legs_[1].uses_).duration());
  EXPECT_EQ(test::at("10:13"), j.legs_[1].arr_time_);
}

TEST(routing, optimize_transfers_takes_walk_over_rule_transfer) {
  auto const tt = load_with_walk("B", "D");
  auto const A = loc_idx(tt, "A");
  auto const B = loc_idx(tt, "B");
  auto const C = loc_idx(tt, "C");
  auto const D = loc_idx(tt, "D");
  auto const E = loc_idx(tt, "E");
  auto const F = loc_idx(tt, "F");

  // The walk fits between B (10:10) and D (10:15), and it beats the rule
  // transfer C -> E.
  auto j = routing::journey{};
  j.start_time_ = test::at("10:00");
  j.dest_time_ = test::at("10:40");
  j.transfers_ = 1U;
  j.add(routing::journey::leg{direction::kForward, A, C, test::at("10:00"),
                              test::at("10:20"), ride(0U, 0U, 3U)});
  j.add(routing::journey::leg{direction::kForward, C, E, test::at("10:20"),
                              test::at("10:25"), footpath{E, 5_minutes}});
  j.add(routing::journey::leg{direction::kForward, E, F, test::at("10:25"),
                              test::at("10:40"), ride(1U, 1U, 3U)});

  optimize_footpaths(tt, nullptr, routing::query{}, j);

  ASSERT_EQ(3U, j.legs_.size());
  EXPECT_EQ(B, j.legs_[1].from_);
  EXPECT_EQ(D, j.legs_[1].to_);
  EXPECT_EQ(3_minutes, get<footpath>(j.legs_[1].uses_).duration());
}

// T1 is entered at X, the stop T0 arrives at (15 min buffer). The change at
// station P (P1 -> P2, 8 min buffer) is no better: both stations are equally
// important, so X stays.
TEST(routing, optimize_transfers_keeps_station_bonus_of_same_stop_change) {
  auto const tt = test::load_feeds({test::feed(
      {{"A", 59.0, 10.0},
       {"P", 59.1, 10.0, "", true},
       {"P1", 59.1001, 10.0, "P"},
       {"P2", 59.1002, 10.0, "P"},
       {"X", 59.2, 10.0},
       {"E", 59.3, 10.0}},
      {{"T0", "R0", {{"A", "10:00"}, {"P1", "10:10"}, {"X", "10:20"}}},
       {"T1", "R1", {{"P2", "10:20"}, {"X", "10:37"}, {"E", "10:45"}}}},
      "")});
  auto const res = test::search_at(tt, "A", "E", "10:00");
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(loc_idx(tt, "X"), begin(res)->legs_.front().to_);
}

// T0 and T1 run in opposite directions K -> D -> L and L -> D -> K. The change
// D1 -> D2 is the shorter walk, but it passes K twice: change at K.
TEST(routing, optimize_transfers_no_double_visit_opposite_directions) {
  auto const tt = test::load_feeds_with_walks(
      {test::feed(
          {{"W", 50.0, 6.0},
           {"K1", 50.1, 6.0},
           {"K2", 50.10045, 6.0},  // 50m
           {"D1", 50.2, 6.0},
           {"D2", 50.2001, 6.0},
           {"L", 50.3, 6.0},
           {"Z", 50.4, 6.0}},
          {{"T0",
            "R0",
            {{"W", "10:00"}, {"K1", "10:02"}, {"D1", "10:03"}, {"L", "10:25"}}},
           {"T1",
            "R1",
            {{"L", "10:00"},
             {"D2", "10:08"},
             {"K2", "10:09"},
             {"Z", "10:20"}}}},
          "")},
      {{"K1", "K2", 3}, {"D1", "D2", 1}});
  auto const res = test::search_at(tt, "W", "Z", "10:00");
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(loc_idx(tt, "K1"), begin(res)->legs_.front().to_);
}

// The stops of a big station O and a small station K.
std::vector<test::stop_def> big_and_small_station() {
  return {{"R", 50.0, 6.0},
          {"K", 50.1, 6.0},
          {"O", 50.2, 6.0},
          {"H", 50.3, 6.0},
          {"Y", 50.4, 6.0}};
}

std::vector<test::trip_def> big_station_trips() {
  return {{"X0", "RX", {{"O", "08:00"}, {"Y", "08:30"}}},
          {"X1", "RX", {{"O", "08:10"}, {"Y", "08:40"}}},
          {"X2", "RX", {{"O", "08:20"}, {"Y", "08:50"}}}};
}

// T0 runs R -> K -> O, T1 back O -> K -> H: change at K, not at the bigger O.
TEST(routing, optimize_transfers_no_double_visit_for_bigger_station) {
  auto trips = big_station_trips();
  trips.push_back(
      {"T0", "R0", {{"R", "10:00"}, {"K", "10:30"}, {"O", "11:00"}}});
  trips.push_back(
      {"T1", "R1", {{"O", "12:00"}, {"K", "12:30"}, {"H", "13:00"}}});
  auto const tt =
      test::load_feeds({test::feed(big_and_small_station(), trips, "")});
  auto const res = test::search_at(tt, "R", "H", "10:00");
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(loc_idx(tt, "K"), begin(res)->legs_.front().to_);
}

// T0 and T1 both run K -> O: the bigger station O is still preferred.
TEST(routing, optimize_transfers_same_direction_prefers_bigger_station) {
  auto trips = big_station_trips();
  trips.push_back(
      {"T0", "R0", {{"R", "10:00"}, {"K", "10:30"}, {"O", "11:00"}}});
  trips.push_back(
      {"T1", "R1", {{"K", "10:40"}, {"O", "11:10"}, {"H", "11:40"}}});
  auto const tt =
      test::load_feeds({test::feed(big_and_small_station(), trips, "")});
  auto const res = test::search_at(tt, "R", "H", "10:00");
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(loc_idx(tt, "O"), begin(res)->legs_.front().to_);
}

// T1 loops S1 -> L -> S2 through station S. Boarding at S1 is the shorter
// change, but T1 then passes S again: board at S2.
TEST(routing, optimize_transfers_no_double_visit_loop) {
  auto const tt = test::load_feeds_with_walks(
      {test::feed({{"R", 50.0, 6.0},
                   {"S", 50.1, 6.0, "", true},
                   {"S1", 50.1001, 6.0, "S"},
                   {"S2", 50.1002, 6.0, "S"},
                   {"L", 50.2, 6.0},
                   {"Z", 50.3, 6.0}},
                  {{"T0", "R0", {{"R", "10:00"}, {"S1", "10:10"}}},
                   {"T1",
                    "R1",
                    {{"S1", "10:30"},
                     {"L", "10:35"},
                     {"S2", "10:40"},
                     {"Z", "10:50"}}}},
                  "")},
      {{"S1", "S2", 3}});
  auto const res = test::search_at(tt, "R", "Z", "10:00");
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(loc_idx(tt, "S2"), begin(res)->legs_.back().from_);
}

// T1 loops H -> B -> K2 -> S. The change T1 -> T2 moves from B to S, so T1
// passes station K again where T0 was left: board T1 at K2 instead of H.
TEST(routing, optimize_transfers_no_double_visit_over_two_changes) {
  auto const tt = test::load_feeds_with_walks(
      {test::feed(
          {{"A", 50.0, 6.0},
           {"K", 50.1, 6.0, "", true},
           {"K1", 50.1001, 6.0, "K"},
           {"K2", 50.1002, 6.0, "K"},
           {"H", 50.2, 6.0},
           {"B", 50.3, 6.0},
           {"S", 50.4, 6.0},
           {"Z", 50.5, 6.0}},
          {{"T0", "R0", {{"A", "04:30"}, {"K1", "04:59"}}},
           {"T1",
            "R1",
            {{"H", "05:14"}, {"B", "05:15"}, {"K2", "05:17"}, {"S", "05:20"}}},
           {"T2", "R2", {{"S", "05:39"}, {"Z", "05:58"}}}},
          "")},
      {{"K1", "K2", 4}, {"K1", "H", 9}, {"B", "S", 3}});
  auto const res = test::search_at(tt, "A", "Z", "04:30");
  ASSERT_EQ(1U, res.size());
  auto const& legs = begin(res)->legs_;
  ASSERT_EQ(5U, legs.size());
  EXPECT_EQ(loc_idx(tt, "K2"), legs[2].from_);
  EXPECT_EQ(loc_idx(tt, "S"), legs[2].to_);
}
