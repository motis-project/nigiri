#include "gtest/gtest.h"

#include <optional>
#include <string>
#include <vector>

#include "gtfsrt/gtfs-realtime.pb.h"

#include "utl/helpers/algorithm.h"

#include "nigiri/routing/query.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

#include "../raptor_search.h"
#include "../transfer_rules_util.h"
#include "./transfer_rules_rt_util.h"

// Real-time updates with transfers.txt rules: real-time virtual locations in
// interval searches, lower bounds, start offsets, reverts and rule precedence.

using namespace nigiri;
using nigiri::test::at;
using nigiri::test::kDay;
using nigiri::test::raptor_search;
using nigiri::test::station_query;
using nigiri::test::trip_update;
using nigiri::test::update;

// test::network() (S: station with platforms S1..S4, X: station with platforms
// X1, X2), plus:
//   T0  (R0): A0 09:30 -> A 09:50       feeder to F
//   DAB (RD): A0 10:00 -> B 10:45       direct, no change
//   XB  (RX): X1 10:31 -> B 10:40       leaves X right after F arrives at S
//   ST1 (RS1): A2 10:00 -> S1 10:30     first half of a stay-seated chain
//   ST2 (RS2): S1 10:30 -> C 11:00      second half
//   Z: a stop about 180 m north of S4, not walkable except through rules
test::network_rows const kRows{
    .stops_ = {{"A0", 50.0, 5.9}, {"Z", 50.0020, 6.5}},
    .trips_ = {{"T0", "R0", {{"A0", "09:30"}, {"A", "09:50"}}},
               {"DAB", "RD", {{"A0", "10:00"}, {"B", "10:45"}}},
               {"XB", "RX", {{"X1", "10:31"}, {"B", "10:40"}}},
               {"ST1", "RS1", {{"A2", "10:00"}, {"S1", "10:30"}}},
               {"ST2", "RS2", {{"S1", "10:30"}, {"C", "11:00"}}}}};

// Two real-time virtual locations exist (F moved to S3, G moved to S4). An
// interval search from S puts both into its start set and must not throw. The
// journey starts where the moved G departs now (S4, 10:40).
TEST(rt_transfer_rules, interval_search_with_two_real_time_virtual_locations) {
  auto const tt = test::load_network("S,S,2,120,,,,\nS,S,2,900,,,F,G", kRows);
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt,
         {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}},
          {"G", {{.seq_ = 1U, .stop_id_ = "S2", .assigned_ = "S4"}}}});
  ASSERT_GE(rtt.n_rt_locations(), 2U) << "precondition";

  auto res = pareto_set<routing::journey>{};
  EXPECT_NO_THROW(
      res = raptor_search(
          tt, &rtt,
          station_query(tt, "S", "B", interval{at("10:35"), at("10:50")}),
          direction::kForward));
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("10:40"), begin(res)->start_time_);
  EXPECT_EQ(at("11:00"), begin(res)->dest_time_);
}

// The timed rule S4 -> X1 for route RF only binds F once F is moved to S4 -
// then F's real-time virtual location has the rule's 0 min transfer to
// X1, and T0 + F + XB arrives at B 10:40, before the direct DAB (10:45) but
// with two changes. When F arrives (round 2), DAB's 10:45 is already known:
// the lower bound of F's location must know that transfer, or the arrival is
// pruned.
TEST(rt_transfer_rules, lower_bound_knows_real_time_rule_edges) {
  auto const tt = test::load_network("S4,X1,1,,RF,,,", kRows);
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S4"}}}});
  ASSERT_EQ(1U, rtt.n_rt_locations()) << "precondition";

  // Control: from A, F is the first vehicle and nothing is known about B yet
  // when it arrives - the rule's transfer takes F + XB to B at 10:40.
  auto const control = raptor_search(
      tt, &rtt, station_query(tt, "A", "B", at("10:00")), direction::kForward);
  ASSERT_TRUE(utl::any_of(control, [](routing::journey const& j) {
    return j.dest_time_ == at("10:40");
  })) << "precondition: the real-time rule edge works";

  auto const res = raptor_search(
      tt, &rtt, station_query(tt, "A0", "B", at("09:30")), direction::kForward);
  EXPECT_TRUE(utl::any_of(res, [](routing::journey const& j) {
    return j.dest_time_ == at("10:40");
  }));
}

// Z -> S4 is a 5 min rule, Z -> S4 for the trip G a 10 min one. Once G and GL
// are moved to S4, a start at Z 10:32 cannot make G (10:40) but makes GL
// (11:10): the start offset of G's real-time virtual location is the trip
// rule's, not its base location's. With the base location's offset, the
// search would find a journey it cannot reconstruct and drop it - and GL,
// which it dominated, with it.
TEST(rt_transfer_rules, start_offset_of_real_time_location_honours_rule) {
  auto const tt = test::load_network("Z,S4,2,300,,,,\nZ,S4,2,600,,,,G", kRows);
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt,
         {{"G", {{.seq_ = 1U, .stop_id_ = "S2", .assigned_ = "S4"}}},
          {"GL", {{.seq_ = 1U, .stop_id_ = "S2", .assigned_ = "S4"}}}});
  ASSERT_EQ(1U, rtt.n_rt_locations()) << "precondition";

  auto q = station_query(tt, "Z", "B", at("10:32"));
  q.start_match_mode_ = routing::location_match_mode::kExact;
  q.use_start_footpaths_ = true;
  auto const res = raptor_search(tt, &rtt, std::move(q), direction::kForward);
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("11:30"), begin(res)->dest_time_);
}

// ST1 -> ST2 stay seated at S1; a rule names ST2 there. Moving the handover
// stop to S3 and taking it back restores the schedule, and a repeated revert
// message changes nothing.
TEST(rt_transfer_rules, revert_at_stay_seated_handover_stop_restores_schedule) {
  auto const tt = test::load_network(
      "S1,S1,4,,,,ST1,ST2\nS,S,2,120,,,,\nS,S,2,900,,,F,ST2", kRows);
  auto rtt = rt::create_rt_timetable(tt, kDay);
  auto n_location_changes = 0U;
  rtt.set_change_callback([&](transport, stop_idx_t, event_type,
                              std::optional<location_idx_t> const l,
                              std::optional<bool>, std::optional<duration_t>) {
    n_location_changes += l.has_value() ? 1U : 0U;
  });

  auto const handover_location = [&]() {
    auto const [r, _] = test::resolve(tt, rtt, "ST2");
    return r.valid() ? rt::frun{tt, &rtt, r}[0U].get_stop().location_idx()
                     : location_idx_t::invalid();
  };
  auto const scheduled = handover_location();
  ASSERT_NE(location_idx_t::invalid(), scheduled) << "precondition";

  update(tt, rtt,
         {{"ST2", {{.seq_ = 1U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  ASSERT_NE(0U, n_location_changes) << "precondition: the move applied";

  auto const revert =
      trip_update{"ST2", {{.seq_ = 1U, .stop_id_ = "S1", .arr_delay_ = 0}}};
  update(tt, rtt, {revert});
  EXPECT_EQ(scheduled, handover_location());

  n_location_changes = 0U;
  update(tt, rtt, {revert});
  EXPECT_EQ(0U, n_location_changes);
}

// S1 is the stay-seated handover stop of ST1 -> ST2: one vehicle, it arrives as
// ST1 and departs as ST2. Getting off there is ST1's rule (ST1 -> G: 5 min),
// not ST2's (ST2 -> G: 1 min) - nobody gets off ST2 at its first stop. Moved to
// S3 and 6 min late (10:36), the passenger of ST1 misses G (10:40) and takes
// GL (B 11:30). The real-time key at the handover stop must not let ST2's rule
// decide.
TEST(rt_transfer_rules,
     track_change_at_handover_stop_keeps_rule_of_arriving_trip) {
  auto const tt = test::load_network(
      "S1,S1,4,,,,ST1,ST2\nS,S,2,120,,,,\nS,S,2,300,,,ST1,G\n"
      "S,S,2,60,,,ST2,G",
      kRows);
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(
      tt, rtt,
      {{"ST1",
        {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3", .arr_delay_ = 6}}}});
  auto const res = raptor_search(
      tt, &rtt, station_query(tt, "A2", "B", at("10:00")), direction::kForward);
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("11:30"), begin(res)->dest_time_);
}

// F is named by a trip rule (15 min, one trip), F0 and F2 only by the route
// rules (RF: 15 min, RF -> RG: 3 min). For F -> G the trip rule is the most
// specific one. Moved to S3, F lands on a location that states the same
// values as F0's - but there the route pair is the most specific rule
// (3 min), so the key must keep the specificity of F's rules.
TEST(rt_transfer_rules, real_time_key_keeps_specificity) {
  auto const tt = test::load_network(
      "S,S,2,120,,,,\nS,S,2,900,,,F,\nS,S,2,900,RF,,,\nS,S,2,180,RF,RG,,",
      kRows);
  auto rtt = rt::create_rt_timetable(tt, kDay);
  auto const before = raptor_search(
      tt, &rtt, station_query(tt, "A", "B", at("10:00")), direction::kForward);
  ASSERT_EQ(1U, before.size());
  ASSERT_EQ(at("11:30"), begin(before)->dest_time_)
      << "precondition: F -> G takes 15 min in the schedule";

  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  auto const after = raptor_search(
      tt, &rtt, station_query(tt, "A", "B", at("10:00")), direction::kForward);
  ASSERT_EQ(1U, after.size());
  EXPECT_EQ(at("11:30"), begin(after)->dest_time_);
}
