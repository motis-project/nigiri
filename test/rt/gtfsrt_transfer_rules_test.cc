#include "gtest/gtest.h"

#include <chrono>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gtfsrt/gtfs-realtime.pb.h"

#include "utl/helpers/algorithm.h"

#include "nigiri/loader/build_footpaths.h"
#include "nigiri/routing/direct.h"
#include "nigiri/routing/leg_alternatives.h"
#include "nigiri/routing/one_to_all.h"
#include "nigiri/routing/query.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/rt/rt_transfer_rules.h"
#include "nigiri/timetable.h"

#include "../raptor_search.h"
#include "../transfer_rules_util.h"
#include "./util.h"

// Real-time updates in the presence of transfers.txt rules, on the network
// test::network() (transfer_rules_util.h). Every test states its own
// transfers.txt rows and real-time messages.

using namespace nigiri;
using nigiri::test::add_empty_profile;
using nigiri::test::arrival;
using nigiri::test::at;
using nigiri::test::g_from_f;
using nigiri::test::gl_from_f;
using nigiri::test::kDay;
using nigiri::test::lidx;
using nigiri::test::load_network;
using nigiri::test::raptor_search;
using nigiri::test::station_query;
using nigiri::test::trip_update;

// GTFS-RT messages for the transfers.txt rule tests: trip updates of kDay,
// source 0.

// The run of trip_id on kDay (source 0) and its trip; the run is invalid if it
// does not run.
std::pair<rt::run, trip_idx_t> resolve(timetable const& tt,
                                       rt_timetable const& rtt,
                                       std::string const& trip_id) {
  auto td = transit_realtime::TripDescriptor{};
  td.set_trip_id(trip_id);
  td.set_start_date("20190501");
  return rt::gtfsrt_resolve_run(kDay, tt, &rtt, source_idx_t{0}, td);
}

// Applies the updates; the transfer rules and the lower bounds follow, as
// motis does after every update.
void update(timetable const& tt,
            rt_timetable& rtt,
            std::vector<trip_update> const& updates) {
  rt::gtfsrt_update_msg(
      tt, rtt, source_idx_t{0}, "",
      test::to_msg(updates, kDay + std::chrono::hours{8}, "20190501"));
  rt::resolve_conflicts(tt, rtt);
  rtt.update_lbs(tt);
}

// Further stops and trips for the tests that need them:
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

// The id of the base location of l, as seen outside the routing.
std::string leg_stop_id(timetable const& tt, location_idx_t const l) {
  if (l >= tt.n_locations()) {
    return "out-of-range";
  }
  return std::string{tt.locations_.ids_[tt.base(l)].view()};
}

// The base location a trip stops at, as seen outside the routing.
std::string stop_id_at(timetable const& tt,
                       rt_timetable const& rtt,
                       std::string const& trip_id,
                       stop_idx_t const stop_idx) {
  auto const [r, _] = resolve(tt, rtt, trip_id);
  if (!r.valid()) {
    return "?";
  }
  return leg_stop_id(tt, rt::frun{tt, &rtt, r}[stop_idx].get_location_idx());
}

constexpr auto const kAtoB = std::pair{"A", "B"};
constexpr auto const kAtoC = std::pair{"A", "C"};

pareto_set<routing::journey> search(
    timetable const& tt,
    rt_timetable const& rtt,
    std::pair<char const*, char const*> const& od,
    std::string_view const hhmm,
    direction const dir = direction::kForward) {
  return raptor_search(tt, &rtt,
                       station_query(tt, od.first, od.second, at(hhmm)), dir);
}

std::vector<unixtime_t> direct_arrivals(
    timetable const& tt,
    rt_timetable const& rtt,
    std::pair<char const*, char const*> const& od,
    std::string_view const from,
    std::string_view const to) {
  auto q = station_query(tt, od.first, od.second, at(from));
  q.slow_direct_ = true;
  q.use_start_footpaths_ = true;
  auto res = pareto_set<routing::journey>{};
  routing::enrich_with_slow_direct<direction::kForward>(
      tt, &rtt, q, interval{at(from), at(to)}, res);
  auto arrivals = std::vector<unixtime_t>{};
  for (auto const& j : res) {
    arrivals.push_back(j.dest_time_);
  }
  return arrivals;
}

// The name of a parameterized case.
template <typename T>
std::string case_name(testing::TestParamInfo<T> const& info) {
  return std::string{info.param.name_};
}

// ===========================================================================
// 1. Delays where a rule decides a change. No track change involved: the rule
//    has to keep applying to the real-time times.
// ===========================================================================

// Trips are late; the arrival is checked before (if before_ is set) and after
// the delays, and so are the transports and locations real time adds (if
// set).
struct delay {
  std::string_view name_;
  std::string_view rules_;
  std::vector<trip_update> updates_;
  std::pair<char const*, char const*> od_;
  unixtime_t (*before_)() = nullptr;
  unixtime_t (*after_)() = nullptr;
  std::optional<std::uint32_t> n_rt_transports_{};
  std::optional<std::uint32_t> n_rt_locations_{};
};

std::ostream& operator<<(std::ostream& out, delay const& c) {
  return out << c.name_;
}

class gtfsrt_delay : public testing::TestWithParam<delay> {};

TEST_P(gtfsrt_delay, arrival) {
  auto const& c = GetParam();
  auto const tt = load_network(c.rules_);
  auto rtt = rt::create_rt_timetable(tt, kDay);
  if (c.before_ != nullptr) {
    EXPECT_EQ(c.before_(), arrival(tt, &rtt, c.od_));
  }
  update(tt, rtt, c.updates_);
  EXPECT_EQ(c.after_(), arrival(tt, &rtt, c.od_));
  if (c.n_rt_transports_.has_value()) {
    EXPECT_EQ(*c.n_rt_transports_, rtt.n_rt_transports());
  }
  if (c.n_rt_locations_.has_value()) {
    EXPECT_EQ(*c.n_rt_locations_, rtt.n_rt_locations());
  }
}

INSTANTIATE_TEST_SUITE_P(
    delays,
    gtfsrt_delay,
    testing::Values(
        // A slower rule that the schedule satisfies (10 min >= 8 min) becomes
        // binding once the feeder is late: 5 min < 8 min although the walk
        // alone would do. Only the trips whose change the rule decides move:
        // F (late) and G. GL, H, HL and F2 stay as scheduled.
        delay{"delay_makes_slower_rule_binding",
              "S,S,2,120,,,,\n"
              "S,S,2,480,,,F,G\n"
              "S,S,2,600,RF,RG,,",
              {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .arr_delay_ = 5}}}},
              kAtoB,
              g_from_f,
              gl_from_f,
              2U},
        // A delay that leaves every rule decision as it was moves nothing:
        // F -> G still has 9 min >= 8 min.
        delay{"delay_within_rule_moves_nothing",
              "S,S,2,120,,,,\nS,S,2,480,,,F,G",
              {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .arr_delay_ = 1}}}},
              kAtoB,
              nullptr,
              g_from_f,
              1U,
              0U},
        // A faster rule can become binding as well: F2 leaves S1 6 min after
        // F arrives, S1 takes 5 min, the RF trips among themselves 0 min. F 3
        // min late: only the rule keeps F2.
        delay{"delay_makes_faster_rule_binding",
              "S1,S1,2,300,,,,\nS1,S1,2,0,RF,RF,,",
              {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .arr_delay_ = 3}}}},
              std::pair{"A", "D"},
              [] { return at("11:06"); },
              [] { return at("11:06"); }},
        // A faster rule (0 min, route pair) keeps a connection the default
        // change time (2 min) would break - also after both trips moved: F
        // arrives 10:33, H leaves 10:33 and reaches C two minutes late.
        delay{"delay_keeps_faster_rule",
              "S1,S1,2,120,,,,\nS1,S1,2,0,RF,RH,,",
              {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .arr_delay_ = 3}}},
               {"H", {{.seq_ = 1U, .stop_id_ = "S1", .dep_delay_ = 2}}}},
              kAtoC,
              g_from_f,
              [] { return at("11:02"); }},
        // A forbidden trip pair stays forbidden however late the connecting
        // trip is.
        delay{"delay_keeps_forbidden_pair",
              "S,S,2,120,,,,\nS,S,3,,,,F,G",
              {{"G", {{.seq_ = 1U, .stop_id_ = "S2", .dep_delay_ = 5}}}},
              kAtoB,
              nullptr,
              gl_from_f},
        // Early events change rule decisions as well: G leaves 5 min early,
        // F -> G has 5 min < 8 min ...
        delay{"early_departure_makes_slower_rule_binding",
              "S,S,2,120,,,,\nS,S,2,480,,,F,G",
              {{"G", {{.seq_ = 1U, .stop_id_ = "S2", .dep_delay_ = -5}}}},
              kAtoB,
              g_from_f,
              gl_from_f},
        // ... F arrives 5 min early, F -> H has 6 min < 8 min ...
        delay{"early_arrival_makes_slower_rule_binding",
              "S,S,2,120,,,,\nS,S,2,480,,,F,H",
              {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .arr_delay_ = -5}}}},
              kAtoC,
              gl_from_f,
              gl_from_f},
        // ... F2 leaves 3 min early: F -> F2 has 3 min < S1's 5 min, the RF ->
        // RF rule (0 min) allows it.
        delay{"early_departure_makes_faster_rule_binding",
              "S1,S1,2,300,,,,\nS1,S1,2,0,RF,RF,,",
              {{"F2", {{.seq_ = 1U, .stop_id_ = "S1", .dep_delay_ = -3}}}},
              std::pair{"A", "D"},
              [] { return at("11:06"); },
              [] { return at("11:03"); }},
        // A timed rule (0 min) decides changes too: F 5 min late, the 1 min
        // gap to F2 is below S1's 2 min, the RF -> RF guarantee keeps F2.
        delay{"delay_makes_timed_rule_binding",
              "S1,S1,1,,RF,RF,,",
              {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .arr_delay_ = 5}}}},
              std::pair{"A", "D"},
              [] { return at("11:06"); },
              [] { return at("11:06"); }}),
    case_name<delay>);

TEST(gtfsrt_transfer_rules, delay_moves_both_visits_of_partner) {
  auto const tt = load_network(
      "S,S,2,120,,,,\n"
      "S,S,2,480,,,F,GO",
      {.stops_ = {{"Y", 50.2, 6.5}},
       .trips_ = {{"GO",
                   "RO",
                   {{"S2", "10:40"},
                    {"Y", "10:41"},
                    {"S2", "10:42"},
                    {"B", "10:55"}}}}});
  auto rtt = rt::create_rt_timetable(tt, kDay);
  EXPECT_EQ(at("10:55"), arrival(tt, &rtt, kAtoB));
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .arr_delay_ = 5}}}});
  EXPECT_EQ(g_from_f(), arrival(tt, &rtt, kAtoB));
  EXPECT_EQ(2U, rtt.n_rt_transports());
}

TEST(gtfsrt_transfer_rules, track_change_of_one_visit_keeps_rule) {
  auto const tt = load_network(
      "S,S,2,120,,,,\n"
      "S,S,2,900,,,F,GO\n"
      "S,S,2,900,,,FO,GQ",
      {.stops_ = {{"Y", 50.2, 6.5},
                  {"Z", 50.3, 6.5},
                  {"A3", 50.0, 5.5},
                  {"E", 50.0, 9.0}},
       .trips_ = {
           {"GO",
            "RO",
            {{"S2", "10:40"}, {"Y", "10:41"}, {"S2", "10:42"}, {"B", "10:55"}}},
           {"FO",
            "RFO",
            {{"A3", "10:00"},
             {"S1", "10:28"},
             {"Z", "10:29"},
             {"S1", "10:32"}}},
           {"GQ", "RGQ", {{"S2", "10:40"}, {"E", "11:00"}}},
           {"GQL", "RGQ", {{"S2", "11:10"}, {"E", "11:30"}}}}});
  for (auto const& [trip, moved, other, scheduled, od, expected] :
       {std::tuple{"GO", stop_idx_t{0U}, stop_idx_t{2U}, "S2", kAtoB, "11:00"},
        std::tuple{"GO", stop_idx_t{2U}, stop_idx_t{0U}, "S2", kAtoB, "11:00"},
        std::tuple{"FO", stop_idx_t{1U}, stop_idx_t{3U}, "S1",
                   std::pair{"A3", "E"}, "11:30"},
        std::tuple{"FO", stop_idx_t{3U}, stop_idx_t{1U}, "S1",
                   std::pair{"A3", "E"}, "11:30"}}) {
    auto rtt = rt::create_rt_timetable(tt, kDay);
    update(
        tt, rtt,
        {{trip,
          {{.seq_ = moved + 1U, .stop_id_ = scheduled, .assigned_ = "S3"}}}});
    EXPECT_EQ("S3", stop_id_at(tt, rtt, trip, moved));
    EXPECT_EQ(scheduled, stop_id_at(tt, rtt, trip, other));
    EXPECT_EQ(at(expected), arrival(tt, &rtt, od));
  }
}

std::vector<unixtime_t> alternative_arrivals(
    timetable const& tt,
    rt_timetable const& rtt,
    std::pair<char const*, char const*> const od) {
  auto const q = station_query(tt, od.first, od.second, at("10:00"));
  auto const res = raptor_search(tt, &rtt, q);
  if (res.size() != 1U) {
    return {};
  }
  auto const& j = *begin(res);
  auto last_transit = j.legs_.size();
  for (auto i = 0U; i != j.legs_.size(); ++i) {
    if (test::is_transit(j.legs_[i])) {
      last_transit = i;
    }
  }
  auto arrivals = std::vector<unixtime_t>{};
  for (auto const& a :
       routing::get_leg_alternatives(tt, &rtt, q, j, last_transit, 5U)) {
    arrivals.push_back(a.dest_time_);
  }
  return arrivals;
}

TEST(gtfsrt_transfer_rules, leg_alternatives_from_real_time_location) {
  auto const tt = load_network("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  EXPECT_EQ(std::vector{at("13:20")}, alternative_arrivals(tt, rtt, kAtoB));
}

TEST(gtfsrt_transfer_rules, leg_alternatives_to_moved_trip) {
  auto const tt = load_network("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"G", {{.seq_ = 1U, .stop_id_ = "S2", .assigned_ = "S4"}}}});
  EXPECT_EQ(std::vector{at("13:20")}, alternative_arrivals(tt, rtt, kAtoB));
}

TEST(gtfsrt_transfer_rules, leg_alternatives_after_delayed_feeder) {
  auto const tt = load_network("S,S,2,120,,,,\nS,S,2,480,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .arr_delay_ = 5}}}});
  EXPECT_EQ(std::vector{at("13:20")}, alternative_arrivals(tt, rtt, kAtoB));
}

TEST(gtfsrt_transfer_rules, td_offset_at_stop_with_real_time_location) {
  auto const tt = load_network("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  auto const res = raptor_search(
      tt, &rtt,
      routing::query{
          .start_time_ = at("09:50"),
          .start_match_mode_ = routing::location_match_mode::kIntermodal,
          .dest_match_mode_ = routing::location_match_mode::kIntermodal,
          .start_ = {{lidx(tt, "A"), 3_minutes, 0U}},
          .td_dest_ = {{lidx(tt, "S3"),
                        {routing::td_offset{.valid_from_ = at("00:00"),
                                            .duration_ = 4_minutes,
                                            .transport_mode_ = 0U}}}}});
  EXPECT_EQ(at("10:34"), arrival(res));
}

// F moves from its scheduled platform S1 to another platform; the arrival is
// checked before (if before_ is set) and after the track change.
struct track_change {
  std::string_view name_;
  std::string_view rules_;
  std::string_view platform_;
  std::pair<char const*, char const*> od_;
  unixtime_t (*before_)() = nullptr;
  unixtime_t (*after_)() = nullptr;
  bool is_assigned_as_stop_id_ = false;
};

std::ostream& operator<<(std::ostream& out, track_change const& c) {
  return out << c.name_;
}

class gtfsrt_track_change : public testing::TestWithParam<track_change> {};

TEST_P(gtfsrt_track_change, arrival) {
  auto const& c = GetParam();
  auto const tt = load_network(c.rules_);
  auto rtt = rt::create_rt_timetable(tt, kDay);
  if (c.before_ != nullptr) {
    EXPECT_EQ(c.before_(), arrival(tt, &rtt, c.od_));
  }
  update(tt, rtt,
         {{"F",
           {{.seq_ = 2U,
             .stop_id_ = "S1",
             .assigned_ = std::string{c.platform_},
             .is_assigned_as_stop_id_ = c.is_assigned_as_stop_id_}}}});
  EXPECT_EQ(c.platform_, stop_id_at(tt, rtt, "F", 1U));
  EXPECT_EQ(c.after_(), arrival(tt, &rtt, c.od_));
}

// ===========================================================================
// 2. Track changes: rules bound to the TRIP (or its route) at station level
//    follow the trip to its new platform.
// ===========================================================================

INSTANTIATE_TEST_SUITE_P(
    trip_rules,
    gtfsrt_track_change,
    testing::Values(
        // Slower: F -> G takes 15 min, also from the new platform.
        track_change{"track_change_keeps_slower_trip_rule",
                     "S,S,2,120,,,,\nS,S,2,900,,,F,G", "S3", kAtoB, nullptr,
                     gl_from_f},
        // ... also stated the old way: a different stop_id, no
        // stop_time_properties.
        track_change{"track_change_via_stop_id_keeps_rule",
                     "S,S,2,120,,,,\nS,S,2,900,,,F,G", "S3", kAtoB, nullptr,
                     gl_from_f, true},
        // Faster: a timed transfer F -> H (H waits, 0 min) survives F moving
        // away from H's platform, where the walk alone would take longer than 1
        // min.
        track_change{"track_change_keeps_timed_trip_rule",
                     "S,S,2,120,,,,\nS,S,1,,,,F,H", "S3", kAtoC, g_from_f,
                     g_from_f},
        // Without the rule (0 min at S1 for everyone) the same track change
        // does break the connection.
        track_change{"track_change_without_rule_breaks_tight_connection",
                     "S1,S1,2,0,,,,", "S3", kAtoC, g_from_f, gl_from_f},
        // Forbidden: the trip pair stays forbidden from the new platform.
        track_change{"track_change_keeps_forbidden_trip_pair",
                     "S,S,2,120,,,,\nS,S,3,,,,F,G", "S3", kAtoB, nullptr,
                     gl_from_f},
        // Route-qualified station rule, new platform S4 never saw an RF trip.
        track_change{"track_change_keeps_route_rule_new_location",
                     "S,S,2,120,,,,\nS,S,2,900,RF,RG,,", "S4", kAtoB, nullptr,
                     gl_from_f},
        // The specific rule wins on the new platform as well: the station says
        // 15 min for RF -> RG, the trip pair F -> G is timed.
        track_change{"track_change_keeps_rule_precedence",
                     "S,S,2,120,,,,\nS,S,2,900,RF,RG,,\nS,S,1,,,,F,G", "S4",
                     kAtoB, g_from_f, g_from_f},
        // F is named by a trip rule (15 min, one trip), F0 and F2 only by the
        // route rules (RF: 15 min, RF -> RG: 3 min). For F -> G the trip rule
        // is the most specific one. Moved to S3, F lands on a location that
        // states the same values as F0's - but there the route pair is the
        // most specific rule (3 min), so the signature must keep the
        // specificity of F's rules.
        track_change{"real_time_signature_keeps_specificity",
                     "S,S,2,120,,,,\nS,S,2,900,,,F,\nS,S,2,900,RF,,,\n"
                     "S,S,2,180,RF,RG,,",
                     "S3", kAtoB, gl_from_f, gl_from_f}),
    case_name<track_change>);

// The rule also holds if its TARGET trip changes platform.
TEST(gtfsrt_transfer_rules, track_change_of_target_trip_keeps_rule) {
  auto const tt = load_network("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"G", {{.seq_ = 1U, .stop_id_ = "S2", .assigned_ = "S4"}}}});
  EXPECT_EQ("S4", stop_id_at(tt, rtt, "G", 0U));
  EXPECT_EQ(gl_from_f(), arrival(tt, &rtt, kAtoB));
  // G itself is still usable from its new platform, and a query from the
  // station finds it there.
  EXPECT_EQ(g_from_f(), arrival(tt, &rtt, std::pair{"S4", "B"}, "10:35"));
  EXPECT_EQ(g_from_f(), arrival(tt, &rtt, std::pair{"S", "B"}, "10:35"));
  EXPECT_EQ(std::vector{g_from_f()},
            direct_arrivals(tt, rtt, std::pair{"S", "B"}, "10:35", "10:45"));
}

// ... and if both change platform (two real-time virtual locations meet):
// slower (F -> G), timed (F -> H), and a faster platform rule (0 min at S3 for
// RF -> RH, and H moves there too) that only becomes binding there.
TEST(gtfsrt_transfer_rules, track_change_of_both_trips_keeps_rule) {
  struct test_case {
    char const* rules_;
    std::vector<trip_update> moves_;
    std::pair<char const*, char const*> od_;
    unixtime_t (*before_)();
    unixtime_t (*after_)();
  };
  auto const cases = std::vector<test_case>{
      {"S,S,2,120,,,,\nS,S,2,900,,,F,G",
       {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}},
        {"G", {{.seq_ = 1U, .stop_id_ = "S2", .assigned_ = "S4"}}}},
       kAtoB,
       nullptr,
       gl_from_f},
      {"S,S,2,120,,,,\nS,S,1,,,,F,H",
       {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S4"}}},
        {"H", {{.seq_ = 1U, .stop_id_ = "S1", .assigned_ = "S2"}}}},
       kAtoC,
       nullptr,
       g_from_f},
      {"S3,S3,2,120,,,,\nS3,S3,2,0,RF,RH,,",
       {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}},
        {"H", {{.seq_ = 1U, .stop_id_ = "S1", .assigned_ = "S3"}}}},
       kAtoC,
       gl_from_f,  // S1: 1 min < 2 min default
       g_from_f}};
  for (auto const& c : cases) {
    SCOPED_TRACE(c.rules_);
    auto const tt = load_network(c.rules_);
    auto rtt = rt::create_rt_timetable(tt, kDay);
    if (c.before_ != nullptr) {
      EXPECT_EQ(c.before_(), arrival(tt, &rtt, c.od_));
    }
    update(tt, rtt, c.moves_);
    for (auto const& m : c.moves_) {
      auto const& stu = m.stus_.front();
      EXPECT_EQ(*stu.assigned_,
                stop_id_at(tt, rtt, m.trip_id_,
                           static_cast<stop_idx_t>(stu.seq_ - 1U)));
    }
    EXPECT_EQ(c.after_(), arrival(tt, &rtt, c.od_));
  }

  // An interval search from S puts both F and G into its start set: the
  // journey starts where the moved G departs now (S4, 10:40).
  auto const tt = load_network(cases.front().rules_);
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, cases.front().moves_);
  ASSERT_GE(rtt.n_rt_locations(), 2U) << "precondition";
  auto const res = raptor_search(
      tt, &rtt, station_query(tt, "S", "B", interval{at("10:35"), at("10:50")}),
      direction::kForward);
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("10:40"), begin(res)->start_time_);
  EXPECT_EQ(g_from_f(), begin(res)->dest_time_);
}

// ===========================================================================
// 3. Track changes: rules bound to the PLATFORM stay with the platform.
// ===========================================================================

INSTANTIATE_TEST_SUITE_P(
    platform_rules,
    gtfsrt_track_change,
    testing::Values(
        // Leaving a slow platform: S1 -> S2 takes 15 min for F, S3 -> S2 does
        // not.
        track_change{"track_change_leaves_platform_rule_behind",
                     "S1,S2,2,120,,,,\nS1,S2,2,900,,,F,", "S3", kAtoB,
                     gl_from_f, g_from_f},
        // Entering a slow platform (trip-qualified: no virtual location at S3):
        // the rule only becomes binding through the track change.
        track_change{"track_change_makes_trip_platform_rule_binding",
                     "S3,S2,2,120,,,,\nS3,S2,2,900,,,F,", "S3", kAtoB, g_from_f,
                     gl_from_f},
        // Entering a slow platform (route-qualified: F0 already stops at a
        // virtual location of S3).
        track_change{"track_change_makes_route_platform_rule_binding",
                     "S3,S2,2,120,,,,\nS3,S2,2,900,RF,,,", "S3", kAtoB,
                     g_from_f, gl_from_f},
        // Entering a forbidden platform pair: no transfer S3 -> S2 for F at
        // all, and G and GL both leave from S2. What is left is the next
        // feeder, P, two hours later.
        track_change{"track_change_makes_forbidden_platform_rule_binding",
                     "S3,S2,2,120,,,,\nS3,S2,3,,,,F,", "S3", kAtoB, g_from_f,
                     [] { return at("13:20"); }},
        // ... also for a route pair whose departing side has no location of
        // its own in the schedule: G has to move, too. (P -> Q is banned, so
        // F -> Q is the only way to 13:20.)
        track_change{"track_change_makes_forbidden_route_pair_binding",
                     "S4,S2,3,,RF,RG,,\nS,S,3,,RP,RQ,,\nX,X,3,,RP,RQ,,", "S4",
                     kAtoB, g_from_f, [] { return at("13:20"); }},
        // An unqualified platform rule needs no virtual location: it stays
        // with S1.
        track_change{"unqualified_platform_rule", "S1,S2,2,900,,,,", "S3",
                     kAtoB, gl_from_f, g_from_f},
        // Precedence against a rule that names the new platform without
        // qualifying the trip: "anything from S4 to an RG trip at S2: 0 min"
        // names both stops exactly and beats "RF trips at the station: 15 min".
        track_change{"track_change_platform_rule_beats_station_rule",
                     "S,S,2,120,,,,\nS,S,2,900,RF,,,\n"
                     "S4,S2,2,120,,,,\nS4,S2,2,0,,RG,,",
                     "S4", kAtoB, gl_from_f, g_from_f}),
    case_name<track_change>);

// ===========================================================================
// 4. Track change and delay together, reverting, searching the other way.
// ===========================================================================

// 8 min rule, satisfied by the schedule; F changes platform and is 5 min late.
TEST(gtfsrt_transfer_rules, track_change_and_delay_make_rule_binding) {
  auto const tt = load_network("S,S,2,120,,,,\nS,S,2,480,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  EXPECT_EQ(g_from_f(), arrival(tt, &rtt, kAtoB));  // 10 min >= 8 min
  update(
      tt, rtt,
      {{"F",
        {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3", .arr_delay_ = 5}}}});
  EXPECT_EQ("S3", stop_id_at(tt, rtt, "F", 1U));
  EXPECT_EQ(gl_from_f(), arrival(tt, &rtt, kAtoB));  // 5 min < 8 min
}

// A track change that is taken back restores the scheduled rules.
TEST(gtfsrt_transfer_rules, track_change_reverted) {
  auto const tt = load_network("S1,S2,2,120,,,,\nS1,S2,2,900,,,F,");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  EXPECT_EQ("S3", stop_id_at(tt, rtt, "F", 1U));
  EXPECT_EQ(g_from_f(), arrival(tt, &rtt, kAtoB));
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .arr_delay_ = 0}}}});
  EXPECT_EQ("S1", stop_id_at(tt, rtt, "F", 1U));
  EXPECT_EQ(gl_from_f(), arrival(tt, &rtt, kAtoB));
}

// ST1 -> ST2 stay seated at S1; a rule names ST2 there. Moving the handover
// stop to S3 and taking it back restores the schedule, and a repeated revert
// message changes nothing.
TEST(gtfsrt_transfer_rules,
     revert_at_stay_seated_handover_stop_restores_schedule) {
  auto const tt = load_network(
      "S1,S1,4,,,,ST1,ST2\nS,S,2,120,,,,\nS,S,2,900,,,F,ST2", kRows);
  auto rtt = rt::create_rt_timetable(tt, kDay);
  auto n_location_changes = 0U;
  rtt.set_change_callback([&](transport, stop_idx_t, event_type,
                              std::optional<location_idx_t> const l,
                              std::optional<bool>, std::optional<duration_t>) {
    n_location_changes += l.has_value() ? 1U : 0U;
  });

  auto const handover_location = [&]() {
    auto const [r, _] = resolve(tt, rtt, "ST2");
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

// A second track change replaces the first one.
TEST(gtfsrt_transfer_rules, track_change_twice) {
  auto const tt = load_network("S3,S2,2,120,,,,\nS3,S2,2,900,,,F,");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  EXPECT_EQ(gl_from_f(), arrival(tt, &rtt, kAtoB));
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S4"}}}});
  EXPECT_EQ("S4", stop_id_at(tt, rtt, "F", 1U));
  EXPECT_EQ(g_from_f(), arrival(tt, &rtt, kAtoB));
}

// Backward search uses the rule of the new platform too: arriving at B by
// 11:05 needs G, which F cannot reach from S3. The same holds for a departure
// interval instead of one departure time (range search), in both directions.
TEST(gtfsrt_transfer_rules, track_change_backward_and_interval_search) {
  auto const tt = load_network("S3,S2,2,120,,,,\nS3,S2,2,900,,,F,");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  auto const before =
      search(tt, rtt, std::pair{"B", "A"}, "11:05", direction::kBackward);
  EXPECT_EQ(1U, before.size());
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  auto const after =
      search(tt, rtt, std::pair{"B", "A"}, "11:05", direction::kBackward);
  EXPECT_EQ(0U, after.size());
  auto const later =
      search(tt, rtt, std::pair{"B", "A"}, "11:35", direction::kBackward);
  ASSERT_EQ(1U, later.size());
  EXPECT_EQ(at("10:00"), begin(later)->dest_time_);

  auto const range = [&](direction const dir, char const* from,
                         char const* to) {
    return raptor_search(
        tt, &rtt,
        station_query(tt, from, to, interval{at("09:30"), at("11:40")}), dir);
  };
  auto const fwd = range(direction::kForward, "A", "B");
  ASSERT_EQ(1U, fwd.size());
  EXPECT_EQ(at("10:00"), begin(fwd)->start_time_);
  EXPECT_EQ(gl_from_f(), begin(fwd)->dest_time_);

  auto const bwd = range(direction::kBackward, "B", "A");
  ASSERT_EQ(1U, bwd.size());
  EXPECT_EQ(at("10:00"), begin(bwd)->dest_time_);
  EXPECT_EQ(gl_from_f(), begin(bwd)->start_time_);
}

// ===========================================================================
// 5. What a journey and a query see of a moved trip.
// ===========================================================================

// The journey names the platform the trip really stops at, and a query to the
// station or to the new platform arrives with the moved trip. (A query from
// the station: track_change_of_target_trip_keeps_rule.)
TEST(gtfsrt_transfer_rules, journey_and_query_see_new_platform) {
  auto const tt = load_network("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  EXPECT_EQ("S3", stop_id_at(tt, rtt, "F", 1U));

  auto const res = search(tt, rtt, std::pair{"A", "B"}, "10:00");
  ASSERT_EQ(1U, res.size());
  auto const& legs = begin(res)->legs_;
  ASSERT_LE(2U, legs.size());
  EXPECT_EQ("S3", leg_stop_id(tt, legs.front().to_));
  EXPECT_EQ("S2", leg_stop_id(tt, legs.back().from_));
  for (auto const& l : legs) {
    EXPECT_LT(l.from_, tt.n_locations());
    EXPECT_LT(l.to_, tt.n_locations());
  }

  EXPECT_EQ(at("10:30"), arrival(tt, &rtt, std::pair{"A", "S"}));
  EXPECT_EQ(at("10:30"), arrival(tt, &rtt, std::pair{"A", "S3"}));
  EXPECT_EQ(std::vector{at("10:30")},
            direct_arrivals(tt, rtt, std::pair{"A", "S"}, "10:00", "10:10"));
}

// P -> Q can change at S or at X. Once P moved to S3, where P -> S2 is not
// possible, the change has to happen at X - and optimize_footpaths, which
// moves changes to better places afterwards, must not move it back to S.
TEST(gtfsrt_transfer_rules, track_change_transfer_not_moved_to_forbidden_pair) {
  auto const tt = load_network("S3,S2,2,120,,,,\nS3,S2,3,,,,P,");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"P", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  EXPECT_EQ("S3", stop_id_at(tt, rtt, "P", 1U));
  auto const res = search(tt, rtt, std::pair{"A", "B"}, "12:00");
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("13:20"), begin(res)->dest_time_);
  auto const& legs = begin(res)->legs_;
  ASSERT_LE(2U, legs.size());
  EXPECT_EQ("X1", leg_stop_id(tt, legs.front().to_));
  EXPECT_EQ("X2", leg_stop_id(tt, legs.back().from_));
}

// ===========================================================================
// 6. Situations around the real-time virtual locations themselves.
// ===========================================================================

// Two trips of one route move to the same new platform, and the route's own
// rule (RF -> RF takes 10 min) is their transfer time - F -> F2 has 6 min,
// 11 min once F2 is late.
TEST(gtfsrt_transfer_rules, track_change_of_route_pair_keeps_own_rule) {
  auto const tt = load_network("S,S,2,120,,,,\nS,S,2,600,RF,RF,,");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  EXPECT_EQ(0U, search(tt, rtt, std::pair{"A", "D"}, "10:00").size());
  update(tt, rtt,
         {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S4"}}},
          {"F2", {{.seq_ = 1U, .stop_id_ = "S1", .assigned_ = "S4"}}}});
  EXPECT_EQ("S4", stop_id_at(tt, rtt, "F", 1U));
  EXPECT_EQ("S4", stop_id_at(tt, rtt, "F2", 0U));
  auto const n_rt_locations = rtt.n_rt_locations();
  EXPECT_EQ(0U, search(tt, rtt, std::pair{"A", "D"}, "10:00").size());
  update(
      tt, rtt,
      {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S4"}}},
       {"F2",
        {{.seq_ = 1U, .stop_id_ = "S1", .assigned_ = "S4", .dep_delay_ = 5}}}});
  EXPECT_EQ(n_rt_locations, rtt.n_rt_locations());  // found, not created
  EXPECT_EQ(at("11:11"), arrival(tt, &rtt, std::pair{"A", "D"}));
}

// No change of vehicles at S4 at all (same-stop ban): that also holds between
// the platform and the real-time virtual location of a moved trip.
TEST(gtfsrt_transfer_rules, track_change_onto_platform_without_transfers) {
  auto const rules = std::string{"S,S,2,120,,,,\nS,S,2,900,,,F,G"};
  auto const moves = std::vector<trip_update>{
      {"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S4"}}},
      {"HL", {{.seq_ = 1U, .stop_id_ = "S1", .assigned_ = "S4"}}}};
  {
    auto const tt = load_network(rules);
    auto rtt = rt::create_rt_timetable(tt, kDay);
    update(tt, rtt, moves);
    EXPECT_EQ(gl_from_f(), arrival(tt, &rtt, kAtoC));  // F -> HL at S4
  }
  {
    auto const tt = load_network(rules + "\nS4,S4,3,,,,,");
    auto rtt = rt::create_rt_timetable(tt, kDay);
    update(tt, rtt, moves);
    EXPECT_EQ(0U, search(tt, rtt, kAtoC, "10:00").size());
  }
}

// The kept timed transfer F -> H from a real-time virtual location, searched
// forward and backward - until the stop there gets skipped.
TEST(gtfsrt_transfer_rules, skipped_stop_at_real_time_location) {
  auto const tt = load_network("S,S,2,120,,,,\nS,S,1,,,,F,H");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  EXPECT_EQ("S3", stop_id_at(tt, rtt, "F", 1U));
  EXPECT_EQ(g_from_f(), arrival(tt, &rtt, kAtoC));
  auto const bwd =
      search(tt, rtt, std::pair{"C", "A"}, "11:00", direction::kBackward);
  ASSERT_EQ(1U, bwd.size());
  EXPECT_EQ(at("10:00"), begin(bwd)->dest_time_);

  update(tt, rtt,
         {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .is_skipped_ = true}}}});
  EXPECT_EQ(0U, search(tt, rtt, kAtoC, "10:00").size());
}

// A profile changes at a virtual location in its transfer time: F -> H (1 min)
// only through the F -> H rule - also once H is a real-time transport (5 min
// late at C), which the profile finds at its scheduled location (not at a
// real-time virtual location).
TEST(gtfsrt_transfer_rules, other_profile_finds_real_time_trips_at_rule_stops) {
  constexpr auto const kProfile = profile_idx_t{1U};
  auto tt = load_network("S,S,2,120,,,,\nS,S,2,60,,,F,H");
  add_empty_profile(tt, kProfile);

  auto rtt = rt::create_rt_timetable(tt, kDay);
  auto const arrival_at_c = [&](profile_idx_t const prf) {
    auto q = station_query(tt, "A", "C", at("10:00"));
    q.prf_idx_ = prf;
    auto const res = raptor_search(tt, &rtt, std::move(q), direction::kForward);
    return res.size() == 1U ? begin(res)->dest_time_ : unixtime_t{};
  };
  EXPECT_EQ(g_from_f(), arrival_at_c(kDefaultProfile));
  EXPECT_EQ(g_from_f(), arrival_at_c(kProfile));

  update(tt, rtt, {{"H", {{.seq_ = 2U, .stop_id_ = "C", .arr_delay_ = 5}}}});
  EXPECT_EQ(at("11:05"), arrival_at_c(kDefaultProfile));
  EXPECT_EQ(at("11:05"), arrival_at_c(kProfile));
}

// Door to door: offsets to the platforms the moved trips use now.
TEST(gtfsrt_transfer_rules, intermodal_with_moved_trips) {
  auto const tt = load_network("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt,
         {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}},
          {"G", {{.seq_ = 1U, .stop_id_ = "S2", .assigned_ = "S4"}}}});
  // ... to a door 4 min from S3, arriving with F
  auto const to_s3 = nigiri::test::raptor_intermodal_search(
      tt, &rtt, {{lidx(tt, "A"), 3_minutes, 0U}},
      {{lidx(tt, "S3"), 4_minutes, 0U}}, at("09:50"));
  ASSERT_EQ(1U, to_s3.size());
  EXPECT_EQ(at("10:34"), begin(to_s3)->dest_time_);
  // ... from a door 5 min from S4, leaving with G
  auto const from_s4 = nigiri::test::raptor_intermodal_search(
      tt, &rtt, {{lidx(tt, "S4"), 5_minutes, 0U}},
      {{lidx(tt, "B"), 2_minutes, 0U}}, at("10:30"));
  ASSERT_EQ(1U, from_s4.size());
  EXPECT_EQ(at("11:02"), begin(from_s4)->dest_time_);
}

// What a stop is called must not depend on whether a rule gave the trip stop
// a virtual location: F stops at platform S1 of station S either way, and an
// alert for the station reaches it.
TEST(gtfsrt_transfer_rules, rule_stop_is_its_platform) {
  auto const with_rule = load_network("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  auto const without_rule = load_network("");
  for (auto const* tt : {&without_rule, &with_rule}) {
    auto rtt = rt::create_rt_timetable(*tt, kDay);
    rtt.alerts_.location_[lidx(*tt, "S")].push_back(alert_idx_t{7U});

    auto const [r, trip] = resolve(*tt, rtt, "F");
    ASSERT_TRUE(r.valid());
    auto const fr = rt::frun{*tt, &rtt, r};  // run_stop points into it
    auto const stop = fr[1U];
    EXPECT_EQ("S", stop.name(lang_t{}));
    EXPECT_EQ("S", stop.id());
    EXPECT_EQ("S1", stop.get_location_id());
    EXPECT_TRUE(rtt.alerts_
                    .get_alerts(*tt, source_idx_t{0}, trip,
                                rt_transport_idx_t::invalid(),
                                stop.get_location_idx(), false)
                    .contains(alert_idx_t{7U}));
  }
}

// The timed rule S4 -> X1 for route RF only binds F once F is moved to S4 -
// then F's real-time virtual location has the rule's 0 min transfer to
// X1, and T0 + F + XB arrives at B 10:40, before the direct DAB (10:45) but
// with two changes. When F arrives (round 2), DAB's 10:45 is already known:
// the lower bound of F's location must know that transfer, or the arrival is
// pruned.
TEST(gtfsrt_transfer_rules, lower_bound_knows_real_time_rule_edges) {
  auto const tt = load_network("S4,X1,1,,RF,,,", kRows);
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S4"}}}});
  ASSERT_EQ(1U, rtt.n_rt_locations()) << "precondition";

  auto const arrives_at_10_40 = [](pareto_set<routing::journey> const& res) {
    return utl::any_of(res, [](routing::journey const& j) {
      return j.dest_time_ == at("10:40");
    });
  };

  // Control: from A, F is the first vehicle and nothing is known about B yet
  // when it arrives - the rule's transfer takes F + XB to B at 10:40.
  ASSERT_TRUE(arrives_at_10_40(search(tt, rtt, kAtoB, "10:00")))
      << "precondition: the real-time rule edge works";

  EXPECT_TRUE(arrives_at_10_40(search(tt, rtt, std::pair{"A0", "B"}, "09:30")));
}

// S1 is the stay-seated handover stop of ST1 -> ST2: one vehicle, it arrives as
// ST1 and departs as ST2. Getting off there is ST1's rule (ST1 -> G: 5 min),
// not ST2's (ST2 -> G: 1 min) - nobody gets off ST2 at its first stop. Moved to
// S3 and 6 min late (10:36), the passenger of ST1 misses G (10:40) and takes
// GL (B 11:30). The real-time signature at the handover stop must not let ST2's
// rule decide.
TEST(gtfsrt_transfer_rules,
     track_change_at_handover_stop_keeps_rule_of_arriving_trip) {
  auto const tt = load_network(
      "S1,S1,4,,,,ST1,ST2\nS,S,2,120,,,,\nS,S,2,300,,,ST1,G\n"
      "S,S,2,60,,,ST2,G",
      kRows);
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(
      tt, rtt,
      {{"ST1",
        {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3", .arr_delay_ = 6}}}});
  EXPECT_EQ(gl_from_f(), arrival(tt, &rtt, std::pair{"A2", "B"}));
}

// ===========================================================================
// 7. Rebuilt walks and start walks.
// ===========================================================================

// The walks of the default profile can come from a street router (motis:
// osr_footpath): a trip that moves to another platform walks like that
// platform does. F moves to S4, where a rule names it, so it moves to a
// real-time virtual location.
TEST(gtfsrt_transfer_rules, track_change_follows_routed_walks) {
  constexpr auto const kRules = "S4,S4,2,120,,,,\nS4,S4,2,900,,,F,HL";
  auto const moves = std::vector<trip_update>{
      {"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S4"}}}};
  for (auto const& [tt, expected] :
       {std::pair{load_network(kRules), g_from_f()},  // beeline S4 -> S2: 2 min
        std::pair{test::load_network_with_walks(kRules, {}, {{"S4", "S2", 12}}),
                  gl_from_f()}}) {
    auto rtt = rt::create_rt_timetable(tt, kDay);
    update(tt, rtt, moves);
    EXPECT_EQ(expected, arrival(tt, &rtt, kAtoB));
  }
}

// Z -> S4 is a 5 min rule, Z -> S4 for the trip G a 10 min one. Once G and GL
// are moved to S4, a start at Z 10:32 cannot make G (10:40) but makes GL
// (11:10): the start offset of G's real-time virtual location is the trip
// rule's, not its base location's. With the base location's offset, the
// search would find a journey it cannot reconstruct and drop it - and GL,
// which it dominated, with it.
TEST(gtfsrt_transfer_rules, start_offset_of_real_time_location_honours_rule) {
  auto const tt = load_network("Z,S4,2,300,,,,\nZ,S4,2,600,,,,G", kRows);
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
  EXPECT_EQ(gl_from_f(), begin(res)->dest_time_);
}

// The first walk of a journey that boards at a real-time virtual location is
// that location's own, the one the search took: the rule's 30 min to G at S4,
// not the 2 min to the platform S4 - neither when the journey is reconstructed
// nor when its walks are shortened afterwards.
TEST(gtfsrt_transfer_rules, start_walk_to_real_time_virtual_location) {
  auto const tt = load_network("S1,S4,2,120,,,,\nS1,S4,2,1800,,,,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"G", {{.seq_ = 1U, .stop_id_ = "S2", .assigned_ = "S4"}}}});
  ASSERT_EQ(1U, rtt.n_rt_locations()) << "precondition";

  auto q = test::exact_query(tt, "S1", "B", at("10:05"));
  q.use_start_footpaths_ = true;
  auto const res = raptor_search(tt, &rtt, std::move(q), direction::kForward);
  ASSERT_EQ(1U, res.size());
  auto const& j = *begin(res);
  ASSERT_TRUE(holds_alternative<footpath>(j.legs_.front().uses_));
  EXPECT_EQ("S4", leg_stop_id(tt, j.legs_.front().to_));
  EXPECT_EQ(30_minutes, get<footpath>(j.legs_.front().uses_).duration());
}

// One-to-all from S4 at 10:30 boards G, moved there: G leaves from its
// real-time virtual location at S4, which starts like the stop itself. B is
// reached with G at 11:00.
TEST(gtfsrt_transfer_rules, one_to_all_starts_at_real_time_virtual_location) {
  auto const tt = load_network("S1,S4,2,120,,,,\nS1,S4,2,1800,,,,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"G", {{.seq_ = 1U, .stop_id_ = "S2", .assigned_ = "S4"}}}});
  ASSERT_EQ(1U, rtt.n_rt_locations()) << "precondition";

  auto const b =
      test::one_to_all(tt, &rtt, direction::kForward, "S4", "B", "10:30")
          .fastest_;
  EXPECT_EQ(1U, b.k_);
  EXPECT_EQ(30, b.duration_);
}

// The via S4 (10 min) holds at G's real-time virtual location there as it
// does at the stop: Z -> S4 is a 10 min rule walk for G, so G (S4 10:40)
// leaves Z at 10:20 at the latest - not at 10:30, without the stay.
TEST(gtfsrt_transfer_rules, start_walk_to_real_time_virtual_location_of_via) {
  auto const tt = load_network("Z,S4,2,300,,,,\nZ,S4,2,600,,,,G", kRows);
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"G", {{.seq_ = 1U, .stop_id_ = "S2", .assigned_ = "S4"}}}});
  ASSERT_EQ(1U, rtt.n_rt_locations()) << "precondition";

  auto q = station_query(tt, "Z", "B", interval{at("10:00"), at("10:35")});
  q.start_match_mode_ = routing::location_match_mode::kExact;
  q.use_start_footpaths_ = true;
  q.via_stops_ = {{lidx(tt, "S4"), 10_minutes}};
  auto const res = raptor_search(tt, &rtt, std::move(q), direction::kForward);
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("10:20"), begin(res)->start_time_);
  EXPECT_EQ(g_from_f(), begin(res)->dest_time_);
}

// A profile sees G at the platform S4, not at its real-time virtual location
// (the default profile's) with the rule's 5 min from S1: the profile has no
// walks between stops, so after F arrives at S1 (10:30), G (10:40) is out of
// reach.
TEST(gtfsrt_transfer_rules, profile_ignores_real_time_virtual_location) {
  constexpr auto const kProfile = profile_idx_t{1U};
  auto tt = load_network("S1,S4,2,120,,,,\nS1,S4,2,300,,,,G");
  add_empty_profile(tt, kProfile);
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"G", {{.seq_ = 1U, .stop_id_ = "S2", .assigned_ = "S4"}}}});
  ASSERT_EQ(1U, rtt.n_rt_locations()) << "precondition";

  auto q = test::exact_query(tt, "A", "B", at("10:00"));
  q.prf_idx_ = kProfile;
  EXPECT_TRUE(
      raptor_search(tt, &rtt, std::move(q), direction::kForward).empty());
}

// S4 bans changes, except RF -> RH (1 min). H4 (RH) leaves S4 15 min after F,
// moved there, arrives: the exception holds at any gap.
TEST(gtfsrt_transfer_rules, track_change_onto_banned_platform_keeps_exception) {
  auto const tt = load_network(
      "S4,S4,3,,,,,\nS4,S4,2,60,RF,RH,,",
      {.trips_ = {{"H4", "RH", {{"S4", "10:45"}, {"C", "11:15"}}}}});
  auto rtt = rt::create_rt_timetable(tt, kDay);
  EXPECT_EQ(at("11:15"), arrival(tt, &rtt, kAtoC));  // F -> walk -> H4
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S4"}}}});
  EXPECT_EQ("S4", stop_id_at(tt, rtt, "F", 1U));
  EXPECT_EQ(at("11:15"), arrival(tt, &rtt, kAtoC));
}

// The RF -> RG rule (8 min) puts F on a location it only arrives at. The
// through trip Y (RF) is 20 min late: Y -> GL (5 min) needs the rule as well,
// so Y moves to F's location - where K's passengers have to be able to board
// it.
//   GX (RG): S2 10:35 -> B 10:55 (makes F -> RG binding)
//   Y  (RF): A 10:00 -> S1 10:45 -> D 11:15
//   K  (RK): C 10:20 -> S2 10:40, then a walk to Y at S1
TEST(gtfsrt_transfer_rules, move_onto_location_without_departures) {
  auto const tt = load_network(
      "S,S,2,120,,,,\nS,S,2,480,RF,RG,,",
      {.trips_ = {
           {"GX", "RG", {{"S2", "10:35"}, {"B", "10:55"}}},
           {"Y", "RF", {{"A", "10:00"}, {"S1", "10:45"}, {"D", "11:15"}}},
           {"K", "RK", {{"C", "10:20"}, {"S2", "10:40"}}}}});
  ASSERT_LT(0U, test::n_virts(tt));
  auto rtt = rt::create_rt_timetable(tt, kDay);
  EXPECT_EQ(at("11:15"), arrival(tt, &rtt, std::pair{"C", "D"}, "10:15"));
  update(tt, rtt,
         {{"Y",
           {{.seq_ = 2U, .stop_id_ = "S1", .arr_delay_ = 20, .dep_delay_ = 20},
            {.seq_ = 3U, .stop_id_ = "D", .arr_delay_ = 20}}}});
  EXPECT_EQ(at("11:35"), arrival(tt, &rtt, std::pair{"C", "D"}, "10:15"));
}
