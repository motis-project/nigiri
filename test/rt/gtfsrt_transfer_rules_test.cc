#include "gtest/gtest.h"

#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gtfsrt/gtfs-realtime.pb.h"

#include "nigiri/loader/build_footpaths.h"
#include "nigiri/routing/direct.h"
#include "nigiri/routing/query.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

#include "../raptor_search.h"
#include "../transfer_rules_util.h"
#include "./transfer_rules_rt_util.h"

// Real-time updates in the presence of transfers.txt rules, on the network
// test::network() (transfer_rules_util.h). Every test states its own
// transfers.txt rows and real-time messages.

using namespace nigiri;
using nigiri::test::add_empty_profile;
using nigiri::test::arrival;
using nigiri::test::at;
using nigiri::test::g_from_f;
using nigiri::test::gl_from_f;
using nigiri::test::h_from_f;
using nigiri::test::hl_from_f;
using nigiri::test::kDay;
using nigiri::test::lidx;
using nigiri::test::load_network;
using nigiri::test::raptor_search;
using nigiri::test::station_query;
using nigiri::test::trip_update;
using nigiri::test::update;

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
  auto const [r, _] = test::resolve(tt, rtt, trip_id);
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

// ===========================================================================
// 1. Delays at stops that carry a qualified rule (= virtual locations).
//    No track change involved: the rule has to keep applying and the update
//    has to arrive, however the feed names the stop.
// ===========================================================================

// G waits: 10:40 -> 10:50. 20 min >= 15 min rule -> G is reachable, and its
// delay reaches B - however the update names the stop: by stop_id and
// stop_sequence, by stop_id only, by stop_sequence only.
TEST(gtfsrt_transfer_rules, delay_at_rule_stop) {
  auto const tt = load_network("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  for (auto const& [has_seq, has_stop_id] :
       {std::pair{true, true}, std::pair{false, true},
        std::pair{true, false}}) {
    SCOPED_TRACE(
        fmt::format("has_seq={}, has_stop_id={}", has_seq, has_stop_id));
    auto rtt = rt::create_rt_timetable(tt, kDay);
    update(tt, rtt,
           {{"G",
             {{.seq_ = 1U,
               .stop_id_ = "S2",
               .dep_delay_ = 10,
               .has_seq_ = has_seq,
               .has_stop_id_ = has_stop_id}}}});
    EXPECT_EQ(at("11:10"), arrival(tt, &rtt, kAtoB));
    EXPECT_EQ("S2", stop_id_at(tt, rtt, "G", 0U));
  }
}

// A slower rule that the schedule satisfies (10 min >= 8 min) becomes binding
// once the feeder is late: 5 min < 8 min although the walk alone would do.
TEST(gtfsrt_transfer_rules, delay_makes_slower_rule_binding) {
  auto const tt = load_network("S,S,2,120,,,,\nS,S,2,480,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  EXPECT_EQ(g_from_f(), arrival(tt, &rtt, kAtoB));
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .arr_delay_ = 5}}}});
  EXPECT_EQ(gl_from_f(), arrival(tt, &rtt, kAtoB));
}

// A faster rule (0 min, route pair) keeps a connection the default change
// time (2 min) would break - also after both trips moved: F arrives 10:33,
// H leaves 10:33 and reaches C two minutes late.
TEST(gtfsrt_transfer_rules, delay_keeps_faster_rule) {
  auto const tt = load_network("S1,S1,2,120,,,,\nS1,S1,2,0,RF,RH,,");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  EXPECT_EQ(h_from_f(), arrival(tt, &rtt, kAtoC));
  update(tt, rtt,
         {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .arr_delay_ = 3}}},
          {"H", {{.seq_ = 1U, .stop_id_ = "S1", .dep_delay_ = 2}}}});
  EXPECT_EQ(at("11:02"), arrival(tt, &rtt, kAtoC));
}

// A forbidden trip pair stays forbidden however late the connecting trip is.
TEST(gtfsrt_transfer_rules, delay_keeps_forbidden_pair) {
  auto const tt = load_network("S,S,2,120,,,,\nS,S,3,,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"G", {{.seq_ = 1U, .stop_id_ = "S2", .dep_delay_ = 5}}}});
  EXPECT_EQ(gl_from_f(), arrival(tt, &rtt, kAtoB));
  EXPECT_EQ("S2", stop_id_at(tt, rtt, "G", 0U));
  // ... and the delay arrived: G now leaves S2 at 10:45, reaches B at 11:05
  EXPECT_EQ(at("11:05"), arrival(tt, &rtt, std::pair{"S2", "B"}, "10:41"));
}

// A skipped stop named by stop_id only.
TEST(gtfsrt_transfer_rules, skipped_rule_stop_with_stop_id_only) {
  auto const tt = load_network("S,S,2,120,,,,\nS,S,2,0,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt,
         {{"G",
           {{.seq_ = 1U,
             .stop_id_ = "S2",
             .has_seq_ = false,
             .is_skipped_ = true}}}});
  EXPECT_EQ(gl_from_f(),
            arrival(tt, &rtt, kAtoB));  // G cannot be boarded at S2
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
};

std::ostream& operator<<(std::ostream& out, track_change const& c) {
  return out << c.name_;
}

std::string track_change_name(
    testing::TestParamInfo<track_change> const& info) {
  return std::string{info.param.name_};
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
             .assigned_ = std::string{c.platform_}}}}});
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
        // Faster: a timed transfer F -> H (H waits, 0 min) survives F moving
        // away from H's platform, where the walk alone would take longer than 1
        // min.
        track_change{"track_change_keeps_timed_trip_rule",
                     "S,S,2,120,,,,\nS,S,1,,,,F,H", "S3", kAtoC, h_from_f,
                     h_from_f},
        // Without the rule (0 min at S1 for everyone) the same track change
        // does break the connection.
        track_change{"track_change_without_rule_breaks_tight_connection",
                     "S1,S1,2,0,,,,", "S3", kAtoC, h_from_f, hl_from_f},
        // Forbidden: the trip pair stays forbidden from the new platform.
        track_change{"track_change_keeps_forbidden_trip_pair",
                     "S,S,2,120,,,,\nS,S,3,,,,F,G", "S3", kAtoB, nullptr,
                     gl_from_f},
        // Route-qualified station rule, new platform S4 never saw an RF trip.
        track_change{"track_change_keeps_route_rule_new_location",
                     "S,S,2,120,,,,\nS,S,2,900,RF,RG,,", "S4", kAtoB, nullptr,
                     gl_from_f}),
    track_change_name);

// The rule also holds if its TARGET trip changes platform.
TEST(gtfsrt_transfer_rules, track_change_of_target_trip_keeps_rule) {
  auto const tt = load_network("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"G", {{.seq_ = 1U, .stop_id_ = "S2", .assigned_ = "S4"}}}});
  EXPECT_EQ("S4", stop_id_at(tt, rtt, "G", 0U));
  EXPECT_EQ(gl_from_f(), arrival(tt, &rtt, kAtoB));
  // G itself is still usable from its new platform.
  EXPECT_EQ(g_from_f(), arrival(tt, &rtt, std::pair{"S4", "B"}, "10:35"));
}

// ... and if both change platform (two real-time virtual locations meet).
TEST(gtfsrt_transfer_rules, track_change_of_both_trips_keeps_rule) {
  auto const tt = load_network("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt,
         {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}},
          {"G", {{.seq_ = 1U, .stop_id_ = "S2", .assigned_ = "S4"}}}});
  EXPECT_EQ("S3", stop_id_at(tt, rtt, "F", 1U));
  EXPECT_EQ("S4", stop_id_at(tt, rtt, "G", 0U));
  EXPECT_EQ(gl_from_f(), arrival(tt, &rtt, kAtoB));
}

// Timed transfer with both trips moved to platforms nobody was scheduled at.
TEST(gtfsrt_transfer_rules, track_change_of_both_trips_keeps_timed_rule) {
  auto const tt = load_network("S,S,2,120,,,,\nS,S,1,,,,F,H");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt,
         {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S4"}}},
          {"H", {{.seq_ = 1U, .stop_id_ = "S1", .assigned_ = "S2"}}}});
  EXPECT_EQ("S4", stop_id_at(tt, rtt, "F", 1U));
  EXPECT_EQ("S2", stop_id_at(tt, rtt, "H", 0U));
  EXPECT_EQ(h_from_f(), arrival(tt, &rtt, kAtoC));
}

// Route-qualified station rule; another RF trip (F0) is scheduled at S3, so
// the location F needs there already exists.
TEST(gtfsrt_transfer_rules, track_change_keeps_route_rule_existing_location) {
  auto const tt = load_network("S,S,2,120,,,,\nS,S,2,900,RF,RG,,");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  EXPECT_EQ(gl_from_f(), arrival(tt, &rtt, kAtoB));
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  EXPECT_EQ("S3", stop_id_at(tt, rtt, "F", 1U));
  EXPECT_EQ(0U, rtt.n_rt_locations());  // F0's location, no new one
  EXPECT_EQ(gl_from_f(), arrival(tt, &rtt, kAtoB));
}

// The track change stated the old way: a different stop_id, no
// stop_time_properties.
TEST(gtfsrt_transfer_rules, track_change_via_stop_id_keeps_rule) {
  auto const tt = load_network("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt,
         {{"F",
           {{.seq_ = 2U,
             .stop_id_ = "S1",
             .assigned_ = "S3",
             .is_assigned_as_stop_id_ = true}}}});
  EXPECT_EQ("S3", stop_id_at(tt, rtt, "F", 1U));
  EXPECT_EQ(gl_from_f(), arrival(tt, &rtt, kAtoB));
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
                     [] { return at("13:20"); }}),
    track_change_name);

// Entering a fast platform: 0 min at S3 for RF -> RH, and H moves there too.
TEST(gtfsrt_transfer_rules, track_change_makes_faster_platform_rule_binding) {
  auto const tt = load_network("S3,S3,2,120,,,,\nS3,S3,2,0,RF,RH,,");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  EXPECT_EQ(hl_from_f(),
            arrival(tt, &rtt, kAtoC));  // S1: 1 min < 2 min default
  update(tt, rtt,
         {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}},
          {"H", {{.seq_ = 1U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  EXPECT_EQ("S3", stop_id_at(tt, rtt, "F", 1U));
  EXPECT_EQ("S3", stop_id_at(tt, rtt, "H", 0U));
  EXPECT_EQ(h_from_f(), arrival(tt, &rtt, kAtoC));
}

// The specific rule wins on the new platform as well: the station says 15 min
// for RF -> RG, the trip pair F -> G is timed.
TEST(gtfsrt_transfer_rules, track_change_keeps_rule_precedence) {
  auto const tt =
      load_network("S,S,2,120,,,,\nS,S,2,900,RF,RG,,\nS,S,1,,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  EXPECT_EQ(g_from_f(), arrival(tt, &rtt, kAtoB));
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S4"}}}});
  EXPECT_EQ("S4", stop_id_at(tt, rtt, "F", 1U));
  EXPECT_EQ(g_from_f(), arrival(tt, &rtt, kAtoB));

  // Delaying G only delays the arrival: still G, 5 min later.
  update(tt, rtt,
         {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S4"}}},
          {"G", {{.seq_ = 1U, .stop_id_ = "S2", .dep_delay_ = 5}}}});
  EXPECT_EQ(at("11:05"), arrival(tt, &rtt, kAtoB));
}

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
// 11:05 needs G, which F cannot reach from S3.
TEST(gtfsrt_transfer_rules, track_change_backward_search) {
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
}

// Backward search across a kept timed transfer.
TEST(gtfsrt_transfer_rules, track_change_backward_search_timed) {
  auto const tt = load_network("S,S,2,120,,,,\nS,S,1,,,,F,H");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  EXPECT_EQ("S3", stop_id_at(tt, rtt, "F", 1U));
  auto const res =
      search(tt, rtt, std::pair{"C", "A"}, "11:00", direction::kBackward);
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("10:00"), begin(res)->dest_time_);
}

// ===========================================================================
// 5. What a journey and a query see of a moved trip.
// ===========================================================================

// The journey names the platform the trip really stops at.
TEST(gtfsrt_transfer_rules, journey_shows_new_platform) {
  auto const tt = load_network("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
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
}

// A query from the station finds the moved trip at its new platform.
TEST(gtfsrt_transfer_rules, start_at_station_with_moved_trip) {
  auto const tt = load_network("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"G", {{.seq_ = 1U, .stop_id_ = "S2", .assigned_ = "S4"}}}});
  EXPECT_EQ("S4", stop_id_at(tt, rtt, "G", 0U));
  EXPECT_EQ(g_from_f(), arrival(tt, &rtt, std::pair{"S", "B"}, "10:35"));
  EXPECT_EQ(std::vector{g_from_f()},
            direct_arrivals(tt, rtt, std::pair{"S", "B"}, "10:35", "10:45"));
}

// A query to the station arrives with the moved trip.
TEST(gtfsrt_transfer_rules, destination_at_station_with_moved_trip) {
  auto const tt = load_network("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  EXPECT_EQ("S3", stop_id_at(tt, rtt, "F", 1U));
  EXPECT_EQ(at("10:30"), arrival(tt, &rtt, std::pair{"A", "S"}));
  // ... and to the new platform itself.
  EXPECT_EQ(at("10:30"), arrival(tt, &rtt, std::pair{"A", "S3"}));
  EXPECT_EQ(std::vector{at("10:30")},
            direct_arrivals(tt, rtt, std::pair{"A", "S"}, "10:00", "10:10"));
}

// A departure interval instead of one departure time (range search).
TEST(gtfsrt_transfer_rules, track_change_interval_search) {
  auto const tt = load_network("S3,S2,2,120,,,,\nS3,S2,2,900,,,F,");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  auto const run = [&](direction const dir, char const* from, char const* to) {
    return raptor_search(
        tt, &rtt,
        station_query(tt, from, to, interval{at("09:30"), at("11:40")}), dir);
  };
  auto const fwd = run(direction::kForward, "A", "B");
  ASSERT_EQ(1U, fwd.size());
  EXPECT_EQ(at("10:00"), begin(fwd)->start_time_);
  EXPECT_EQ(gl_from_f(), begin(fwd)->dest_time_);

  auto const bwd = run(direction::kBackward, "B", "A");
  ASSERT_EQ(1U, bwd.size());
  EXPECT_EQ(at("10:00"), begin(bwd)->dest_time_);
  EXPECT_EQ(gl_from_f(), begin(bwd)->start_time_);
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

// ... while without the rule the same journey may change at either station.
TEST(gtfsrt_transfer_rules, transfer_at_either_station) {
  auto const tt = load_network("S3,S2,2,120,,,,");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"P", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  EXPECT_EQ(at("13:20"), arrival(tt, &rtt, kAtoB, "12:00"));
}

// ===========================================================================
// 6. Situations around the real-time virtual locations themselves.
// ===========================================================================

// Two trips of one route move to the same new platform: they share one
// real-time virtual location, and the route's own rule (RF -> RF takes
// 10 min) is its transfer time - F -> F2 has 6 min, 11 min once F2 is late.
TEST(gtfsrt_transfer_rules, track_change_shared_location_own_transfer_time) {
  auto const tt = load_network("S,S,2,120,,,,\nS,S,2,600,RF,RF,,");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  EXPECT_EQ(0U, search(tt, rtt, std::pair{"A", "D"}, "10:00").size());
  update(tt, rtt,
         {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S4"}}},
          {"F2", {{.seq_ = 1U, .stop_id_ = "S1", .assigned_ = "S4"}}}});
  EXPECT_EQ("S4", stop_id_at(tt, rtt, "F", 1U));
  EXPECT_EQ("S4", stop_id_at(tt, rtt, "F2", 0U));
  EXPECT_EQ(1U, rtt.n_rt_locations());
  EXPECT_EQ(0U, search(tt, rtt, std::pair{"A", "D"}, "10:00").size());
  update(
      tt, rtt,
      {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S4"}}},
       {"F2",
        {{.seq_ = 1U, .stop_id_ = "S1", .assigned_ = "S4", .dep_delay_ = 5}}}});
  EXPECT_EQ(1U, rtt.n_rt_locations());  // found again, not created again
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
    EXPECT_EQ(hl_from_f(), arrival(tt, &rtt, kAtoC));  // F -> HL at S4
  }
  {
    auto const tt = load_network(rules + "\nS4,S4,3,,,,,");
    auto rtt = rt::create_rt_timetable(tt, kDay);
    update(tt, rtt, moves);
    EXPECT_EQ(0U, search(tt, rtt, kAtoC, "10:00").size());
  }
}

// A stop at a real-time virtual location gets skipped afterwards.
TEST(gtfsrt_transfer_rules, skipped_stop_at_real_time_location) {
  auto const tt = load_network("S,S,2,120,,,,\nS,S,1,,,,F,H");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S3"}}}});
  EXPECT_EQ(h_from_f(), arrival(tt, &rtt, kAtoC));
  update(tt, rtt,
         {{"F", {{.seq_ = 2U, .stop_id_ = "S1", .is_skipped_ = true}}}});
  EXPECT_EQ(0U, search(tt, rtt, kAtoC, "10:00").size());
}

INSTANTIATE_TEST_SUITE_P(
    precedence,
    gtfsrt_track_change,
    testing::Values(
        // Precedence against a rule that names the new platform without
        // qualifying the trip: "anything from S4 to an RG trip at S2: 0 min"
        // names both stops exactly and beats "RF trips at the station: 15 min".
        track_change{"track_change_platform_rule_beats_station_rule",
                     "S,S,2,120,,,,\nS,S,2,900,RF,,,\n"
                     "S4,S2,2,120,,,,\nS4,S2,2,0,,RG,,",
                     "S4", kAtoB, gl_from_f, g_from_f}),
    track_change_name);

// An assignment that names the scheduled platform is no stop change - and
// must not swallow the delay of the stop either.
TEST(gtfsrt_transfer_rules, assignment_to_scheduled_platform) {
  auto const tt = load_network("S,S,2,120,,,,\nS,S,2,480,,,F,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(
      tt, rtt,
      {{"F",
        {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S1", .arr_delay_ = 5}}}});
  EXPECT_EQ("S1", stop_id_at(tt, rtt, "F", 1U));
  EXPECT_EQ(0U, rtt.n_rt_locations());
  EXPECT_EQ(gl_from_f(), arrival(tt, &rtt, kAtoB));  // 5 min < 8 min
}

// A profile that ignores transfers.txt routes on the base locations (virtual
// locations are projected away): F -> HL is forbidden for the default profile
// but fine for this one - also once HL is a real-time transport, which is
// registered at its virtual location and has to be found from the base
// location.
TEST(gtfsrt_transfer_rules, other_profile_finds_real_time_trips_at_rule_stops) {
  constexpr auto const kProfile = profile_idx_t{1U};
  auto tt = load_network("S,S,2,120,,,,\nS,S,3,,,,F,HL");
  add_empty_profile(tt, kProfile);

  auto rtt = rt::create_rt_timetable(tt, kDay);
  auto const run = [&](profile_idx_t const prf) {
    auto q = station_query(tt, "A", "C", at("10:00"));
    q.prf_idx_ = prf;
    return raptor_search(tt, &rtt, std::move(q), direction::kForward);
  };
  EXPECT_EQ(0U, run(kDefaultProfile).size());
  auto const scheduled = run(kProfile);
  ASSERT_EQ(1U, scheduled.size());
  EXPECT_EQ(hl_from_f(), begin(scheduled)->dest_time_);

  update(tt, rtt, {{"HL", {{.seq_ = 1U, .stop_id_ = "S1", .dep_delay_ = 1}}}});
  EXPECT_EQ(0U, run(kDefaultProfile).size());
  auto const delayed = run(kProfile);
  ASSERT_EQ(1U, delayed.size());
  EXPECT_EQ(at("11:31"), begin(delayed)->dest_time_);
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
// a virtual location: F stops at platform S1 of station S either way.
TEST(gtfsrt_transfer_rules, stop_name_and_id_at_rule_stop) {
  auto const with_rule = load_network("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  auto const without_rule = load_network("");
  for (auto const* tt : {&without_rule, &with_rule}) {
    auto const rtt = rt::create_rt_timetable(*tt, kDay);
    auto const [r, _] = test::resolve(*tt, rtt, "F");
    ASSERT_TRUE(r.valid());
    auto const fr = rt::frun{*tt, &rtt, r};  // run_stop points into it
    auto const stop = fr[1U];
    EXPECT_EQ("S", stop.name(lang_t{}));
    EXPECT_EQ("S", stop.id());
    EXPECT_EQ("S1", stop.get_location_id());
  }
}

// An alert for the station reaches a trip that stops at a virtual location
// below one of the station's platforms (F at S1).
TEST(gtfsrt_transfer_rules, station_alert_at_rule_stop) {
  auto const with_rule = load_network("S,S,2,120,,,,\nS,S,2,900,,,F,G");
  auto const without_rule = load_network("");
  for (auto const* tt : {&without_rule, &with_rule}) {
    auto rtt = rt::create_rt_timetable(*tt, kDay);
    auto const src = source_idx_t{0};
    rtt.alerts_.location_[lidx(*tt, "S")].push_back(alert_idx_t{7U});

    auto const [r, trip] = test::resolve(*tt, rtt, "F");
    ASSERT_TRUE(r.valid());
    auto const fr = rt::frun{*tt, &rtt, r};
    auto const alerts =
        rtt.alerts_.get_alerts(*tt, src, trip, rt_transport_idx_t::invalid(),
                               fr[1U].get_location_idx(), false);
    EXPECT_TRUE(alerts.contains(alert_idx_t{7U}));
  }
}

// ===========================================================================
// 7. Unqualified rules, stops without rules and rebuilt walks.
// ===========================================================================

INSTANTIATE_TEST_SUITE_P(unqualified_rules,
                         gtfsrt_track_change,
                         testing::Values(
                             // An unqualified platform rule needs no virtual
                             // location: it stays with S1.
                             track_change{"unqualified_platform_rule",
                                          "S1,S2,2,900,,,,", "S3", kAtoB,
                                          gl_from_f, g_from_f}),
                         track_change_name);

// The walks of the default profile can be replaced after the import (street
// routing with osr_footpath): a trip that moves to another platform walks like
// that platform does then. F moves to S4, where a rule names it, so it moves
// to a real-time virtual location.
TEST(gtfsrt_transfer_rules, track_change_inherits_rebuilt_walks) {
  auto const moves = std::vector<trip_update>{
      {"F", {{.seq_ = 2U, .stop_id_ = "S1", .assigned_ = "S4"}}}};
  auto tt = load_network("S4,S4,2,120,,,,\nS4,S4,2,900,,,F,HL");
  {
    auto rtt = rt::create_rt_timetable(tt, kDay);
    update(tt, rtt, moves);
    EXPECT_EQ(1U, rtt.n_rt_locations());
    EXPECT_EQ(g_from_f(), arrival(tt, &rtt, kAtoB));  // beeline S4 -> S2: 2 min
  }

  test::rebuild_default_profile(tt, {{"S4", "S2", 12}});
  {
    auto rtt = rt::create_rt_timetable(tt, kDay);
    update(tt, rtt, moves);
    EXPECT_EQ(1U, rtt.n_rt_locations());
    EXPECT_EQ(gl_from_f(), arrival(tt, &rtt, kAtoB));
  }
}

// The first walk of a journey that boards at a real-time virtual location is
// that location's own, the one the search took: the rule's 30 min to G at S4,
// not the 2 min to the platform S4 - neither when the journey is reconstructed
// nor when its walks are shortened afterwards. test::raptor_search also checks
// that the GPU shows the same journey.
TEST(gtfsrt_transfer_rules, start_walk_to_real_time_virtual_location) {
  auto const tt = load_network("S1,S4,2,120,,,,\nS1,S4,2,1800,,,,G");
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"G", {{.seq_ = 1U, .stop_id_ = "S2", .assigned_ = "S4"}}}});
  ASSERT_EQ(1U, rtt.n_rt_locations()) << "precondition";

  auto q = routing::query{.start_time_ = at("10:05"),
                          .use_start_footpaths_ = true,
                          .start_ = {{lidx(tt, "S1"), 0_minutes, 0U}},
                          .destination_ = {{lidx(tt, "B"), 0_minutes, 0U}}};
  auto const res = raptor_search(tt, &rtt, std::move(q), direction::kForward);
  ASSERT_EQ(1U, res.size());
  auto const& j = *begin(res);
  ASSERT_TRUE(holds_alternative<footpath>(j.legs_.front().uses_));
  EXPECT_EQ("S4", leg_stop_id(tt, j.legs_.front().to_));
  EXPECT_EQ(30_minutes, get<footpath>(j.legs_.front().uses_).duration());
}

// A profile that projects virtual locations sees G at the platform S4, not at
// its real-time virtual location with the rule's 5 min from S1: the profile
// has no footpaths, so after F arrives at S1 (10:30), G (10:40) is out of
// reach. test::raptor_search checks that the GPU finds the same journeys.
TEST(gtfsrt_transfer_rules,
     projecting_profile_ignores_real_time_virtual_location) {
  constexpr auto const kProfile = profile_idx_t{1U};
  auto tt = load_network("S1,S4,2,120,,,,\nS1,S4,2,300,,,,G");
  add_empty_profile(tt, kProfile);
  auto rtt = rt::create_rt_timetable(tt, kDay);
  update(tt, rtt, {{"G", {{.seq_ = 1U, .stop_id_ = "S2", .assigned_ = "S4"}}}});
  ASSERT_EQ(1U, rtt.n_rt_locations()) << "precondition";

  auto q = routing::query{.start_time_ = at("10:00"),
                          .start_ = {{lidx(tt, "A"), 0_minutes, 0U}},
                          .destination_ = {{lidx(tt, "B"), 0_minutes, 0U}},
                          .prf_idx_ = kProfile};
  EXPECT_TRUE(
      raptor_search(tt, &rtt, std::move(q), direction::kForward).empty());
}
