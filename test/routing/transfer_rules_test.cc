#include "gtest/gtest.h"

#include <algorithm>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "utl/helpers/algorithm.h"

#include "nigiri/loader/build_footpaths.h"
#include "nigiri/routing/direct.h"
#include "nigiri/routing/get_fastest_direct.h"
#include "nigiri/routing/leg_alternatives.h"
#include "nigiri/routing/one_to_all.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/pong.h"
#include "nigiri/routing/transfer_durations.h"
#include "nigiri/routing/transfers.h"
#include "nigiri/rt/frun.h"
#include "nigiri/td_footpath.h"
#include "nigiri/timetable.h"
#include "nigiri/timetable_metrics.h"

#include "../raptor_search.h"
#include "../transfer_rules_util.h"

// Consumers of virtual locations besides the GTFS loader: transfer time
// settings, td footpaths, start and destination matching, profiles,
// recommended transfers, leg alternatives, footpath lookups and direct
// connections. What the rules mean is tested with the GTFS loader
// (test/loader/gtfs/transfer_rules_test.cc). All feeds run on 2019-05-01 in
// Europe/Berlin; the default transfer time at a stop is 2 min.

using namespace nigiri;
using nigiri::test::add_empty_profile;
using nigiri::test::arrival_at;
using nigiri::test::at;
using nigiri::test::feed;
using nigiri::test::lidx;
using nigiri::test::load_feeds;
using nigiri::test::n_virts;
using nigiri::test::raptor_search;
using nigiri::test::search_at;
using nigiri::test::search_pong;

// The shared feed of the tests that change out of a trip at a virtual
// location: FA (route RF1) arrives at U 3 min before FC (RF3) leaves, which
// the RF1 -> RF3 rule (5 min) forbids, so FA and FC stop at locations of their
// own. FB and FB2 (RF2) leave from U itself, FA -> FB is a plain change at U
// (2 min).
std::string virt_feed() {
  return feed({{"U", 55.0, 13.0},
               {"L", 55.1, 13.0},
               {"M", 55.2, 13.0},
               {"N", 55.3, 13.0}},
              {{"FA", "RF1", {{"L", "10:00"}, {"U", "10:30"}}},
               {"FB", "RF2", {{"U", "10:33"}, {"M", "11:00"}}},
               {"FB2", "RF2", {{"U", "10:50"}, {"M", "11:20"}}},
               {"FC", "RF3", {{"U", "10:33"}, {"N", "11:00"}}}},
              "U,U,2,120,,,,\n"
              "U,U,2,300,RF1,RF3,,\n");
}

// W,W,1 makes W's own transfer time 0 min. GA (RG1) arrives at W 3 min before
// GD (RG3) leaves, which the RG1 -> RG3 rule (5 min) forbids, so GA stops at a
// location of its own and W gets 0 min transfers.
std::string zero_min_transfer_feed() {
  return feed({{"W", 56.0, 14.0},
               {"N", 56.1, 14.0},
               {"N2", 56.1, 14.2},
               {"O", 56.2, 14.0},
               {"P", 56.3, 14.0}},
              {{"GA", "RG1", {{"N", "10:00"}, {"W", "10:30"}}},
               {"GB", "RG2", {{"W", "10:32"}, {"O", "11:00"}}},
               {"GB2", "RG2", {{"W", "10:40"}, {"O", "11:10"}}},
               {"GC", "RG4", {{"N2", "10:00"}, {"W", "10:30"}}},
               {"GD", "RG3", {{"W", "10:33"}, {"P", "11:00"}}}},
              "W,W,1,,,,,\n"
              "W,W,2,300,RG1,RG3,,\n");
}

// Q2 leaves QS 10 min after Q1 arrives, the Q1 -> Q2 rule asks for 15 min: Q1
// and Q2 stop at locations of their own. Without the rule, changing between
// them takes QS's own 2 min.
std::string virt_change_feed() {
  return feed({{"QS", 63.0, 22.0}, {"QA", 63.1, 22.0}, {"QB", 63.2, 22.0}},
              {{"Q1", "RQ1", {{"QA", "10:00"}, {"QS", "10:30"}}},
               {"Q2", "RQ2", {{"QS", "10:40"}, {"QB", "11:00"}}}},
              "QS,QS,2,120,,,,\n"
              "QS,QS,2,900,,,Q1,Q2\n");
}

location_idx_t leg_location(timetable const& tt,
                            routing::journey::leg const& l,
                            bool const is_exit) {
  auto const& ree = std::get<routing::journey::run_enter_exit>(l.uses_);
  return routing::search_location(
      kDefaultProfile,
      rt::frun{
          tt, nullptr,
          ree.r_}[is_exit ? static_cast<stop_idx_t>(ree.stop_range_.to_ - 1U)
                          : ree.stop_range_.from_]);
}

pareto_set<routing::journey> search_to_o(
    timetable const& tt,
    location_idx_t const from,
    std::string_view const hhmm,
    routing::transfer_time_settings const tts) {
  return raptor_search(
      tt, nullptr,
      routing::query{.start_time_ = at(hhmm),
                     .use_start_footpaths_ = true,
                     .start_ = {{from, 0_minutes, 0U}},
                     .destination_ = {{lidx(tt, "O"), 0_minutes, 0U}},
                     .transfer_time_settings_ = tts});
}

// ===========================================================================
// Transfer time settings and transfers of weight 0 (zero_min_transfer_feed).
// Without settings, GB (2 min after the arrival at W) is reachable. With a
// min_transfer_time_ of 5 min or 3 min additional_time_, every change at W
// takes longer than 2 min: GB2 (10 min) is the next. The same holds for GA,
// which arrives at its virtual location and changes through W's 0 min transfer,
// for GC, which arrives at W itself, and for a start where GD leaves (as a leg
// alternative or an itinerary refresh starts) - and in each case the search
// and the reconstruction have to agree on what the change costs.
// ===========================================================================

TEST(transfer_rules, transfer_time_settings_apply_to_zero_min_transfer) {
  auto const tt = load_feeds({zero_min_transfer_feed()});
  ASSERT_LT(0U, n_virts(tt));

  auto const gd = search_at(tt, "W", "P", "10:30");
  ASSERT_EQ(1U, gd.size());
  // The location the trip of the journey's first transit leg leaves from.
  auto const& gd_legs = begin(gd)->legs_;
  auto const gd_leg = utl::find_if(gd_legs, test::is_transit);
  ASSERT_NE(end(gd_legs), gd_leg);
  auto const gd_start = leg_location(tt, *gd_leg, false);
  ASSERT_NE(lidx(tt, "W"), gd_start);

  auto const starts = std::vector<std::pair<location_idx_t, std::string_view>>{
      {lidx(tt, "N"), "10:00"}, {lidx(tt, "N2"), "10:00"}, {gd_start, "10:30"}};
  for (auto const& [from, hhmm] : starts) {
    SCOPED_TRACE(to_idx(from));
    auto const arrival = [&](routing::transfer_time_settings const tts) {
      return nigiri::test::arrival(search_to_o(tt, from, hhmm, tts));
    };
    EXPECT_EQ(at("11:00"), arrival({}));
    EXPECT_EQ(at("11:10"),
              arrival({.default_ = false, .min_transfer_time_ = 5_minutes}));
    EXPECT_EQ(at("11:10"),
              arrival({.default_ = false, .additional_time_ = 3_minutes}));
  }

  // The start walk to W takes 5 min like any transfer.
  auto const res =
      search_to_o(tt, gd_start, "10:30",
                  {.default_ = false, .min_transfer_time_ = 5_minutes});
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("10:35"), begin(res)->legs_.front().arr_time_);
}

// ===========================================================================
// Consumers of the footpaths and transfer times besides RAPTOR.
// ===========================================================================

// Time-dependent walks are routed between stops: every location of the stop
// walks to every location of the target, and to the stop's other locations in
// its transfer time (virt_change_feed).
TEST(transfer_rules, td_footpaths_reach_virtual_locations) {
  auto const tt = load_feeds({virt_change_feed()});
  auto const members = [&](location_idx_t const l) {
    auto m = std::vector<location_idx_t>{l};
    tt.locations_.for_each_virt(
        l, [&](location_idx_t const v) { m.push_back(v); });
    return m;
  };
  auto const qs = lidx(tt, "QS");
  auto const qb = lidx(tt, "QB");
  ASSERT_LT(1U, members(qs).size()) << "precondition";

  auto fps = std::map<location_idx_t, std::vector<td_footpath>>{
      {qs, {{qb, kNull, 5_minutes}}}};
  expand_td_footpaths_to_virts(tt, fps);

  EXPECT_EQ(members(qs).size(), fps.size());
  for (auto const from : members(qs)) {
    auto expected = std::vector<td_footpath>{};
    for (auto const to : members(qb)) {
      expected.push_back({to, kNull, 5_minutes});
    }
    for (auto const to : members(qs)) {
      if (to != from) {
        expected.push_back(
            {to, kNull, duration_t{tt.locations_.transfer_time_[qs]}});
      }
    }
    utl::sort(expected);
    EXPECT_EQ(expected, fps.at(from));
  }
}

// An exact start or destination (the default match mode) includes where the
// trips at the stop stop, as it includes its real-time locations: Q1 arrives
// at, and Q2 leaves from, QS without a change at QS.
TEST(transfer_rules, exact_match_includes_virtual_locations) {
  auto const tt = load_feeds({virt_change_feed()});
  ASSERT_LT(0U, n_virts(tt));

  EXPECT_EQ(at("10:30"), arrival_at(tt, "QA", "QS", "10:00"));
  EXPECT_EQ(at("11:00"), arrival_at(tt, "QS", "QB", "10:40"));

  auto const qb =
      test::one_to_all(tt, nullptr, direction::kForward, "QS", "QB", "10:40")
          .fastest_;
  EXPECT_EQ(20, qb.duration_);  // arrival 11:00
  EXPECT_EQ(1U, qb.k_);
}

// A rule for a trip pair holds for the riders of its trip: F -> Z changes from
// S1 to S2 in 1 min (type 1: 0 min), a walk takes 2 min. A journey that starts
// at S1 or ends at S2 walks, though an exact start or destination includes the
// virtual locations F and Z stop at.
TEST(transfer_rules, trip_pair_rule_walk_holds_for_its_trip) {
  for (auto const rule : {"S1,S2,2,60,RF,RZ,,\n", "S1,S2,1,,RF,RZ,,\n"}) {
    auto const tt = test::load_network(
        fmt::format("S1,S2,2,120,,,,\n{}", rule),
        {.stops_ = {{"E", 50.0, 9.0}},
         .trips_ = {{"Z", "RZ", {{"S2", "10:31"}, {"E", "11:00"}}},
                    {"ZL", "RZL", {{"S2", "11:01"}, {"E", "11:30"}}}}});
    ASSERT_LT(0U, n_virts(tt));
    auto const search =
        [&](std::string_view const from, std::string_view const to,
            interval<unixtime_t> const times, direction const dir) {
          auto q = test::exact_query(tt, from, to, times);
          q.use_start_footpaths_ = true;
          auto const res = raptor_search(tt, nullptr, std::move(q), dir);
          EXPECT_EQ(1U, res.size())
              << rule << test::print_results(tt, nullptr, res);
          return res.empty() ? std::pair<unixtime_t, unixtime_t>{}
                             : std::pair{res.begin()->departure_time(),
                                         res.begin()->arrival_time()};
        };
    EXPECT_EQ(std::pair(at("10:00"), at("11:00")),
              search("A", "E", {at("10:00"), at("10:01")}, direction::kForward))
        << rule;
    EXPECT_EQ(
        std::pair(at("10:59"), at("11:30")),
        search("S1", "E", {at("10:30"), at("11:00")}, direction::kForward))
        << rule;
    EXPECT_EQ(
        std::pair(at("10:59"), at("11:30")),
        search("E", "S1", {at("11:30"), at("11:31")}, direction::kBackward))
        << rule;
    EXPECT_EQ(
        std::pair(at("10:00"), at("10:32")),
        search("A", "S2", {at("10:00"), at("10:01")}, direction::kForward))
        << rule;

    // one-to-all as well
    EXPECT_EQ(32, test::one_to_all(tt, nullptr, direction::kForward, "A", "S2",
                                   "10:00")
                      .fastest_.duration_)
        << rule;
  }
}

// The same for direct connections between stations: F -> Z changes from S1 to
// Y1 in 5 min, a walk takes 15 min.
TEST(transfer_rules, direct_trip_pair_rule_walk_holds_for_its_trip) {
  auto const tt = test::load_network(
      "S1,Y1,2,900,,,,\nS1,Y1,2,300,RF,RZ,,\n",
      {.stops_ = {{"Y", 50.0, 6.51, "", true},
                  {"Y1", 50.0001, 6.51, "Y"},
                  {"E", 50.0, 9.0}},
       .trips_ = {{"Z", "RZ", {{"Y1", "10:40"}, {"E", "11:00"}}},
                  {"ZL", "RZL", {{"Y1", "11:10"}, {"E", "11:30"}}}}});
  ASSERT_LT(0U, n_virts(tt));
  EXPECT_EQ(at("11:00"),
            test::arrival(tt, nullptr, {"A", "E"}, "10:00"));  // F -> Z

  auto const first_direct = [&](std::string_view const from,
                                std::string_view const to,
                                std::string_view const hhmm) {
    auto q = test::station_query(tt, from, to, at(hhmm));
    q.use_start_footpaths_ = true;
    auto first = std::optional<unixtime_t>{};
    for (auto&& legs : routing::get_direct_journeys<direction::kForward>(
             tt, nullptr, q, at(hhmm))) {
      if (!first.has_value() || legs.back().arr_time_ < *first) {
        first = legs.back().arr_time_;
      }
    }
    return first;
  };
  EXPECT_EQ(at("11:30"), first_direct("S", "E", "10:30"));
  EXPECT_EQ(at("10:45"), first_direct("A", "Y", "10:00"));
  EXPECT_EQ(at("10:45"), test::arrival(tt, nullptr, {"A", "Y"}, "10:00"));
}

// One-to-all reaches D, where changing is forbidden, at the arrival (other
// stops include their time to change), but nothing beyond it. Its users take
// the best times for what is reached.
TEST(transfer_rules, one_to_all_reaches_stop_without_changes) {
  auto const tt = load_feeds(
      {feed({{"A", 55.0, 13.0}, {"D", 55.1, 13.0}, {"E", 55.2, 13.0}},
            {{"T1", "R1", {{"A", "10:00"}, {"D", "10:30"}}},
             {"T2", "R2", {{"D", "10:40"}, {"E", "11:00"}}}},
            "D,D,3,,,,,\n")});
  auto const fastest = [&](direction const dir, std::string_view const from,
                           std::string_view const to, std::string_view time) {
    auto const [f, state] = test::one_to_all(tt, nullptr, dir, from, to, time);
    auto const invalid = dir == direction::kForward
                             ? kInvalidDelta<direction::kForward>
                             : kInvalidDelta<direction::kBackward>;
    EXPECT_EQ(f.k_ != routing::fastest_offset{}.k_,
              state.get_best<0U>()[to_idx(lidx(tt, to))][0] != invalid)
        << from << " -> " << to;
    return f;
  };
  EXPECT_EQ(30, fastest(direction::kForward, "A", "D", "10:00").duration_);
  EXPECT_EQ(routing::fastest_offset{}.k_,
            fastest(direction::kForward, "A", "E", "10:00").k_);
  EXPECT_EQ(-20, fastest(direction::kBackward, "E", "D", "11:00").duration_);
  EXPECT_EQ(routing::fastest_offset{}.k_,
            fastest(direction::kBackward, "E", "A", "11:00").k_);
}

// One-to-all reaches a stop at the arrival, its virtual locations are no
// stops of their own: FA arrives at, and FB leaves from, virtual locations of
// U. The RF1 -> RF2 rule (1 min) holds for changing to RF2 only.
TEST(transfer_rules, one_to_all_reaches_stop_at_the_arrival) {
  auto const tt = load_feeds(
      {feed({{"L", 55.0, 13.0}, {"U", 55.1, 13.0}, {"M", 55.2, 13.0}},
            {{"FA", "RF1", {{"L", "10:00"}, {"U", "10:30"}}},
             {"FB", "RF2", {{"U", "10:31"}, {"M", "11:00"}}}},
            "U,U,2,120,,,,\n"
            "U,U,2,60,RF1,RF2,,\n")});
  ASSERT_LT(0U, n_virts(tt));
  auto const one_to_all = [&](direction const dir, std::string_view const from,
                              std::string_view const time) {
    auto const [u, state] = test::one_to_all(tt, nullptr, dir, from, "U", time);
    auto const invalid = dir == direction::kForward
                             ? kInvalidDelta<direction::kForward>
                             : kInvalidDelta<direction::kBackward>;
    for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
      if (tt.locations_.is_virt(l)) {
        EXPECT_EQ(invalid, state.get_best<0U>()[to_idx(l)][0]) << loc{tt, l};
      }
    }
    return std::pair{u.duration_, u.k_};
  };
  // arrival 10:30
  EXPECT_EQ(std::pair(delta_t{30}, std::uint8_t{1U}),
            one_to_all(direction::kForward, "L", "10:00"));
  // departure 10:31
  EXPECT_EQ(std::pair(delta_t{-29}, std::uint8_t{1U}),
            one_to_all(direction::kBackward, "M", "11:00"));
}

// One-to-all reaches a stop that bans changes at the arrival (departure), also
// where its trips stop at virtual locations for an exception: S1 bans changes
// except RF -> RH (1 min), F arrives at 10:30, H leaves at 10:31.
TEST(transfer_rules, one_to_all_reaches_stop_without_changes_by_exception) {
  auto const tt = test::load_network("S1,S1,3,,,,,\nS1,S1,2,60,RF,RH,,");
  ASSERT_LT(0U, n_virts(tt));
  auto const fastest = [&](direction const dir, std::string_view const from,
                           std::string_view const to, std::string_view time) {
    auto const f = test::one_to_all(tt, nullptr, dir, from, to, time).fastest_;
    return std::pair{f.duration_, f.k_};
  };
  EXPECT_EQ(std::pair(delta_t{30}, std::uint8_t{1U}),
            fastest(direction::kForward, "A", "S1", "10:00"));
  // H arrives at 11:00
  EXPECT_EQ(std::pair(delta_t{60}, std::uint8_t{2U}),
            fastest(direction::kForward, "A", "C", "10:00"));
  EXPECT_EQ(std::pair(delta_t{-29}, std::uint8_t{1U}),
            fastest(direction::kBackward, "C", "S1", "11:00"));
}

// A profile changes at a virtual location in its transfer time, as the
// default profile does (between the locations of a stop: the stop's own time).
// T0 and T1 (route R) change at S in 1 min through the R -> R rule, at X in
// X's own 2 min: every profile changes at S. FA (RF1) -> FC (RF3) at their
// location of U (3 min) is too short for the RF1 -> RF3 rule (5 min), not for
// U's 2 min.
TEST(transfer_rules, profile_changes_in_virtual_location_transfer_time) {
  constexpr auto const kProfile = profile_idx_t{1U};
  auto tt = load_feeds(
      {feed({{"A", 64.0, 23.0},
             {"S", 64.1, 23.0},
             {"X", 64.2, 23.0},
             {"E", 64.3, 23.0}},
            {{"T0", "R", {{"A", "10:00"}, {"S", "10:10"}, {"X", "10:20"}}},
             {"T1", "R", {{"S", "10:11"}, {"X", "10:25"}, {"E", "10:40"}}}},
            "S,S,2,120,,,,\n"
            "S,S,2,60,R,R,,\n")});
  ASSERT_LT(0U, n_virts(tt));
  add_empty_profile(tt, kProfile);

  auto const changes_at = [&](profile_idx_t const prf) {
    auto q = test::exact_query(tt, "A", "E", at("10:00"));
    q.prf_idx_ = prf;
    auto const res = raptor_search(tt, nullptr, std::move(q));
    EXPECT_EQ(1U, res.size());
    return res.size() == 0U ? location_idx_t::invalid()
                            : tt.base(begin(res)->legs_.front().to_);
  };
  EXPECT_EQ(lidx(tt, "S"), changes_at(0U));
  EXPECT_EQ(lidx(tt, "S"), changes_at(kProfile));

  auto u = load_feeds({virt_feed()});
  ASSERT_LT(0U, n_virts(u));
  add_empty_profile(u, kProfile);
  auto const n_journeys = [&](profile_idx_t const prf) {
    auto q = test::exact_query(u, "L", "N", at("10:00"));
    q.prf_idx_ = prf;
    return raptor_search(u, nullptr, std::move(q)).size();
  };
  EXPECT_EQ(0U, n_journeys(0U));
  EXPECT_EQ(0U, n_journeys(kProfile));
}

std::string recommended_transfer_feed(std::string_view const transfers) {
  return feed(
      {{"A", 59.0, 10.0},
       {"X", 59.1, 10.0, "", true},
       {"X1", 59.1001, 10.0, "X"},
       {"X2", 59.1002, 10.0, "X"},
       {"S", 59.2, 10.0, "", true},
       {"S1", 59.2001, 10.0, "S"},
       {"S2", 59.2002, 10.0, "S"},
       {"E", 59.3, 10.0}},
      {{"T0", "R0", {{"A", "10:00"}, {"X1", "10:10"}, {"S1", "10:20"}}},
       {"T1", "R1", {{"X2", "10:27"}, {"S2", "10:30"}, {"E", "10:40"}}}},
      transfers);
}

// T0 -> T1 can change at station X (X1 -> X2, 15 min buffer) or at station S
// (S1 -> S2, 8 min): X, unless a recommended transfer (type 0) names S. That
// holds as well when S asks for 15 min and only the 1 min T0 -> T1 rule
// allows the change there.
TEST(transfer_rules, recommended_transfer_at_virtual_locations) {
  auto const changes_at = [](std::string_view const transfers) {
    auto const tt = load_feeds({recommended_transfer_feed(transfers)});
    if (transfers.find("T0,T1") != std::string_view::npos) {
      EXPECT_LT(0U, n_virts(tt));
    }
    auto const res = search_at(tt, "A", "E", "10:00");
    EXPECT_EQ(1U, res.size());
    return res.size() == 0U
               ? std::string{}
               : std::string{
                     tt.locations_.ids_[tt.base(begin(res)->legs_.front().to_)]
                         .view()};
  };
  EXPECT_EQ("X1", changes_at(""));
  EXPECT_EQ("S1", changes_at("S,S,0,,,,,\n"));
  EXPECT_EQ("S1", changes_at("S,S,0,,,,,\nS,S,2,900,,,,\nS,S,2,60,,,T0,T1\n"));
}

// FB2 (10:50) is an alternative for the FB leg of L -> FA -> FB -> M, FC
// (10:33) is not: the alternatives follow the rule of the change they
// replace (FA -> FC: 5 min). The walk from where FA arrives to U is found as
// motis' itinerary refresh looks it up between two transit legs.
TEST(transfer_rules, leg_alternatives_after_trip_at_virtual_location) {
  auto const tt = load_feeds({virt_feed()});
  ASSERT_LT(0U, n_virts(tt));
  auto const q = test::exact_query(tt, "L", "M", at("10:00"));
  auto const res = raptor_search(tt, nullptr, routing::query{q});
  ASSERT_EQ(1U, res.size());
  auto const& j = *begin(res);

  auto const& fa = j.legs_.front();
  ASSERT_TRUE(test::is_transit(fa));
  EXPECT_EQ(lidx(tt, "U"), fa.to_);
  auto const arrived_at = leg_location(tt, fa, true);
  ASSERT_NE(lidx(tt, "U"), arrived_at);
  EXPECT_TRUE(routing::lookup_footpath(
                  arrived_at, fa.arr_time_, routing::side::kAlighting, tt,
                  nullptr, routing::query{}, {{lidx(tt, "U"), 0_minutes, 0U}},
                  routing::location_match_mode::kExact, true)
                  .has_value());

  auto fb_leg = j.legs_.size();
  for (auto i = 0U; i != j.legs_.size(); ++i) {
    if (test::is_transit(j.legs_[i]) && j.legs_[i].dep_time_ == at("10:33")) {
      fb_leg = i;
    }
  }
  ASSERT_NE(j.legs_.size(), fb_leg);

  auto const alternatives =
      routing::get_leg_alternatives(tt, nullptr, q, j, fb_leg, 3U);
  auto const offers = [&](std::string_view const dep) {
    return utl::any_of(alternatives, [&](routing::journey const& alt) {
      return utl::any_of(alt.legs_, [&](routing::journey::leg const& l) {
        return test::is_transit(l) && l.dep_time_ == at(dep);
      });
    });
  };
  EXPECT_TRUE(offers("10:50"));
  EXPECT_FALSE(offers("10:33"));
}

// The metrics count the stops a feed has, not their virtual locations.
TEST(transfer_rules, metrics_count_stops) {
  auto const tt = load_feeds({virt_feed()});
  ASSERT_LT(0U, n_virts(tt));
  EXPECT_EQ(4U, get_metrics(tt).feeds_.front().locations_);
}

// Door to door, the direct connections L -> U (FA arrives at a location of
// its own) and U -> N (FC leaves from one) exist as in the search, and their
// offsets name the query's stops.
TEST(transfer_rules, direct_intermodal_reaches_virtual_locations) {
  auto const tt = load_feeds({virt_feed()});
  ASSERT_LT(0U, n_virts(tt));
  auto const direct = [&](std::string_view const from,
                          std::string_view const to) {
    auto const q = routing::query{
        .start_time_ = at("09:55"),
        .start_match_mode_ = routing::location_match_mode::kIntermodal,
        .dest_match_mode_ = routing::location_match_mode::kIntermodal,
        .start_ = {{lidx(tt, from), 1_minutes, 0U}},
        .destination_ = {{lidx(tt, to), 3_minutes, 0U}}};
    auto journeys = std::vector<std::vector<routing::journey::leg>>{};
    for (auto&& legs : routing::get_direct_journeys<direction::kForward>(
             tt, nullptr, q, at("09:55"))) {
      journeys.push_back(std::move(legs));
    }
    return journeys;
  };
  auto const offset_target = [](routing::journey::leg const& l) {
    return std::holds_alternative<routing::offset>(l.uses_)
               ? std::get<routing::offset>(l.uses_).target()
               : location_idx_t::invalid();
  };

  auto const to_u = direct("L", "U");
  ASSERT_EQ(1U, to_u.size());
  EXPECT_EQ(at("10:33"), to_u.front().back().arr_time_);
  EXPECT_EQ(lidx(tt, "U"), offset_target(to_u.front().back()));

  auto const from_u = direct("U", "N");
  ASSERT_EQ(1U, from_u.size());
  EXPECT_EQ(at("10:32"), from_u.front().front().dep_time_);
  EXPECT_EQ(lidx(tt, "U"), offset_target(from_u.front().front()));

  // An exact start at U includes the location FC leaves from.
  auto const exact = test::exact_query(tt, "U", "N", at("10:30"));
  auto n_fc = 0U;
  for (auto&& legs : routing::get_direct_journeys<direction::kForward>(
           tt, nullptr, exact, at("10:30"))) {
    n_fc += static_cast<unsigned>(
        utl::count_if(legs, [](routing::journey::leg const& l) {
          return test::is_transit(l) && l.dep_time_ == at("10:33");
        }));
  }
  EXPECT_EQ(1U, n_fc);
}

// S -> B via the start station S with a 10 min stay: the stay holds at its
// stop S1 as well. From 10:00, Y (10:05) is too early, Y2 (10:20) is not.
//
// Via = start with a stay: X arrives at S 3 min before Y leaves, the RX -> RY
// rule (5 min) forbids it, so Y leaves from a location of its own. From S at
// 10:00 with a 10 min stay at S, Y (10:05) is too early, Y2 (10:20) is not -
// at every location of S.
TEST(transfer_rules, via_stay_at_start_holds_at_every_location) {
  struct layout {
    std::string feed_;
    routing::location_match_mode match_mode_;
    bool has_virts_;
  };
  for (auto const& [f, match_mode, has_virts] : std::vector<layout>{
           {feed({{"S", 55.0, 13.0, "", true},
                  {"S1", 55.0001, 13.0, "S"},
                  {"B", 55.2, 13.0}},
                 {{"Y", "RY", {{"S1", "10:05"}, {"B", "10:30"}}},
                  {"Y2", "RY", {{"S1", "10:20"}, {"B", "10:45"}}}},
                 ""),
            routing::location_match_mode::kEquivalent, false},
           {feed({{"S", 55.0, 13.0}, {"A", 55.1, 13.0}, {"B", 55.2, 13.0}},
                 {{"X", "RX", {{"A", "09:00"}, {"S", "10:02"}}},
                  {"Y", "RY", {{"S", "10:05"}, {"B", "10:30"}}},
                  {"Y2", "RY2", {{"S", "10:20"}, {"B", "10:45"}}}},
                 "S,S,2,120,,,,\nS,S,2,300,RX,RY,,\n"),
            routing::location_match_mode::kExact, true}}) {
    auto const tt = load_feeds({f});
    if (has_virts) {
      ASSERT_LT(0U, n_virts(tt));
    }
    auto q = test::exact_query(tt, "S", "B", at("10:00"));
    q.start_match_mode_ = match_mode;
    q.dest_match_mode_ = match_mode;
    q.via_stops_ = {{lidx(tt, "S"), 10_minutes}};
    auto const res = raptor_search(tt, nullptr, std::move(q));
    ASSERT_EQ(1U, res.size());
    EXPECT_EQ(at("10:45"), begin(res)->dest_time_);
  }
}

// A footpath leg between two transport legs shows the time the rules state
// for the two transports, whatever the footpath of their virtual locations
// says: T0 -> T1 at S is their 1 min rule. The leg stays anchored to T0's
// arrival, the wait before T1 absorbs the difference.
TEST(transfer_rules, journey_shows_rule_transfer_time) {
  auto const tt = load_feeds({recommended_transfer_feed(
      "S,S,0,,,,,\nS,S,2,900,,,,\nS,S,2,60,,,T0,T1\n")});
  ASSERT_LT(0U, n_virts(tt));
  auto const res = search_at(tt, "A", "E", "10:00");
  ASSERT_EQ(1U, res.size());
  auto j = *begin(res);
  ASSERT_EQ(3U, j.legs_.size());
  auto& fp = j.legs_[1];
  ASSERT_TRUE(std::holds_alternative<footpath>(fp.uses_));
  EXPECT_EQ("S1", tt.locations_.ids_[tt.base(fp.from_)].view());
  fp.uses_ = footpath{fp.to_, 7_minutes};
  fp.arr_time_ = fp.dep_time_ + 7_minutes;
  routing::fix_transfer_durations(tt, nullptr, routing::query{}, j);
  EXPECT_EQ(1_minutes, std::get<footpath>(j.legs_[1].uses_).duration());
  EXPECT_EQ(j.legs_[0].arr_time_, j.legs_[1].dep_time_);
  EXPECT_EQ(j.legs_[1].dep_time_ + 1_minutes, j.legs_[1].arr_time_);
}

// Between the trips before and after it, the ping-pong search replaces a
// transit leg by the earliest alternative: a direct connection from where the
// trip before arrives to where the trip after leaves. After T0 (X 10:10) the
// earliest X -> S trip is T1b (10:13) with X's 2 min, T1 (10:20) when X asks
// for 5 min, T1b again when the T0 -> T1b rule asks for 1 min, and T1 when a
// rule forbids T0 -> T1b at X or T1b -> T2 at S.
TEST(transfer_rules, pong_alternative_follows_rules_at_virtual_locations) {
  auto const alternative_departs = [](std::string_view const transfers) {
    auto const tt =
        load_feeds({feed({{"A", 60.0, 10.0},
                          {"X", 60.1, 10.0},
                          {"S", 60.2, 10.0},
                          {"E", 60.3, 10.0}},
                         {{"T0", "R0", {{"A", "10:00"}, {"X", "10:10"}}},
                          {"T1", "R1", {{"X", "10:20"}, {"S", "10:30"}}},
                          {"T1b", "R1", {{"X", "10:13"}, {"S", "10:23"}}},
                          {"T2", "R2", {{"S", "10:40"}, {"E", "10:50"}}}},
                         transfers)});
    auto const res = search_at(tt, "A", "E", "10:00");
    EXPECT_EQ(1U, res.size());
    if (res.size() != 1U) {
      return unixtime_t{};
    }
    auto const& legs = begin(res)->legs_;
    auto const& before = std::get<routing::journey::run_enter_exit>(
        utl::find_if(legs, test::is_transit)->uses_);
    auto const& after = std::get<routing::journey::run_enter_exit>(
        std::find_if(rbegin(legs), rend(legs), test::is_transit)->uses_);
    auto const run_before = rt::frun{tt, nullptr, before.r_};
    auto const run_after = rt::frun{tt, nullptr, after.r_};
    auto const from =
        run_before[static_cast<stop_idx_t>(before.stop_range_.to_ - 1U)];
    auto const to = run_after[after.stop_range_.from_];
    auto const earlier = routing::get_earliest_alternative(
        tt, nullptr, routing::query{},
        routing::search_location(kDefaultProfile, from),
        routing::search_location(kDefaultProfile, to),
        from.time(event_type::kArr), to.time(event_type::kDep));
    EXPECT_TRUE(earlier.has_value());
    return earlier.has_value() ? earlier->at(1).dep_time_ : unixtime_t{};
  };
  EXPECT_EQ(at("10:13"), alternative_departs(""));
  EXPECT_EQ(at("10:20"), alternative_departs("X,X,2,300,,,,\n"));
  EXPECT_EQ(at("10:13"),
            alternative_departs("X,X,2,300,,,,\nX,X,2,60,,,T0,T1b\n"));
  EXPECT_EQ(at("10:20"), alternative_departs("X,X,3,,,,T0,T1b\n"));
  EXPECT_EQ(at("10:20"), alternative_departs("S,S,3,,,,T1b,T2\n"));
}

// The footpath of a virtual location answers which changes work, not how long
// they take: at X one footpath serves T0 -> T1 (1 min by rule), T0b -> T1b (2
// min by rule) and T0b -> T1c (X's 5 min), so the search sees one duration for
// all three.
std::string fold_feed() {
  return feed({{"A", 61.0, 10.0},
               {"X", 61.1, 10.0},
               {"S", 61.2, 10.0},
               {"E", 61.3, 10.0}},
              {{"T0", "R0", {{"A", "10:00"}, {"X", "10:10"}}},
               {"T0b", "R0", {{"A", "11:00"}, {"X", "11:10"}}},
               {"T1", "R1", {{"X", "10:13"}, {"S", "10:23"}}},
               {"T1b", "R1", {{"X", "11:12"}, {"S", "11:22"}}},
               {"T1c", "R1", {{"X", "11:20"}, {"S", "11:30"}}},
               {"T2", "R2", {{"S", "10:40"}, {"E", "10:50"}}},
               {"T2b", "R2", {{"S", "11:40"}, {"E", "11:50"}}}},
              "X,X,2,300,,,,\nX,X,2,60,,,T0,T1\nX,X,2,120,,,T0b,T1b\n");
}

// The journeys show each rule's time, in the ping-pong search (which rebuilds
// the middle leg from a direct connection) as well.
TEST(transfer_rules, journeys_show_rule_transfer_times) {
  auto const tt = load_feeds({fold_feed()});
  ASSERT_LT(0U, n_virts(tt));
  auto const q = test::exact_query(
      tt, "A", "E", interval<unixtime_t>{at("10:00"), at("12:00")});
  for (auto const& res : {raptor_search(tt, nullptr, routing::query{q}),
                          search_pong(tt, nullptr, routing::query{q})}) {
    ASSERT_EQ(2U, res.size());
    auto const& first = *begin(res);
    auto const& second = *std::next(begin(res));
    ASSERT_EQ(5U, first.legs_.size());
    ASSERT_EQ(5U, second.legs_.size());
    // One footpath serves both changes at X.
    EXPECT_EQ(leg_location(tt, first.legs_[0], true),
              leg_location(tt, second.legs_[0], true));
    EXPECT_EQ(leg_location(tt, first.legs_[2], false),
              leg_location(tt, second.legs_[2], false));
    auto const d = routing::shortest_transfer(
        tt, nullptr, kDefaultProfile, leg_location(tt, first.legs_[0], true),
        leg_location(tt, first.legs_[2], false));
    ASSERT_TRUE(d.has_value());
    EXPECT_LT(*d, 3_minutes);
    for (auto const& j : res) {
      auto const expected =
          j.start_time_ == at("10:00") ? 1_minutes : 2_minutes;
      auto const& fp = j.legs_[1];
      ASSERT_TRUE(std::holds_alternative<footpath>(fp.uses_));
      EXPECT_EQ(expected, std::get<footpath>(fp.uses_).duration());
      EXPECT_EQ(j.legs_[0].arr_time_, fp.dep_time_);
      EXPECT_EQ(fp.dep_time_ + expected, fp.arr_time_);
      ASSERT_TRUE(std::holds_alternative<footpath>(j.legs_[3].uses_));
      EXPECT_EQ(2_minutes, std::get<footpath>(j.legs_[3].uses_).duration());
    }
  }
}

// The alternatives of a leg are direct connections from where the trip before
// arrives to where the trip after leaves, their walks come from the search's
// footpaths. T1c is the alternative to T1b in the 11:00 journey; its walk from
// T0b shows X's 5 min (no rule for T0b -> T1c), not the shared footpath's
// duration, and ends when T1c leaves.
TEST(transfer_rules, leg_alternatives_show_rule_transfer_times) {
  auto const tt = load_feeds({fold_feed()});
  auto const q = test::exact_query(tt, "A", "E", at("11:00"));
  auto const res = raptor_search(tt, nullptr, routing::query{q});
  ASSERT_EQ(1U, res.size());
  auto const& j = *begin(res);
  ASSERT_EQ(5U, j.legs_.size());
  EXPECT_EQ(at("11:12"), j.legs_[2].dep_time_);
  auto const alternatives =
      routing::get_leg_alternatives(tt, nullptr, q, j, 2U, 3U);
  ASSERT_EQ(1U, alternatives.size());
  auto const& legs = alternatives.front().legs_;
  auto const transit = utl::find_if(legs, test::is_transit);
  ASSERT_NE(end(legs), transit);
  ASSERT_NE(begin(legs), transit);
  ASSERT_NE(end(legs), std::next(transit));
  EXPECT_EQ(at("11:20"), transit->dep_time_);
  auto const& walk_before = *std::prev(transit);
  ASSERT_TRUE(std::holds_alternative<footpath>(walk_before.uses_));
  EXPECT_EQ(5_minutes, std::get<footpath>(walk_before.uses_).duration());
  EXPECT_EQ(at("11:15"), walk_before.dep_time_);
  EXPECT_EQ(at("11:20"), walk_before.arr_time_);
  auto const& walk_after = *std::next(transit);
  ASSERT_TRUE(std::holds_alternative<footpath>(walk_after.uses_));
  EXPECT_EQ(2_minutes, std::get<footpath>(walk_after.uses_).duration());
  EXPECT_EQ(walk_after.dep_time_ + 2_minutes, walk_after.arr_time_);
}

TEST(transfer_rules,
     fastest_direct_ignores_rule_footpaths_of_virtual_locations) {
  auto const tt =
      load_feeds({feed({{"A", 55.0, 13.0},
                        {"X", 55.1, 13.0},
                        {"Y", 55.11, 13.0},
                        {"E", 55.3, 13.0}},
                       {{"F", "RF", {{"A", "09:00"}, {"X", "09:30"}}},
                        {"P", "RP", {{"A", "09:05"}, {"X", "09:35"}}},
                        {"B", "RB", {{"X", "10:00"}, {"Y", "10:15"}}},
                        {"Q", "RQ", {{"Y", "10:20"}, {"E", "10:50"}}}},
                       "X,Y,2,1800,,,,\nX,Y,2,120,RF,,,\n")});
  ASSERT_LT(0U, n_virts(tt));
  auto const q = test::exact_query(tt, "X", "Y", at("10:00"));
  EXPECT_EQ(30_minutes,
            routing::get_fastest_direct(tt, q, direction::kForward));
  EXPECT_EQ(at("10:15"), nigiri::test::arrival(raptor_search(tt, nullptr, q)));
}

TEST(transfer_rules, one_to_all_reaches_stop_at_the_arrival_with_rule_to_any) {
  auto const load = [](std::string_view const rule) {
    return load_feeds(
        {feed({{"L", 55.0, 13.0}, {"U", 55.1, 13.0}, {"M", 55.2, 13.0}},
              {{"FA", "RF1", {{"L", "10:00"}, {"U", "10:30"}}},
               {"FB", "RF2", {{"U", "10:31"}, {"M", "11:00"}}},
               {"FC", "RF3", {{"U", "10:35"}, {"M", "11:05"}}}},
              std::string{"U,U,2,120,,,,\n"} + std::string{rule})});
  };
  auto const fastest =
      [](timetable const& tt, direction const dir, std::string_view const from,
         std::string_view const to, std::string_view const time) {
        return test::one_to_all(tt, nullptr, dir, from, to, time)
            .fastest_.duration_;
      };
  constexpr auto const kUnreachable = routing::fastest_offset{}.duration_;

  auto const plain = load("");
  EXPECT_EQ(delta_t{30},
            fastest(plain, direction::kForward, "L", "U", "10:00"));
  EXPECT_NE(kUnreachable,
            fastest(plain, direction::kForward, "L", "M", "10:00"));

  for (auto const rule : {"U,U,3,,RF1,,,\n", "U,U,2,600,RF1,,,\n"}) {
    auto const tt = load(rule);
    ASSERT_LT(0U, n_virts(tt)) << rule;
    EXPECT_EQ(delta_t{30}, fastest(tt, direction::kForward, "L", "U", "10:00"))
        << rule;
    EXPECT_EQ(delta_t{-29},
              fastest(tt, direction::kBackward, "M", "U", "11:00"))
        << rule;
    // FB (10:31) and FC (10:35) leave before the rule allows a change.
    EXPECT_EQ(kUnreachable, fastest(tt, direction::kForward, "L", "M", "10:00"))
        << rule;
  }
}
