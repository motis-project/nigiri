#include "gtest/gtest.h"

#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "utl/helpers/algorithm.h"

#include "nigiri/loader/build_footpaths.h"
#include "nigiri/routing/direct.h"
#include "nigiri/routing/leg_alternatives.h"
#include "nigiri/routing/one_to_all.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/transfers.h"
#include "nigiri/rt/frun.h"
#include "nigiri/timetable.h"

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

// The location the trip of the journey's first transit leg leaves from.
location_idx_t first_departure(timetable const& tt, routing::journey const& j) {
  auto const leg = utl::find_if(j.legs_, [](routing::journey::leg const& l) {
    return std::holds_alternative<routing::journey::run_enter_exit>(l.uses_);
  });
  if (leg == end(j.legs_)) {
    return location_idx_t::invalid();
  }
  auto const& ree = std::get<routing::journey::run_enter_exit>(leg->uses_);
  return routing::search_location(
      kDefaultProfile, rt::frun{tt, nullptr, ree.r_}[ree.stop_range_.from_]);
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
  auto const gd_start = first_departure(tt, *begin(gd));
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
  loader::expand_td_footpaths_to_virts(tt, fps);

  EXPECT_EQ(members(qs).size(), fps.size());
  for (auto const from : members(qs)) {
    auto expected = std::vector<td_footpath>{};
    for (auto const to : members(qb)) {
      expected.push_back({to, kNull, 5_minutes});
    }
    for (auto const to : members(qs)) {
      if (to != from) {
        expected.push_back(
            {to, kNull, to_fp_duration(tt.locations_.transfer_time_[qs])});
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

  auto const q = routing::query{.start_time_ = at("10:40"),
                                .start_ = {{lidx(tt, "QS"), 0_minutes, 0U}}};
  auto const state = routing::one_to_all<direction::kForward>(tt, nullptr, q);
  auto const qb = routing::get_fastest_one_to_all_offsets(
      tt, state, direction::kForward, lidx(tt, "QB"), at("10:40"),
      q.max_transfers_);
  EXPECT_EQ(22, qb.duration_);  // arrival 11:00 + QB's 2 min to change
  EXPECT_EQ(1U, qb.k_);
}

// T0 and T1 (route R) change at S in 1 min through the R -> R rule, at X in
// X's own 2 min with 3 min to spare. A profile that ignores transfer rules
// changes at S in S's own 2 min, which T1 (1 min after T0) does not leave: the
// transfer optimization keeps X.
TEST(transfer_rules, profile_optimizes_transfers_with_stop_time) {
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
    auto const res = raptor_search(
        tt, nullptr,
        routing::query{.start_time_ = at("10:00"),
                       .start_ = {{lidx(tt, "A"), 0_minutes, 0U}},
                       .destination_ = {{lidx(tt, "E"), 0_minutes, 0U}},
                       .prf_idx_ = prf});
    EXPECT_EQ(1U, res.size());
    return res.size() == 0U ? location_idx_t::invalid()
                            : tt.base(begin(res)->legs_.front().to_);
  };
  EXPECT_EQ(lidx(tt, "S"), changes_at(0U));
  EXPECT_EQ(lidx(tt, "X"), changes_at(kProfile));
}

// T0 -> T1 can change at station X (X1 -> X2, 15 min buffer) or at station S
// (S1 -> S2, 8 min): X, unless a recommended transfer (type 0) names S. That
// holds as well when S asks for 15 min and only the 1 min T0 -> T1 rule
// allows the change there.
TEST(transfer_rules, recommended_transfer_at_virtual_locations) {
  auto const changes_at = [](std::string_view const transfers) {
    auto const tt = load_feeds({feed(
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
        transfers)});
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

// FB2 (10:50) is an alternative for the FB leg of L -> FA -> FB -> M. The
// walk from where FA arrives to U is found as motis' itinerary refresh looks
// it up between two transit legs.
TEST(transfer_rules, leg_alternatives_after_trip_at_virtual_location) {
  auto const tt = load_feeds({virt_feed()});
  ASSERT_LT(0U, n_virts(tt));
  auto const q =
      routing::query{.start_time_ = at("10:00"),
                     .start_ = {{lidx(tt, "L"), 0_minutes, 0U}},
                     .destination_ = {{lidx(tt, "M"), 0_minutes, 0U}}};
  auto const res = raptor_search(tt, nullptr, routing::query{q});
  ASSERT_EQ(1U, res.size());
  auto const& j = *begin(res);

  auto const& fa = j.legs_.front();
  ASSERT_TRUE(
      std::holds_alternative<routing::journey::run_enter_exit>(fa.uses_));
  EXPECT_EQ(lidx(tt, "U"), fa.to_);
  auto const& ree = std::get<routing::journey::run_enter_exit>(fa.uses_);
  auto const arrived_at = routing::search_location(
      kDefaultProfile, rt::frun{tt, nullptr, ree.r_}[ree.stop_range_.to_ - 1U]);
  ASSERT_NE(lidx(tt, "U"), arrived_at);
  EXPECT_TRUE(routing::lookup_footpath(
                  arrived_at, fa.arr_time_, routing::side::kAlighting, tt,
                  nullptr, routing::query{}, {{lidx(tt, "U"), 0_minutes, 0U}},
                  routing::location_match_mode::kExact, true)
                  .has_value());

  auto fb_leg = j.legs_.size();
  for (auto i = 0U; i != j.legs_.size(); ++i) {
    if (std::holds_alternative<routing::journey::run_enter_exit>(
            j.legs_[i].uses_) &&
        j.legs_[i].dep_time_ == at("10:33")) {
      fb_leg = i;
    }
  }
  ASSERT_NE(j.legs_.size(), fb_leg);

  auto const alternatives =
      routing::get_leg_alternatives(tt, nullptr, q, j, fb_leg, 3U);
  EXPECT_TRUE(utl::any_of(alternatives, [&](routing::journey const& alt) {
    return utl::any_of(alt.legs_, [&](routing::journey::leg const& l) {
      return std::holds_alternative<routing::journey::run_enter_exit>(
                 l.uses_) &&
             l.dep_time_ == at("10:50");
    });
  }));
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
}
