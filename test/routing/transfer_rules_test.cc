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
          auto const res = raptor_search(
              tt, nullptr,
              routing::query{.start_time_ = times,
                             .use_start_footpaths_ = true,
                             .start_ = {{lidx(tt, from), 0_minutes, 0U}},
                             .destination_ = {{lidx(tt, to), 0_minutes, 0U}}},
              dir);
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
    auto const q = routing::query{.start_time_ = at("10:00"),
                                  .start_ = {{lidx(tt, "A"), 0_minutes, 0U}}};
    auto const state = routing::one_to_all<direction::kForward>(tt, nullptr, q);
    EXPECT_EQ(32, routing::get_fastest_one_to_all_offsets(
                      tt, state, direction::kForward, lidx(tt, "S2"),
                      at("10:00"), q.max_transfers_)
                      .duration_)
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
    auto const q = routing::query{.start_time_ = at(time),
                                  .start_ = {{lidx(tt, from), 0_minutes, 0U}}};
    auto const state =
        dir == direction::kForward
            ? routing::one_to_all<direction::kForward>(tt, nullptr, q)
            : routing::one_to_all<direction::kBackward>(tt, nullptr, q);
    auto const f = routing::get_fastest_one_to_all_offsets(
        tt, state, dir, lidx(tt, to), at(time), q.max_transfers_);
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

// One-to-all reaches a stop in its own time, its virtual locations are no
// stops of their own: FA arrives at, and FB leaves from, virtual locations of
// U. The RF1 -> RF2 rule (1 min) holds for changing to RF2 only, U itself takes
// 2 min.
TEST(transfer_rules, one_to_all_reaches_stop_in_its_own_time) {
  auto const tt = load_feeds(
      {feed({{"L", 55.0, 13.0}, {"U", 55.1, 13.0}, {"M", 55.2, 13.0}},
            {{"FA", "RF1", {{"L", "10:00"}, {"U", "10:30"}}},
             {"FB", "RF2", {{"U", "10:31"}, {"M", "11:00"}}}},
            "U,U,2,120,,,,\n"
            "U,U,2,60,RF1,RF2,,\n")});
  ASSERT_LT(0U, n_virts(tt));
  auto const one_to_all = [&](direction const dir, std::string_view const from,
                              std::string_view const time) {
    auto const q = routing::query{.start_time_ = at(time),
                                  .start_ = {{lidx(tt, from), 0_minutes, 0U}}};
    auto state =
        dir == direction::kForward
            ? routing::one_to_all<direction::kForward>(tt, nullptr, q)
            : routing::one_to_all<direction::kBackward>(tt, nullptr, q);
    auto const u = routing::get_fastest_one_to_all_offsets(
        tt, state, dir, lidx(tt, "U"), at(time), q.max_transfers_);
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
  // arrival 10:30 + U's 2 min
  EXPECT_EQ(std::pair(delta_t{32}, std::uint8_t{1U}),
            one_to_all(direction::kForward, "L", "10:00"));
  // departure 10:31 - U's 2 min
  EXPECT_EQ(std::pair(delta_t{-31}, std::uint8_t{1U}),
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
    auto const q = routing::query{.start_time_ = at(time),
                                  .start_ = {{lidx(tt, from), 0_minutes, 0U}}};
    auto const state =
        dir == direction::kForward
            ? routing::one_to_all<direction::kForward>(tt, nullptr, q)
            : routing::one_to_all<direction::kBackward>(tt, nullptr, q);
    auto const f = routing::get_fastest_one_to_all_offsets(
        tt, state, dir, lidx(tt, to), at(time), q.max_transfers_);
    return std::pair{f.duration_, f.k_};
  };
  EXPECT_EQ(std::pair(delta_t{30}, std::uint8_t{1U}),
            fastest(direction::kForward, "A", "S1", "10:00"));
  // H arrives at 11:00, C's 2 min
  EXPECT_EQ(std::pair(delta_t{62}, std::uint8_t{2U}),
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
  EXPECT_EQ(lidx(tt, "S"), changes_at(kProfile));

  auto u = load_feeds({virt_feed()});
  ASSERT_LT(0U, n_virts(u));
  add_empty_profile(u, kProfile);
  auto const n_journeys = [&](profile_idx_t const prf) {
    return raptor_search(
               u, nullptr,
               routing::query{.start_time_ = at("10:00"),
                              .start_ = {{lidx(u, "L"), 0_minutes, 0U}},
                              .destination_ = {{lidx(u, "N"), 0_minutes, 0U}},
                              .prf_idx_ = prf})
        .size();
  };
  EXPECT_EQ(0U, n_journeys(0U));
  EXPECT_EQ(0U, n_journeys(kProfile));
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

// FB2 (10:50) is an alternative for the FB leg of L -> FA -> FB -> M, FC
// (10:33) is not: the alternatives follow the rule of the change they
// replace (FA -> FC: 5 min). The walk from where FA arrives to U is found as
// motis' itinerary refresh looks it up between two transit legs.
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
  auto const offers = [&](std::string_view const dep) {
    return utl::any_of(alternatives, [&](routing::journey const& alt) {
      return utl::any_of(alt.legs_, [&](routing::journey::leg const& l) {
        return std::holds_alternative<routing::journey::run_enter_exit>(
                   l.uses_) &&
               l.dep_time_ == at(dep);
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
  auto const exact =
      routing::query{.start_time_ = at("10:30"),
                     .start_ = {{lidx(tt, "U"), 0_minutes, 0U}},
                     .destination_ = {{lidx(tt, "N"), 0_minutes, 0U}}};
  auto n_fc = 0U;
  for (auto&& legs : routing::get_direct_journeys<direction::kForward>(
           tt, nullptr, exact, at("10:30"))) {
    n_fc += static_cast<unsigned>(
        utl::count_if(legs, [](routing::journey::leg const& l) {
          return std::holds_alternative<routing::journey::run_enter_exit>(
                     l.uses_) &&
                 l.dep_time_ == at("10:33");
        }));
  }
  EXPECT_EQ(1U, n_fc);
}

// Via = start with a stay: X arrives at S 3 min before Y leaves, the RX -> RY
// rule (5 min) forbids it, so Y leaves from a location of its own. From S at
// 10:00 with a 10 min stay at S, Y (10:05) is too early, Y2 (10:20) is not -
// at every location of S.
TEST(transfer_rules, via_stay_at_start_holds_at_virtual_locations) {
  auto const tt = load_feeds(
      {feed({{"S", 55.0, 13.0}, {"A", 55.1, 13.0}, {"B", 55.2, 13.0}},
            {{"X", "RX", {{"A", "09:00"}, {"S", "10:02"}}},
             {"Y", "RY", {{"S", "10:05"}, {"B", "10:30"}}},
             {"Y2", "RY2", {{"S", "10:20"}, {"B", "10:45"}}}},
            "S,S,2,120,,,,\nS,S,2,300,RX,RY,,\n")});
  ASSERT_LT(0U, n_virts(tt));
  auto const res = raptor_search(
      tt, nullptr,
      routing::query{.start_time_ = at("10:00"),
                     .start_ = {{lidx(tt, "S"), 0_minutes, 0U}},
                     .destination_ = {{lidx(tt, "B"), 0_minutes, 0U}},
                     .via_stops_ = {{lidx(tt, "S"), 10_minutes}}});
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("10:45"), begin(res)->dest_time_);
}
