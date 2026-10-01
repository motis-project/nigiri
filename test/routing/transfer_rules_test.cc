#include "gtest/gtest.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "utl/helpers/algorithm.h"

#include "nigiri/routing/direct.h"
#include "nigiri/routing/for_each_hub_source.h"
#include "nigiri/routing/get_fastest_direct.h"
#include "nigiri/routing/leg_alternatives.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/search.h"
#include "nigiri/routing/tb/preprocess.h"
#include "nigiri/routing/tb/query_engine.h"
#include "nigiri/timetable.h"

#include "../raptor_search.h"
#include "../transfer_rules_util.h"

// Consumers of transfers.txt rules, virtual locations and hubs: transfer time
// settings, trip-based routing, offsets, leg alternatives, footpath lookups,
// direct connections and walks. What the rules mean is tested with the GTFS
// loader (test/loader/gtfs/transfer_rules_test.cc). All feeds run on
// 2019-05-01 in Europe/Berlin; the default transfer time at a stop is 2 min.

using namespace nigiri;
using nigiri::test::add_empty_profile;
using nigiri::test::at;
using nigiri::test::feed;
using nigiri::test::lidx;
using nigiri::test::load_feeds;
using nigiri::test::n_virts;
using nigiri::test::raptor_search;
using nigiri::test::search_at;

// The shared feed of the tests that change out of a trip at a virtual
// location: FA (route RF1) stops at a virtual location of U (the RF1 -> RF3
// rule), FB and FB2 (RF2) leave from U itself, FA -> FB is a plain change at
// U (2 min).
std::string virt_feed() {
  return feed({{"U", 55.0, 13.0}, {"L", 55.1, 13.0}, {"M", 55.2, 13.0}},
              {{"FA", "RF1", {{"L", "10:00"}, {"U", "10:30"}}},
               {"FB", "RF2", {{"U", "10:33"}, {"M", "11:00"}}},
               {"FB2", "RF2", {{"U", "10:50"}, {"M", "11:20"}}}},
              "U,U,2,120,,,,\n"
              "U,U,2,300,RF1,RF3,,\n",
              {"RF3"});
}

// W,W,1 makes W's own transfer time 0 min, and the RG1 -> RG3 rule gives GA a
// virtual location of W, so W gets a hub of weight 0.
std::string zero_min_hub_feed() {
  return feed({{"W", 56.0, 14.0},
               {"N", 56.1, 14.0},
               {"N2", 56.1, 14.2},
               {"O", 56.2, 14.0}},
              {{"GA", "RG1", {{"N", "10:00"}, {"W", "10:30"}}},
               {"GB", "RG2", {{"W", "10:32"}, {"O", "11:00"}}},
               {"GB2", "RG2", {{"W", "10:40"}, {"O", "11:10"}}},
               {"GC", "RG4", {{"N2", "10:00"}, {"W", "10:30"}}}},
              "W,W,1,,,,,\n"
              "W,W,2,300,RG1,RG3,,\n",
              {"RG3"});
}

unixtime_t arrival_at_o(timetable const& tt,
                        std::string_view const from,
                        routing::transfer_time_settings const tts) {
  auto const res = raptor_search(
      tt, nullptr,
      routing::query{.start_time_ = at("10:00"),
                     .start_ = {{lidx(tt, from), 0_minutes, 0U}},
                     .destination_ = {{lidx(tt, "O"), 0_minutes, 0U}},
                     .transfer_time_settings_ = tts});
  EXPECT_EQ(1U, res.size());
  return res.size() == 0U ? unixtime_t{} : begin(res)->dest_time_;
}

// Trip-based routing, forward, on the profile of the query.
pareto_set<routing::journey> tb_search(timetable const& tt, routing::query q) {
  auto const tbd = routing::tb::preprocess(tt, q.prf_idx_);
  auto search_state = routing::search_state{};
  auto algo_state = routing::tb::query_state{tt, tbd};
  return *(
      routing::search<direction::kForward, routing::tb::query_engine<false>>{
          tt, nullptr, search_state, algo_state, std::move(q)}
          .execute()
          .journeys_);
}

// ===========================================================================
// Transfer time settings and hubs of weight 0 (zero_min_hub_feed). Without
// settings, GB (2 min after the arrival at W) is reachable. With a
// min_transfer_time_ of 5 min or 3 min additional_time_, every change at W
// takes longer than 2 min: GB2 (10 min) is the next. The same holds for GA,
// which arrives at its virtual location and changes through W's 0 min hub,
// and for GC, which arrives at W itself - and in both cases the search and
// the reconstruction have to agree on what the change costs.
// ===========================================================================

TEST(transfer_rules, transfer_time_settings_apply_to_zero_min_hub) {
  auto const tt = load_feeds({zero_min_hub_feed()});
  for (auto const from : {"N", "N2"}) {
    SCOPED_TRACE(from);
    EXPECT_EQ(at("11:00"), arrival_at_o(tt, from, {}));
    EXPECT_EQ(at("11:10"), arrival_at_o(tt, from,
                                        {.default_ = false,
                                         .min_transfer_time_ = 5_minutes}));
    EXPECT_EQ(at("11:10"),
              arrival_at_o(tt, from,
                           {.default_ = false, .additional_time_ = 3_minutes}));
  }
}

// ===========================================================================
// Consumers of the transfer relation besides RAPTOR: the change FA (virtual
// location below U) -> FB (at U) exists only through U's hub.
// ===========================================================================

TEST(transfer_rules, trip_based_routing_sees_hub_transfers) {
  auto const tt = load_feeds({virt_feed()});

  // Control: RAPTOR finds FA -> FB.
  auto const raptor = search_at(tt, "L", "M", "10:00");
  ASSERT_EQ(1U, raptor.size());
  EXPECT_EQ(at("11:00"), begin(raptor)->dest_time_);

  auto const res = tb_search(
      tt, routing::query{.start_time_ = at("10:00"),
                         .start_ = {{lidx(tt, "L"), 0_minutes, 0U}},
                         .destination_ = {{lidx(tt, "M"), 0_minutes, 0U}}});
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("11:00"), begin(res)->dest_time_);
}

// Trip-based routing in a profile that projects virtual locations sees them
// as their stop, as RAPTOR does: Q1 and Q2 stop at virtual locations of QS
// (the rule Q1 -> Q2), and changing between them takes QS's own 2 min.
TEST(transfer_rules, trip_based_routing_projects_virtual_locations) {
  constexpr auto const kProfile = profile_idx_t{1U};
  auto tt = load_feeds(
      {feed({{"QS", 63.0, 22.0}, {"QA", 63.1, 22.0}, {"QB", 63.2, 22.0}},
            {{"Q1", "RQ1", {{"QA", "10:00"}, {"QS", "10:30"}}},
             {"Q2", "RQ2", {{"QS", "10:40"}, {"QB", "11:00"}}}},
            "QS,QS,2,120,,,,\n"
            "QS,QS,2,0,,,Q1,Q2\n")});
  ASSERT_NE(0U, n_virts(tt)) << "precondition: QS has virtual locations";
  add_empty_profile(tt, kProfile);

  auto const q =
      routing::query{.start_time_ = at("10:00"),
                     .start_ = {{lidx(tt, "QA"), 0_minutes, 0U}},
                     .destination_ = {{lidx(tt, "QB"), 0_minutes, 0U}},
                     .prf_idx_ = kProfile};
  auto const raptor = raptor_search(tt, nullptr, routing::query{q});
  ASSERT_EQ(1U, raptor.size());
  EXPECT_EQ(at("11:00"), begin(raptor)->dest_time_);

  auto const res = tb_search(tt, q);
  ASSERT_EQ(1U, res.size());
  EXPECT_EQ(at("11:00"), begin(res)->dest_time_);
}

// A time-dependent offset (motis: flex) names the stop. FA arrives at, and FC
// leaves from, virtual locations of U (the RF1 -> RF3 rule): the offset holds
// there too, like any other offset at U.
TEST(transfer_rules, td_offsets_hold_at_virtual_locations) {
  auto const tt = load_feeds(
      {feed({{"U", 55.0, 13.0}, {"L", 55.1, 13.0}, {"M", 55.2, 13.0}},
            {{"FA", "RF1", {{"L", "10:00"}, {"U", "10:30"}}},
             {"FC", "RF3", {{"U", "10:35"}, {"M", "11:00"}}}},
            "U,U,2,120,,,,\n"
            "U,U,2,300,RF1,RF3,,\n")});
  auto const u = lidx(tt, "U");
  ASSERT_NE(0U, n_virts(tt)) << "precondition: U has virtual locations";
  auto const td = [](std::string_view const hhmm, duration_t const d) {
    return std::vector<routing::td_offset>{
        {.valid_from_ = at(hhmm), .duration_ = d, .transport_mode_payload_ = 0},
        {.valid_from_ = at("12:00"),
         .duration_ = footpath::kMaxDuration,
         .transport_mode_payload_ = 0}};
  };

  // To U by flex (2 min), then FC from its virtual location.
  auto const from_u = raptor_search(
      tt, nullptr,
      routing::query{
          .start_time_ = at("10:30"),
          .start_match_mode_ = routing::location_match_mode::kIntermodal,
          .dest_match_mode_ = routing::location_match_mode::kEquivalent,
          .destination_ = {{lidx(tt, "M"), 0_minutes, 0U}},
          .td_start_ = {{u, td("10:00", 2_minutes)}}},
      direction::kForward);
  ASSERT_EQ(1U, from_u.size());
  EXPECT_EQ(at("11:00"), begin(from_u)->dest_time_);
  // The start leg names the query's stop, not the virtual location FC leaves.
  auto const& access = begin(from_u)->legs_.front();
  ASSERT_TRUE(std::holds_alternative<routing::offset>(access.uses_));
  EXPECT_EQ(u, std::get<routing::offset>(access.uses_).target());

  // FA to its virtual location of U, then on by flex (5 min).
  auto const to_u = raptor_search(
      tt, nullptr,
      routing::query{
          .start_time_ = at("10:00"),
          .start_match_mode_ = routing::location_match_mode::kEquivalent,
          .dest_match_mode_ = routing::location_match_mode::kIntermodal,
          .start_ = {{lidx(tt, "L"), 0_minutes, 0U}},
          .td_dest_ = {{u, td("10:00", 5_minutes)}}},
      direction::kForward);
  ASSERT_EQ(1U, to_u.size());
  EXPECT_EQ(at("10:35"), begin(to_u)->dest_time_);
  auto const& egress = begin(to_u)->legs_.back();
  ASSERT_TRUE(std::holds_alternative<routing::offset>(egress.uses_));
  EXPECT_EQ(u, std::get<routing::offset>(egress.uses_).target());
}

// FB2 (10:50) is an alternative for the FB leg of L -> FA -> FB -> M.
TEST(transfer_rules, leg_alternatives_after_trip_at_virtual_location) {
  auto const tt = load_feeds({virt_feed()});
  auto const q =
      routing::query{.start_time_ = at("10:00"),
                     .start_ = {{lidx(tt, "L"), 0_minutes, 0U}},
                     .destination_ = {{lidx(tt, "M"), 0_minutes, 0U}}};
  auto const res = raptor_search(tt, nullptr, routing::query{q});
  ASSERT_EQ(1U, res.size());
  auto const& j = *begin(res);

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

// The walk FA's virtual location -> U, as motis' itinerary refresh looks it
// up between two transit legs.
TEST(transfer_rules, lookup_footpath_from_virtual_location_to_its_stop) {
  auto const tt = load_feeds({virt_feed()});
  auto const res = search_at(tt, "L", "M", "10:00");
  ASSERT_EQ(1U, res.size());
  auto const& fa = begin(res)->legs_.front();
  ASSERT_TRUE(
      std::holds_alternative<routing::journey::run_enter_exit>(fa.uses_));
  ASSERT_EQ(location_type::kVirt, tt.locations_.types_[fa.to_]);

  auto const fp = routing::lookup_footpath(
      fa.to_, fa.arr_time_, routing::side::kAlighting, tt, nullptr,
      routing::query{}, {{lidx(tt, "U"), 0_minutes, 0U}},
      routing::location_match_mode::kExact, true);
  EXPECT_TRUE(fp.has_value());
}

// Door to door, the egress offset belongs to the query: its target is U, not
// the virtual location FA arrives at.
TEST(transfer_rules, egress_offset_names_the_query_stop) {
  auto const tt = load_feeds({virt_feed()});
  auto const res = nigiri::test::raptor_intermodal_search(
      tt, nullptr, {{lidx(tt, "L"), 0_minutes, 0U}},
      {{lidx(tt, "U"), 3_minutes, 0U}}, at("10:00"));
  ASSERT_EQ(1U, res.size());
  auto const& egress = begin(res)->legs_.back();
  ASSERT_TRUE(std::holds_alternative<routing::offset>(egress.uses_));
  EXPECT_EQ(lidx(tt, "U"), std::get<routing::offset>(egress.uses_).target());
}

// P and Q (44 m apart) both carry two virtual locations, so their walk is a
// walk hub and the stored footpath P -> Q is pruned. The fastest direct
// connection P -> Q is still that 2 min walk.
TEST(transfer_rules, fastest_direct_sees_walk_hubs) {
  auto const tt =
      load_feeds({feed({{"P", 57.0, 15.0},
                        {"Q", 57.0004, 15.0},
                        {"PA", 57.1, 15.0},
                        {"QA", 57.1, 15.2}},
                       {{"TP1", "RP1", {{"PA", "09:00"}, {"P", "09:30"}}},
                        {"TP2", "RP2", {{"PA", "09:10"}, {"P", "09:40"}}},
                        {"TQ1", "RQ1", {{"QA", "09:00"}, {"Q", "09:30"}}},
                        {"TQ2", "RQ2", {{"QA", "09:10"}, {"Q", "09:40"}}}},
                       "P,P,2,120,,,,\n"
                       "P,P,2,300,RP1,RX,,\n"
                       "P,P,2,420,RP2,RX,,\n"
                       "Q,Q,2,120,,,,\n"
                       "Q,Q,2,300,RQ1,RX,,\n"
                       "Q,Q,2,420,RQ2,RX,,\n",
                       {"RX"})});
  auto const p = lidx(tt, "P");
  auto const q = lidx(tt, "Q");

  // Precondition: P -> Q exists only through a hub.
  EXPECT_TRUE(
      utl::none_of(tt.locations_.footpaths_out_[kDefaultProfile][p],
                   [&](footpath const fp) { return fp.target() == q; }));
  auto hub_walk = std::optional<duration_t>{};
  routing::for_each_hub_source<direction::kForward>(
      tt, kDefaultProfile, q, [&](footpath const fp) {
        if (fp.target() == p) {
          hub_walk = fp.duration();
        }
        return true;
      });
  ASSERT_TRUE(hub_walk.has_value());

  auto const direct = routing::get_fastest_direct(
      tt,
      routing::query{.start_ = {{p, 0_minutes, 0U}},
                     .destination_ = {{q, 0_minutes, 0U}}},
      direction::kForward);
  EXPECT_EQ(hub_walk->count(), direct.count());
}

// ===========================================================================
// Tables sized before the virtual locations exist.
// ===========================================================================

// motis' flex routing indexes location_location_groups_ with every location
// near a coordinate, virtual ones included.
TEST(transfer_rules, location_group_table_covers_virtual_locations) {
  auto const tt = load_feeds({virt_feed()});
  ASSERT_NE(0U, n_virts(tt)) << "precondition";
  EXPECT_EQ(tt.n_locations(), tt.location_location_groups_.size());
}

// ===========================================================================
// Walks and the stops' transfer times.
// ===========================================================================

// E1 has a 5 min transfer time. E2 (same feed, 22 m) and E3 (other feed, 21 m)
// are equally close: leaving E1 on foot has to cost the same either way. The
// cross-feed link takes max(transfer times, walk), so the same-feed one has to
// as well.
TEST(transfer_rules, same_feed_walk_respects_transfer_time_like_cross_feed) {
  auto const tt = load_feeds(
      {feed({{"E1", 60.5, 20.0}, {"E2", 60.5002, 20.0}, {"E0", 60.6, 20.0}},
            {{"ET1", "RE1", {{"E0", "10:00"}, {"E1", "10:30"}}},
             {"ET2", "RE2", {{"E2", "10:40"}, {"E0", "11:00"}}}},
            "E1,E1,2,300,,,,\n"),
       feed({{"E3", 60.5, 20.0004}, {"E9", 60.6, 20.3}},
            {{"ET3", "RE3", {{"E3", "10:40"}, {"E9", "11:00"}}}}, "")});
  auto const e1 = lidx(tt, "E1");
  auto const walk_to = [&](location_idx_t const target) {
    auto best = std::optional<duration_t>{};
    for (auto const fp : tt.locations_.footpaths_out_[kDefaultProfile][e1]) {
      if (fp.target() == target) {
        best = fp.duration();
      }
    }
    return best;
  };
  auto const to_e2 = walk_to(lidx(tt, "E2"));
  auto const to_e3 = walk_to(lidx(tt, "E3", source_idx_t{1U}));
  ASSERT_TRUE(to_e2.has_value());
  ASSERT_TRUE(to_e3.has_value());
  EXPECT_EQ(to_e3->count(), to_e2->count());
}
