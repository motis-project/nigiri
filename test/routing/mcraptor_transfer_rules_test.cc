#include "gtest/gtest.h"

#include "nigiri/routing/raptor/bmraptor.h"
#include "nigiri/routing/raptor/mcraptor.h"
#include "nigiri/routing/raptor_search.h"
#include "nigiri/routing/search.h"

#if defined(NIGIRI_CUDA)
#include "nigiri/routing/gpu/mcraptor.h"
#endif

#include "../raptor_search.h"
#include "../transfer_rules_util.h"

using namespace nigiri;
using namespace nigiri::test;
using namespace std::chrono_literals;

namespace {

template <typename State, typename... StateArgs>
pareto_set<routing::journey> mc_search(timetable const& tt,
                                       routing::query const& q,
                                       StateArgs&&... state_args) {
  auto ss = routing::search_state{};
  auto state = State{std::forward<StateArgs>(state_args)...};
  return *routing::raptor_search(tt, nullptr, ss, state, q, direction::kForward)
              .journeys_;
}

}  // namespace

// No changing vehicles at Q: X1 arrives there by transport (10:30) and cannot
// go on, X2's walk from P arrives later but may board Y. The arrival by
// transport must neither reject nor evict the walk-in, though it dominates it.
//     X1: A 10:00 -> Q 10:30
//     X2: A 10:00 -> P 10:30, walk to Q (55 m)
//     Y:  Q 10:40 -> D 11:00
TEST(mcraptor_transfer_rules, banned_stop_keeps_walk_ins) {
  auto const tt =
      load_feeds({feed({{"A", 50.0, 6.0},
                        {"Q", 50.0, 7.0},
                        {"P", 50.0005, 7.0},
                        {"D", 50.0, 8.0}},
                       {{"X1", "RX1", {{"A", "10:00"}, {"Q", "10:30"}}},
                        {"X2", "RX2", {{"A", "10:00"}, {"P", "10:30"}}},
                        {"Y", "RY", {{"Q", "10:40"}, {"D", "11:00"}}}},
                       "Q,Q,3,,,,,\n")});
  // the harness also runs every mcraptor engine and expects raptor's journeys
  EXPECT_EQ(at("11:00"), arrival_at(tt, "A", "D", "10:00"));

  // BMRAPP's bounds must not prune the walk-in at Q either (a departure
  // interval: BMRAPP hands a fixed time to mcraptor)
  auto const q = exact_query(tt, "A", "D", interval{at("09:55"), at("10:05")});
  auto ss = routing::search_state{};
  auto as = routing::raptor_state{};
  auto ms = routing::mcraptor_non_transit_state{};
  EXPECT_EQ(1U, routing::bmrap_profile_search(tt, nullptr, ss, as, ms, q,
                                              direction::kForward)
                    .journeys_->size());
}

// S takes 6 min to change (S,S row), R1 -> R2 15 min: R1 cannot share a
// location of S with both R2 (T2: more than 10 min) and R3 (T3: at most 10
// min), so S gets a virtual location, and one of the changes walks a footpath
// of at least 6 min (a walking unit: kNonTransitBin) between two locations of
// S. Either is a transfer within the stop, no walk, on the CPU and the GPU.
//     T1 (R1):  A 10:00 -> S 10:30
//     T2 (R2):  S 10:40 -> D 11:00 (10 min: too short)
//     T2L (R2): S 10:50 -> D 11:10
//     T3 (R3):  S 10:40 -> E 10:50 (10 min: enough)
TEST(mcraptor_transfer_rules, change_within_stop_is_no_walk) {
  auto const tt = load_feeds({feed(
      {{"A", 51.0, 6.0}, {"S", 51.0, 7.0}, {"D", 51.0, 8.0}, {"E", 51.0, 9.0}},
      {{"T1", "R1", {{"A", "10:00"}, {"S", "10:30"}}},
       {"T2", "R2", {{"S", "10:40"}, {"D", "11:00"}}},
       {"T2L", "R2", {{"S", "10:50"}, {"D", "11:10"}}},
       {"T3", "R3", {{"S", "10:40"}, {"E", "10:50"}}}},
      "S,S,2,360,,,,\n"
      "S,S,2,900,R1,R2,,\n")});
  ASSERT_LT(0U, n_virts(tt));

  for (auto const& [to, arr] :
       {std::pair{"D", at("11:10")}, std::pair{"E", at("10:50")}}) {
    auto const q = exact_query(tt, "A", to, at("10:00"));
    auto const check = [&](pareto_set<routing::journey> const& js,
                           char const* engine) {
      ASSERT_EQ(1U, js.size()) << to << " " << engine;
      EXPECT_EQ(arr, begin(js)->arrival_time()) << to << " " << engine;
      EXPECT_EQ(0U, begin(js)->criteria_cost_) << to << " " << engine;
    };
    check(mc_search<routing::mcraptor_non_transit_state>(tt, q), "cpu");
#if defined(NIGIRI_CUDA)
    auto const gtt = routing::gpu::gpu_timetable{tt};
    check(mc_search<routing::gpu::gpu_mcraptor_non_transit_state>(tt, q, gtt),
          "gpu");
#endif
  }
}
