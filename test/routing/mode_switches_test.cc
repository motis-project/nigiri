#include "gtest/gtest.h"

#include "nigiri/routing/raptor/mcraptor.h"

using namespace nigiri;
using namespace nigiri::routing;

namespace {

std::uint8_t switches(std::initializer_list<clasz> const rides) {
  auto d = mode_switches_dim::at_start(0U);
  for (auto const c : rides) {
    d = mode_switches_dim::from_ride(0U, ride_attrs{c}, d);
  }
  return d.switches_;
}

}  // namespace

TEST(mode_switches, local_transit_is_one_group) {
  EXPECT_EQ(0U, switches({clasz::kBus, clasz::kTram, clasz::kSubway,
                          clasz::kSuburban, clasz::kBus}));
}

TEST(mode_switches, other_classes_count) {
  EXPECT_EQ(1U, switches({clasz::kTram, clasz::kRegional}));
  EXPECT_EQ(2U, switches({clasz::kBus, clasz::kRegional, clasz::kTram}));
  EXPECT_EQ(0U, switches({clasz::kRegional, clasz::kRegional}));
  EXPECT_EQ(1U, switches({clasz::kCoach, clasz::kLongDistance}));
}

TEST(mode_switches, rail_classes_and_coach_are_separate_groups) {
  EXPECT_EQ(2U, switches({clasz::kRegional, clasz::kCoach, clasz::kRegional}));
  EXPECT_EQ(1U, switches({clasz::kHighSpeed, clasz::kLongDistance}));
  EXPECT_EQ(1U, switches({clasz::kNight, clasz::kRegional}));
}

TEST(mode_switches, remaining_classes_are_one_group) {
  EXPECT_EQ(0U, switches({clasz::kShip, clasz::kFunicular, clasz::kAerialLift,
                          clasz::kAir, clasz::kOther}));
  EXPECT_EQ(1U, switches({clasz::kShip, clasz::kBus}));
}

TEST(non_transit, walks_count_in_floored_units) {
  if constexpr (routing::kNonTransitFullResolution) {
    GTEST_SKIP() << "full resolution";
  }
  using routing::non_transit_add;
  EXPECT_EQ(0U, non_transit_add(0U, routing::kNonTransitBin - 1U));
  EXPECT_EQ(1U, non_transit_add(0U, routing::kNonTransitBin));
  // per walk: two short walks stay free, one of their combined length counts
  EXPECT_EQ(0U, non_transit_add(non_transit_add(0U, 3U), 3U));
  EXPECT_EQ(1U, non_transit_add(0U, 6U));
  EXPECT_EQ(routing::kNonTransitMax,
            non_transit_add(routing::kNonTransitMax, routing::kNonTransitBin));
}
