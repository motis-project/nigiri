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
