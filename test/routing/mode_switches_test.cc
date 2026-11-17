#include "gtest/gtest.h"

#include "nigiri/routing/raptor/mcraptor.h"

using namespace nigiri;
using namespace nigiri::routing;

namespace {

using msd = mode_switches_dim;

std::uint32_t switches(std::initializer_list<clasz> const rides) {
  auto f = mc_fields{};
  for (auto const c : rides) {
    msd::after_ride(f, c, mc_context{});
  }
  return f.switches_;
}

mc_fields at(clasz const c, std::uint32_t const switches) {
  return {.group_ = static_cast<std::uint16_t>(mode_group(c)),
          .switches_ = static_cast<std::uint16_t>(switches)};
}

mc_fields complete(mc_fields f) {
  msd::complete(f);
  return f;
}

std::uint32_t join(mc_fields const a, mc_fields const b) {
  auto out = mc_fields{};
  msd::join(a, b, out);
  return out.switches_;
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

// At a stop, a label of another group only dominates with a switch to spare:
// the next ride may switch one of the two but not the other. Before the first
// ride (kNoModeGroup) there is nothing to switch from.
TEST(mode_switches, other_group_dominates_with_a_switch_to_spare) {
  auto const start = mc_fields{};

  EXPECT_TRUE(
      msd::dominates(at(clasz::kRegional, 1U), at(clasz::kRegional, 1U)));
  EXPECT_FALSE(msd::dominates(at(clasz::kRegional, 1U), at(clasz::kBus, 1U)));
  EXPECT_FALSE(msd::dominates(at(clasz::kBus, 1U), at(clasz::kRegional, 1U)));
  EXPECT_TRUE(msd::dominates(at(clasz::kRegional, 0U), at(clasz::kBus, 1U)));
  EXPECT_FALSE(
      msd::dominates(at(clasz::kRegional, 2U), at(clasz::kRegional, 1U)));
  EXPECT_TRUE(msd::dominates(start, at(clasz::kBus, 0U)));

  // a complete journey has no next ride: the count alone decides
  EXPECT_TRUE(msd::dominates(complete(at(clasz::kRegional, 1U)),
                             complete(at(clasz::kBus, 1U))));
  EXPECT_FALSE(msd::dominates(complete(at(clasz::kRegional, 2U)),
                              complete(at(clasz::kBus, 1U))));

  // halves of different groups meet with one more switch
  EXPECT_EQ(2U, join(at(clasz::kRegional, 1U), at(clasz::kBus, 0U)));
  EXPECT_EQ(1U, join(at(clasz::kBus, 1U), at(clasz::kBus, 0U)));
  EXPECT_EQ(1U, join(start, at(clasz::kBus, 1U)));
}

// The destination frontier prunes with ordered() before comparing field-wise:
// complete dominance has to imply ordered(a) <= ordered(b), and the integer
// has to keep every field.
TEST(mc_extras, fields_are_independent_and_ordered) {
  using x = arr_non_transit_mode_filter_mode_switches_criteria::extras_t;
  auto const make = [](std::uint32_t const nt, std::uint32_t const mf,
                       std::uint32_t const g, std::uint32_t const sw) {
    return x::bits({.non_transit_ = static_cast<std::uint16_t>(nt),
                    .mode_filter_ = static_cast<std::uint16_t>(mf),
                    .group_ = static_cast<std::uint16_t>(g),
                    .switches_ = static_cast<std::uint16_t>(sw)});
  };
  for (auto const nt : {0U, 1U, kNonTransitMax}) {
    for (auto const mf : {0U, 1U}) {
      for (auto const g : {kNoModeGroup, 3U, 7U}) {
        for (auto const sw : {0U, 2U, kMaxModeSwitches}) {
          auto const a = make(nt, mf, g, sw);
          auto const f = x::fields(a);
          EXPECT_EQ(nt, std::uint32_t{f.non_transit_});
          EXPECT_EQ(mf, std::uint32_t{f.mode_filter_});
          EXPECT_EQ(g, std::uint32_t{f.group_});
          EXPECT_EQ(sw, std::uint32_t{f.switches_});
          for (auto const b : {make(nt, 1U, 1U, sw),
                               make(kNonTransitMax, mf, g, kMaxModeSwitches)}) {
            if (x::completed_dominates(a, b)) {
              EXPECT_LE(x::ordered(a), x::ordered(b));
            }
          }
        }
      }
    }
  }
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
