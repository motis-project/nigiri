#include "gtest/gtest.h"

#include "nigiri/common/interval.h"
#include "nigiri/types.h"

using namespace nigiri;

unixtime_t clamp(interval<unixtime_t> const& i, unixtime_t const t) {
  // i.to_ is the first invalid value
  // so (i.to_ - 1) is the last valid value
  return std::clamp(i.from_, i.to_ - 1_minutes, t);
}

TEST(interval, clamp) {
  auto const t = [](auto&& x) { return unixtime_t{duration_t{x}}; };
  auto const i = interval{t(10), t(15)};

  EXPECT_TRUE(i.contains(t(10)));
  EXPECT_TRUE(i.contains(t(11)));
  EXPECT_TRUE(i.contains(t(12)));
  EXPECT_TRUE(i.contains(t(13)));
  EXPECT_TRUE(i.contains(t(14)));
  EXPECT_FALSE(i.contains(t(15)));

  EXPECT_EQ(t(14), clamp(i, t(20)));
}
