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

TEST(interval, shift) {
  auto const i = interval{stop_idx_t{3}, stop_idx_t{14}};

  // operator>>()
  EXPECT_EQ((interval{stop_idx_t{25}, stop_idx_t{36}}), i >> 22);
  // operator<<()
  EXPECT_EQ((interval{stop_idx_t{1}, stop_idx_t{12}}), i << 2);
  // Unsigned underflow
  EXPECT_EQ((interval{static_cast<stop_idx_t>(-stop_idx_t{12}),
                      static_cast<stop_idx_t>(-stop_idx_t{1})}),
            i >> -15);
  EXPECT_EQ((interval{stop_idx_t{65500}, stop_idx_t{65511}}), i << 39);
}
