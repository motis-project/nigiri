#include "gtest/gtest.h"

#include "nigiri/common/delta_t.h"

using namespace nigiri;

TEST(DeltaT, NegDeltaValue) {
  {
    auto [day, mam] = split_day_mam(day_idx_t{10}, delta_t{0});
    EXPECT_EQ(day, day_idx_t{10});
    EXPECT_EQ(mam.count(), 0);
  }
  {
    auto [day, mam] = split_day_mam(day_idx_t{10}, delta_t{1441});
    EXPECT_EQ(day, day_idx_t{11});
    EXPECT_EQ(mam.count(), 1);
  }
  {
    auto [day, mam] = split_day_mam(day_idx_t{10}, delta_t{1440});
    EXPECT_EQ(day, day_idx_t{11});
    EXPECT_EQ(mam.count(), 0);
  }
  {
    auto [day, mam] = split_day_mam(day_idx_t{10}, delta_t{-1441});
    EXPECT_EQ(day, day_idx_t{8});
    EXPECT_EQ(mam.count(), 1439);
  }
  {
    auto [day, mam] = split_day_mam(day_idx_t{10}, delta_t{-1440});
    EXPECT_EQ(day, day_idx_t{9});
    EXPECT_EQ(mam.count(), 0);
  }
}