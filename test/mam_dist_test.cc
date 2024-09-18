#include "gtest/gtest.h"

#include "nigiri/common/mam_dist.h"
#include "nigiri/types.h"

using namespace nigiri;

TEST(mam_dist, a_equal_b) {
  EXPECT_EQ((std::pair{i32_minutes{0}, date::days{0}}),
            mam_dist(i32_minutes{23}, i32_minutes{23}));
}

TEST(mam_dist, a_greater_b) {
  EXPECT_EQ((std::pair{i32_minutes{19}, date::days{0}}),
            mam_dist(i32_minutes{42}, i32_minutes{23}));
}

TEST(mam_dist, b_greater_a) {
  EXPECT_EQ((std::pair{i32_minutes{19}, date::days{0}}),
            mam_dist(i32_minutes{23}, i32_minutes{42}));
}

TEST(mam_dist, a_greater_b_midnight) {
  EXPECT_EQ((std::pair{i32_minutes{40U}, date::days{+1}}),
            mam_dist(i32_minutes{1423}, i32_minutes{23}));
}

TEST(mam_dist, b_greater_a_midnight) {
  EXPECT_EQ((std::pair{i32_minutes{40U}, date::days{-1}}),
            mam_dist(i32_minutes{23}, i32_minutes{1423}));
}