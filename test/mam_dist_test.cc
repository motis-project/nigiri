#include "gtest/gtest.h"

#include "nigiri/common/mam_dist.h"

using namespace nigiri;

TEST(mam_dist, a_equal_b) { EXPECT_EQ(0, mam_dist(23, 23)); }

TEST(mam_dist, a_greater_b) { EXPECT_EQ(19, mam_dist(42, 23)); }

TEST(mam_dist, b_greater_a) { EXPECT_EQ(19, mam_dist(23, 42)); }

TEST(mam_dist, a_greater_b_midnight) { EXPECT_EQ(40, mam_dist(1423, 23)); }

TEST(mam_dist, b_greater_a_midnight) { EXPECT_EQ(40, mam_dist(23, 1423)); }