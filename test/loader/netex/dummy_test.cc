//
// Created by mirko on 8/23/23.
//
#include "gtest/gtest.h"

#include "nigiri/loader/netex/load_timetable.h"
#include "nigiri/timetable.h"

namespace {
std::vector<int> x = {1, 2, 3};
std::vector<int> y = {1, 2, 3};

}  // namespace

TEST(netex_dummy, first_gtest_case) {
  ASSERT_EQ(x.size(), y.size()) << "Vectors x and y are of unequal length";

  for (ulong i = 0; i < x.size(); ++i) {
    EXPECT_EQ(x[i], y[i]) << "Vectors x and y differ at index " << i;
  }
}

TEST(netex_dummy, second_gtest_case) { EXPECT_EQ(2, 2); }
