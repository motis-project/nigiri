#include "gtest/gtest.h"

#include "nigiri/loader/hrd/load_timetable.h"

#include "./hrd_timetable.h"

using namespace nigiri::loader::hrd;

TEST(hrd, hash) {
  EXPECT_TRUE(
      applicable(hrd_5_20_26, nigiri::test_data::hrd_timetable::files()));

  auto h = std::uint64_t{};
  EXPECT_NO_THROW(
      h = hash(hrd_5_20_26, nigiri::test_data::hrd_timetable::files()));
  EXPECT_NE(0U, h);
}
