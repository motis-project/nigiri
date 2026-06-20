#include "gtest/gtest.h"

#include "date/date.h"

#include "nigiri/common/parse_time.h"

using namespace nigiri;
using namespace date;
using namespace std::chrono_literals;

TEST(parse_time, invalid_time_tz) {
  EXPECT_THROW(parse_time_tz("invalid", "%Y-%m-%d %H:%M %Z"), std::exception);
}

TEST(parse_time, valid_time_tz) {
  EXPECT_EQ(
      unixtime_t{date::sys_days{2000_y / January / 1} + 7h + 7min},
      parse_time_tz("2000-01-01 08:07 Europe/Berlin", "%Y-%m-%d %H:%M %Z"));
}

TEST(parse_time, valid_time_offset) {
  EXPECT_EQ(unixtime_t{date::sys_days{2025_y / July / 3} + 9h + 21min},
            parse_time("2025-07-03T11:21:00+02:00", "%FT%T%Ez"));
}

TEST(parse_time, invalid_time) {
  EXPECT_THROW(parse_time("invalid", "%FT%T%z"), std::exception);
}

TEST(parse_time, valid_time) {
  EXPECT_EQ(unixtime_t{date::sys_days{2000_y / January / 1} + 7h + 7min},
            parse_time("2000-01-01T07:07:00+0000", "%FT%T%z"));
}

TEST(parse_time, invalid_time_no_tz) {
  EXPECT_THROW(parse_time_no_tz("invalid"), std::exception);
}

TEST(parse_time, valid_time_no_tz) {
  EXPECT_EQ(unixtime_t{date::sys_days{2000_y / January / 1} + 7h + 7min},
            parse_time_no_tz("2000-01-01T07:07:00"));
}