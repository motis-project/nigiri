#include "gtest/gtest.h"

#include "nigiri/types.h"

using namespace nigiri;
using namespace date;
using namespace std::chrono_literals;

TEST(netex, time) {
  auto const tz = date::locate_zone("Europe/Berlin");

  auto to_utc = [&, info = std::optional<date::local_info>{}](
                    date::sys_days const day,
                    minutes_after_midnight_t const x) mutable {
    if (!info || !interval{info->first.begin, info->first.end}.contains(
                     std::chrono::time_point_cast<date::sys_seconds::duration>(
                         day + x - info->first.offset))) {
      info = tz->get_info(
          date::local_time<i32_minutes>{day.time_since_epoch() + x});
    }
    return x - std::chrono::duration_cast<duration_t>(info->first.offset);
  };

  EXPECT_EQ(1h, to_utc(date::sys_days(2025_y / March / 30), 2h));
  EXPECT_EQ(5h, to_utc(date::sys_days(2025_y / March / 30), 7h));
  EXPECT_EQ(0h, to_utc(date::sys_days(2025_y / October / 26), 2h));
  EXPECT_EQ(6h, to_utc(date::sys_days(2025_y / October / 26), 7h));
}