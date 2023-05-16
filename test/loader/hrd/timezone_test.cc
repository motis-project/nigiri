#include "gtest/gtest.h"

#include "nigiri/loader/hrd/stamm/timezone.h"

using namespace nigiri;
using namespace nigiri::loader::hrd;
using namespace date;

TEST(hrd, timezone_roundtrip) {
  constexpr auto const timezones_file_content = R"(
0000000 +0100 +0200 29032020 0200 25102020 0300 +0200 28032021 0200 31102021 0300
)";

  auto tt = timetable{};
  auto const timezones =
      parse_timezones(hrd_5_00_8, tt, timezones_file_content);

  ASSERT_FALSE(timezones.empty());
  auto const tz = timezones.at(eva_number{0}).second;
  EXPECT_EQ(60_minutes, tz.offset_);
  EXPECT_EQ(2, tz.seasons_.size());
  EXPECT_EQ(120_minutes, tz.seasons_[0].offset_);
  EXPECT_EQ(date::sys_days{2020_y / March / 29}, tz.seasons_[0].begin_);
  EXPECT_EQ(date::sys_days{2020_y / October / 25}, tz.seasons_[0].end_);
  EXPECT_EQ(120_minutes, tz.seasons_[0].season_begin_mam_);
  EXPECT_EQ(180_minutes, tz.seasons_[0].season_end_mam_);
  EXPECT_EQ(120_minutes, tz.seasons_[1].offset_);
  EXPECT_EQ(date::sys_days{2021_y / March / 28}, tz.seasons_[1].begin_);
  EXPECT_EQ(date::sys_days{2021_y / October / 31}, tz.seasons_[1].end_);
  EXPECT_EQ(120_minutes, tz.seasons_[1].season_begin_mam_);
  EXPECT_EQ(180_minutes, tz.seasons_[1].season_end_mam_);

  auto const start = unixtime_t{date::sys_days{2020_y / January / 1}};
  auto const end = unixtime_t{date::sys_days{2021_y / January / 1}};
  for (auto t = start; t <= end; t += 30_minutes) {
    auto const local = to_local_time(tz, t);
    auto const local_day =
        unixtime_t{(local.time_since_epoch() / 1_days) * 1_days};
    auto const local_mam = duration_t{local.time_since_epoch().count() % 1440};
    auto const [mam, offset, valid] =
        local_mam_to_utc_mam(tz, local_day, local_mam);
    if (!valid) {
      continue;
    }

    // For times where local time jumps back (in this case 03:00 -> 02:00) the
    // conversion from local time to UTC time is ambiguous. Since
    // local_mam_to_utc_mam takes a local time, it's possible to match the UTC
    // time exactly in both cases. We exclude these cases from the test case.
    if (t != unixtime_t{date::sys_days{2020_y / October / 25}} + 90_minutes &&
        t != unixtime_t{date::sys_days{2020_y / October / 25}} + 1_hours) {
      EXPECT_EQ(local_day + offset + mam, t);
    }
  }
}

TEST(hrd, timezone_avv) {
  constexpr auto const input = R"(0000800 +0100 % Europe/Berlin
0000800 +0200 28032021 0200 31102021 0300)";
  auto tt = timetable{};
  auto const timezones = parse_timezones(hrd_5_20_avv, tt, input);

  ASSERT_FALSE(timezones.empty());
  auto const tz = timezones.at(eva_number{800}).second;
  EXPECT_EQ(60_minutes, tz.offset_);
  EXPECT_EQ(1, tz.seasons_.size());
  EXPECT_EQ(120_minutes, tz.seasons_[0].offset_);
  EXPECT_EQ(date::sys_days{2021_y / March / 28}, tz.seasons_[0].begin_);
  EXPECT_EQ(date::sys_days{2021_y / October / 31}, tz.seasons_[0].end_);
  EXPECT_EQ(120_minutes, tz.seasons_[0].season_begin_mam_);
  EXPECT_EQ(180_minutes, tz.seasons_[0].season_end_mam_);
}
