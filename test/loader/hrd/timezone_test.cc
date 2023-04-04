#include "gtest/gtest.h"

#include "nigiri/loader/hrd/stamm/timezone.h"

using namespace nigiri;
using namespace nigiri::loader::hrd;

constexpr auto const timezones_file_content = R"(
0000000 +0100 +0200 29032020 0200 25102020 0300 +0200 28032021 0200 31102021 0300
)";

TEST(loader_hrd_timezone, roundtrip) {
  using namespace date;
  auto const& c = configs[0];
  timetable tt{};
  auto const timezones = parse_timezones(c, tt, timezones_file_content);
  auto const tz = timezones.at(eva_number{0}).second;
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
