#include <tuple>
#include <vector>

#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/local_to_utc.h"
#include "nigiri/loader/gtfs/parse_time.h"

using namespace nigiri;
using namespace nigiri::loader::gtfs;
using namespace date;

auto const oct_27 = October / 27 / 2019;
auto const march_31 = March / 31 / 2019;
auto const test_cases = std::vector<std::tuple<date::year_month_day,
                                               std::string,
                                               std::time_t,
                                               std::string,
                                               std::string>>{
    {oct_27, "2019-10-27T01:00+02:00", 1572130800, "00:00", "01:00"},
    {oct_27, "2019-10-27T01:59+02:00", 1572134340, "00:59", "01:59"},
    {oct_27, "2019-10-27T02:59+02:00", 1572137940, "01:59", "02:59"},
    {oct_27, "2019-10-27T02:00+01:00", 1572138000, "02:00", "02:00"},
    {oct_27, "2019-10-27T02:59+01:00", 1572141540, "02:59", "02:59"},
    {oct_27, "2019-10-27T03:00+01:00", 1572141600, "03:00", "03:00"},
    {march_31, "2019-03-31T01:00+01:00", 1553990400, "02:00", "01:00"},
    {march_31, "2019-03-31T01:59+01:00", 1553993940, "02:59", "01:59"},
    {march_31, "2019-03-31T03:00+02:00", 1553994000, "03:00", "03:00"},
};

TEST(gtfs, gtfs_to_utc) {
  auto const tz = date::locate_zone("Europe/Berlin");
  for (auto const& [day, iso, unixtime, gtfs, wall_clock] : test_cases) {
    auto const local_mam = minutes_after_midnight_t{hhmm_to_min(gtfs)};
    auto const utc_time = date::sys_days{day} + local_mam -
                          get_noon_offset(date::local_days{day}, tz);
    EXPECT_EQ(unixtime, std::chrono::duration_cast<std::chrono::seconds>(
                            utc_time.time_since_epoch())
                            .count());
    EXPECT_EQ(iso, date::format("%FT%R%Ez", zoned_time{tz, utc_time}));
    EXPECT_EQ(wall_clock,
              date::format("%R", zoned_time{tz, utc_time}.get_local_time()));
  }
}
