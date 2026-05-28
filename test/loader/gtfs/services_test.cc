#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/calendar.h"
#include "nigiri/loader/gtfs/calendar_date.h"
#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/parse_date.h"
#include "nigiri/loader/gtfs/services.h"

#include "nigiri/common/interval.h"
#include "nigiri/types.h"

#include "./test_data.h"

using namespace nigiri;
using namespace nigiri::loader;
using namespace nigiri::loader::gtfs;
using namespace date;

/*
 * -- BASE --
 * WE:       11         WD:       00
 *      0000011              1111100
 *      0000011              1111100
 *      0000011              1111100
 *      0000011              1111100
 *      0                    1
 */

/*
 * -- DATES --
 * WE:       11         WD:       00
 *      1100011              0011100
 *      0000011              1111100
 *      0000011              1111100
 *      0000011              1111100
 *      0                    1
 */

TEST(gtfs, service_dates) {
  auto const i = interval{date::sys_days{July / 1 / 2006},
                          date::sys_days{August / 1 / 2006}};
  auto dates =
      read_calendar_date(example_files().get_file(kCalendarDatesFile).data());
  auto calendar = read_calendar(example_files().get_file(kCalenderFile).data());
  auto traffic_days = merge_traffic_days(i, calendar, dates);

  auto we_bit_str = std::string{"1111000110000011000001100000110"};
  auto wd_bit_str = std::string{"0000111001111100111110011111001"};
  std::reverse(begin(we_bit_str), end(we_bit_str));
  std::reverse(begin(wd_bit_str), end(wd_bit_str));
  auto const we_traffic_days = bitfield{we_bit_str};
  auto const wd_traffic_days = bitfield{wd_bit_str};

  EXPECT_EQ(we_traffic_days, *traffic_days["WE"]);
  EXPECT_EQ(wd_traffic_days, *traffic_days["WD"]);
}

TEST(gtfs, von_dates) {
  constexpr auto const kCalendar =
      R"(service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
"T0+ss#68","1","1","1","1","1","0","0","20240226","20240428")";
  auto const i = interval{date::sys_days{August / 7 / 2024},
                          date::sys_days{August / 7 / 2025}};
  auto const dates = read_calendar_date("");
  auto const calendar = read_calendar(kCalendar);

  {
    auto const traffic_days = merge_traffic_days(i, calendar, dates);
    EXPECT_TRUE(traffic_days.at("T0+ss#68")->none());
  }

  {
    auto const traffic_days =
        merge_traffic_days(i, calendar, dates, parse_date(20240428));
    EXPECT_TRUE(traffic_days.at("T0+ss#68")->any());
  }
}
