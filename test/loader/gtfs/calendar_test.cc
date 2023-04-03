#include "doctest/doctest.h"

#include "nigiri/loader/gtfs/calendar.h"

using namespace nigiri;
using namespace nigiri::loader::gtfs;
using namespace date;

constexpr auto const file_content = std::string_view{
    R"(service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
WE,0,0,0,0,0,1,1,20060701,20060731
WD,1,1,1,1,1,0,0,20060701,20060731)"};

TEST_CASE("gtfs.calendar") {
  auto const calendar = read_calendar(file_content);

  CHECK_EQ(2, calendar.size());

  auto const we_it = calendar.find("WE");
  REQUIRE_NE(we_it, end(calendar));

  auto const wd_it = calendar.find("WD");
  REQUIRE_NE(wd_it, end(calendar));

  CHECK(we_it->second.week_days_.test(0));
  CHECK_FALSE(we_it->second.week_days_.test(1));
  CHECK_FALSE(we_it->second.week_days_.test(2));
  CHECK_FALSE(we_it->second.week_days_.test(3));
  CHECK_FALSE(we_it->second.week_days_.test(4));
  CHECK_FALSE(we_it->second.week_days_.test(5));
  CHECK(we_it->second.week_days_.test(6));

  CHECK_FALSE(wd_it->second.week_days_.test(0));
  CHECK(wd_it->second.week_days_.test(1));
  CHECK(wd_it->second.week_days_.test(2));
  CHECK(wd_it->second.week_days_.test(3));
  CHECK(wd_it->second.week_days_.test(4));
  CHECK(wd_it->second.week_days_.test(5));
  CHECK_FALSE(wd_it->second.week_days_.test(6));

  CHECK_EQ(July / 1 / 2006, we_it->second.first_day_);
  CHECK_EQ(July / 31 / 2006, we_it->second.last_day_);
  CHECK_EQ(July / 1 / 2006, wd_it->second.first_day_);
  CHECK_EQ(July / 31 / 2006, wd_it->second.last_day_);
}
