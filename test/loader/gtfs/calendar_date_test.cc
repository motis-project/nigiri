#include "doctest/doctest.h"

#include "nigiri/loader/gtfs/calendar_date.h"

using namespace nigiri;
using namespace nigiri::loader::gtfs;
using namespace date;

constexpr auto const file_content = R"(service_id,date,exception_type
WD,20060703,2
WE,20060703,1
WD,20060704,2
WE,20060704,1)";

TEST_CASE("gtfs.calendar_date") {
  auto const dates = read_calendar_date(file_content);

  CHECK_EQ(2, dates.size());

  auto const we_it = dates.find("WE");
  REQUIRE_NE(we_it, end(dates));

  auto const wd_it = dates.find("WD");
  REQUIRE_NE(wd_it, end(dates));

  CHECK_EQ(2, dates.size());
  CHECK_EQ(2, dates.at("WD").size());
  CHECK_EQ(2, dates.at("WE").size());

  CHECK_EQ(July / 3 / 2006, dates.at("WD")[0].day_);
  CHECK_EQ(calendar_date::kRemove, dates.at("WD")[0].type_);

  CHECK_EQ(July / 3 / 2006, dates.at("WE")[0].day_);
  CHECK_EQ(calendar_date::kAdd, dates.at("WE")[0].type_);

  CHECK_EQ(July / 4 / 2006, dates.at("WD")[1].day_);
  CHECK_EQ(calendar_date::kRemove, dates.at("WD")[1].type_);

  CHECK_EQ(July / 4 / 2006, dates.at("WE")[1].day_);
  CHECK_EQ(calendar_date::kAdd, dates.at("WE")[1].type_);
}
