#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/agency.h"
#include "nigiri/timetable.h"

using namespace nigiri;
using namespace nigiri::loader::gtfs;

constexpr auto const file_content = std::string_view{
    R"(agency_id,agency_name,agency_url,agency_timezone
DTA,Demo Transit Authority,http://google.com,America/Los_Angeles
"11","Schweizerische Bundesbahnen SBB","http://www.sbb.ch/","Europe/Berlin","DE","0900 300 300 ")"};

TEST(gtfs, agency) {
  timetable tt;
  auto const agencies = parse_agencies(tt, file_content);

  auto const dta_it = agencies.find("DTA");
  ASSERT_NE(dta_it, end(agencies));

  auto const sbb_it = agencies.find("11");
  ASSERT_NE(sbb_it, end(agencies));

  EXPECT_EQ("Demo Transit Authority",
            tt.providers_.at(dta_it->second).long_name_);
  EXPECT_EQ("Schweizerische Bundesbahnen SBB",
            tt.providers_.at(sbb_it->second).long_name_);
}
