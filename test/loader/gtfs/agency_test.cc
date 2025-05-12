#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/agency.h"
#include "nigiri/loader/gtfs/files.h"
#include "nigiri/timetable.h"

#include "./test_data.h"

using namespace nigiri;
using namespace nigiri::loader::gtfs;

TEST(gtfs, agency) {
  timetable tt;
  tz_map timezones;
  auto const agencies = read_agencies(
      tt, timezones, example_files().get_file(kAgencyFile).data());

  auto const dta_it = agencies.find("DTA");
  ASSERT_NE(dta_it, end(agencies));

  auto const sbb_it = agencies.find("11");
  ASSERT_NE(sbb_it, end(agencies));

  auto const& dta = tt.providers_.at(dta_it->second);
  auto const& sbb = tt.providers_.at(sbb_it->second);
  EXPECT_EQ("Demo Transit Authority", tt.strings_.get(dta.long_name_));
  EXPECT_EQ("http://google.com", tt.strings_.get(dta.url_));
  EXPECT_EQ("Schweizerische Bundesbahnen SBB", tt.strings_.get(sbb.long_name_));
  EXPECT_EQ("http://www.sbb.ch/", tt.strings_.get(sbb.url_));
}
