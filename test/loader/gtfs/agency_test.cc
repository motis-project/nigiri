#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/agency.h"
#include "nigiri/loader/gtfs/files.h"
#include "nigiri/timetable.h"

#include "./test_data.h"

using namespace nigiri;
using namespace nigiri::loader::gtfs;

TEST(gtfs, agency) {
  auto tt = timetable{};
  auto timezones = tz_map{};
  auto i18n = translator{.tt_ = tt};
  auto const agencies = read_agencies(
      source_idx_t{0}, tt, i18n, timezones,
      example_files().get_file(kAgencyFile).data(), "Europe/Berlin");

  auto const dta_it = agencies.find("DTA");
  ASSERT_NE(dta_it, end(agencies));

  auto const sbb_it = agencies.find("11");
  ASSERT_NE(sbb_it, end(agencies));

  auto const& dta = tt.providers_.at(dta_it->second);
  auto const& sbb = tt.providers_.at(sbb_it->second);
  EXPECT_EQ("Demo Transit Authority", tt.get_default_translation(dta.name_));
  EXPECT_EQ("http://google.com", tt.get_default_translation(dta.url_));
  EXPECT_EQ("Schweizerische Bundesbahnen SBB",
            tt.get_default_translation(sbb.name_));
  EXPECT_EQ("http://www.sbb.ch/", tt.get_default_translation(sbb.url_));
}
