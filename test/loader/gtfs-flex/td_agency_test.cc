#include <nigiri/loader/gtfs-flex/td_agency.h>

#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/agency.h"
#include "nigiri/loader/gtfs/files.h"
#include "nigiri/timetable.h"

#include "./test_data.h"

using namespace nigiri;
using namespace nigiri::loader::gtfs_flex;

TEST(gtfs, agency) {
  loader::gtfs::tz_map timezones;
  auto const agencies = read_td_agencies(loader::gtfs::example_files().get_file(k_td_AgencyFile).data());

  auto const dta_it = agencies.find("DTA");
  ASSERT_NE(dta_it, end(agencies));

  auto const sbb_it = agencies.find("11");
  ASSERT_NE(sbb_it, end(agencies));

  auto const& dta = agencies.at(dta_it->second);
  auto const& sbb = agencies.at(sbb_it->second);
  EXPECT_EQ("Demo Transit Authority", dta.get()->name_);
  EXPECT_EQ("http://google.com", dta.get()->url_);
  EXPECT_EQ("Schweizerische Bundesbahnen SBB", sbb.get()->name_);
  EXPECT_EQ("http://www.sbb.ch/", sbb.get()->url_);
}
