#include <nigiri/loader/gtfs/booking_rule.h>

#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/area.h"
#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/parse_date.h"
#include "nigiri/loader/gtfs/parse_time.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "./test_data.h"

using namespace nigiri;
using namespace nigiri::loader::gtfs;

TEST(gtfs, area) {
  timetable tt;
  source_idx_t src = source_idx_t{0};

  auto const files = example_files();

  tz_map timezones;

  auto const stops = read_stops(source_idx_t{0}, tt, timezones,
                                files.get_file(kStopFile).data(),
                                files.get_file(kTransfersFile).data(), 0U);

  auto const areas =
      read_areas(tt, stops, files.get_file(kStopAreasFile).data(),
                 files.get_file(kLocationGroupsFile).data(),
                 files.get_file(kLocationGroupStopsFile).data());
  // Stop Area
  auto const test_area =
      [&](std::string const& id,
          std::vector<std::string> const&& expected_stop_ids) {
        auto const& area_idx = areas.at(id);
        ASSERT_NE(area_idx, area_idx_t::invalid());

        auto const& actual_stops = tt.area_idx_to_location_idxs_.at(area_idx);
        ASSERT_EQ(actual_stops.size(), expected_stop_ids.size());
        for (auto i = 0; i < expected_stop_ids.size(); ++i) {
          EXPECT_EQ(actual_stops.at(i), stops.at(expected_stop_ids.at(i)));
        }
      };

  // Stop Areas
  test_area("a_1", {"S1"});
  test_area("a_2", {"S2", "S3", "S4", "S5", "S6"});
  test_area("a_3", {"S1", "S2", "S7", "S8"});
  // Location Group Stops
  test_area("l_g_s_1", {"S1", "S2", "S3"});
  test_area("l_g_s_2", {"S4", "S5", "S6", "S7"});
  test_area("l_g_s_3", {"S8", "S2"});
  // Location Group
  test_area("l_g_1", {"S1", "S2"});
  test_area("l_g_2", {"S2", "S3"});
  test_area("l_g_3", {"S4"});
}