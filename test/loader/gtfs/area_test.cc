#include <nigiri/loader/gtfs/booking_rule.h>

#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/area.h"
#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/parse_date.h"
#include "nigiri/loader/gtfs/parse_time.h"
#include "nigiri/timetable.h"

#include "./test_data.h"

using namespace nigiri;
using namespace nigiri::loader::gtfs;

TEST(gtfs, area) {
  timetable tt;
  source_idx_t src = source_idx_t{0};

  auto const files = example_files();

  auto const geojson = read_location_geojson(
      src, tt, files.get_file(kLocationGeojsonFile).data());

  tz_map timezones;

  auto const stops = read_stops(source_idx_t{0}, tt, timezones,
                                files.get_file(kStopFile).data(),
                                files.get_file(kTransfersFile).data(), 0U);

  auto const areas = read_areas(src, src, src, src, src, tt, stops, geojson,
                                files.get_file(kStopAreasFile).data(),
                                files.get_file(kLocationGroupsFile).data(),
                                files.get_file(kLocationGroupStopsFile).data());
  // Stop Area
  auto const test_area =
      [&](std::string const& id, source_idx_t src,
          std::vector<std::string> const&& expected_stop_ids,
          std::vector<std::string> const&& expected_location_geojson_ids) {
        auto const& area_idx = areas.at(id);

        if (area_idx.location_ != kInvalidAreaIndex) {
          auto const& actual_stops =
              tt.areas_.area_idx_to_location_idxs_.at(area_idx.location_);
          ASSERT_EQ(actual_stops.size(), expected_stop_ids.size());
          for (auto i = 0; i < expected_stop_ids.size(); ++i) {
            EXPECT_EQ(actual_stops.at(i),
                      tt.locations_.location_id_to_idx_.at(
                          location_id{expected_stop_ids.at(i), src}));
          }
        }

        if (area_idx.location_geojson_ != kInvalidAreaIndex) {
          auto const& actual_geojsons =
              tt.areas_.area_idx_to_location_geojson_idxs_.at(
                  area_idx.location_geojson_);
          ASSERT_EQ(actual_geojsons.size(),
                    expected_location_geojson_ids.size());
          for (auto i = 0; i < expected_location_geojson_ids.size(); ++i) {
            EXPECT_EQ(
                actual_geojsons.at(i),
                tt.location_id_to_location_geojson_idx_.at(locationGeoJson_id{
                    expected_location_geojson_ids.at(i), src}));
          }
        }
      };

  // Stop Areas
  test_area("a_1", src, {"S1"}, {});
  test_area("a_2", src, {"S2", "S3", "S4", "S5", "S6"}, {});
  test_area("a_3", src, {"S1", "S2", "S7", "S8"}, {});
  // Location Group Stops
  test_area("l_g_s_1", src, {"S1", "S2", "S3"}, {});
  test_area("l_g_s_2", src, {"S4", "S5", "S6", "S7"}, {});
  test_area("l_g_s_3", src, {"S8", "S2"}, {});
  // Location Group
  test_area("l_g_1", src, {"S1", "S2"}, {"l_geo_1", "l_geo_2"});
  test_area("l_g_2", src, {}, {"l_geo_3"});
  test_area("l_g_3", src, {"S4"}, {});
}
