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
  // area_idx_t const kMaxAreaIndex = area_idx_t{UINT32_MAX};
  //
  // timetable tt;
  // source_idx_t src = source_idx_t{0};
  //
  // auto const files = example_files();
  //
  // auto const geojson = read_location_geojson(
  //     src, tt, files.get_file(kLocationGeojsonFile).data());
  //
  // tz_map timezones;
  //
  // auto const stops = read_stops(source_idx_t{0}, tt, timezones,
  //                               files.get_file(kStopFile).data(),
  //                               files.get_file(kTransfersFile).data(), 0U);
  //
  // auto const areas = read_areas(src, src, src, src, src, tt, stops, geojson,
  //                               files.get_file(kStopAreasFile).data(),
  //                               files.get_file(kLocationGroupsFile).data(),
  //                               files.get_file(kLocationGroupStopsFile).data());
  //
  // // Stop Area
  // auto const test_area =
  //     [&](std::string const& id, std::vector<std::string> const&& stop_ids,
  //         std::vector<std::string> const&& location_geojson_ids) {
  //       if (stop_ids.empty() && location_geojson_ids.empty()) {
  //         EXPECT_TRUE(id.empty());
  //         return;
  //       }
  //       ASSERT_NO_THROW({
  //         ASSERT_FALSE(id.empty());
  //         auto const idx = areas.at(id);
  //         if (location_geojson_ids.empty()) {
  //           ASSERT_NE(idx.location_, kMaxAreaIndex);
  //           EXPECT_EQ(idx.location_geojson_, kMaxAreaIndex);
  //           auto const stop_idxs =
  //               tt.areas_.area_idx_to_location_idxs_.at(idx.location_);
  //           ASSERT_EQ(stop_idxs.size(), stop_ids.size());
  //           for (auto i = 0; i < stop_idxs.size(); ++i) {
  //             EXPECT_EQ(stop_idxs.at(i), stops.at(stop_ids.at(i)));
  //           }
  //         } else if (stop_ids.empty()) {
  //           ASSERT_NE(idx.location_geojson_, kMaxAreaIndex);
  //           EXPECT_EQ(idx.location_, kMaxAreaIndex);
  //           auto const location_geojson_idxs =
  //               tt.areas_.area_idx_to_location_geojson_idxs_.at(
  //                   idx.location_geojson_);
  //           ASSERT_EQ(location_geojson_ids.size(),
  //           location_geojson_ids.size()); for (auto i = 0; i <
  //           location_geojson_idxs.size(); ++i) {
  //             EXPECT_EQ(location_geojson_idxs.at(i),
  //                       geojson.at(location_geojson_ids.at(i)));
  //           }
  //         } else {
  //           ASSERT_NE(idx.location_, kMaxAreaIndex);
  //           ASSERT_NE(idx.location_geojson_, kMaxAreaIndex);
  //
  //           auto const location_geojson_idxs =
  //               tt.areas_.area_idx_to_location_geojson_idxs_.at(
  //                   idx.location_geojson_);
  //           auto const stop_idxs =
  //               tt.areas_.area_idx_to_location_idxs_.at(idx.location_);
  //
  //           ASSERT_EQ(stop_idxs.size(), stop_ids.size());
  //           ASSERT_EQ(location_geojson_ids.size(),
  //           location_geojson_ids.size());
  //
  //           for (auto i = 0; i < stop_idxs.size(); ++i) {
  //             EXPECT_EQ(stop_idxs.at(i), stops.at(stop_ids.at(i)));
  //           }
  //
  //           for (auto i = 0; i < location_geojson_idxs.size(); ++i) {
  //             EXPECT_EQ(location_geojson_idxs.at(i),
  //                       geojson.at(location_geojson_ids.at(i)));
  //           }
  //         }
  //       });
  //     };
  //
  // auto const test_area_stops_only =
  //     [&](std::string const& id, std::vector<std::string> const&& stop_ids) {
  //       test_area(id, std::move(stop_ids), {});
  //     };
  //
  // test_area_stops_only("a_1", {"S1"});
  // test_area_stops_only("a_2", {"S2", "S3", "S4", "S5", "S6"});
  // test_area_stops_only("a_3", {"S1", "S2", "S7", "S8"});
  // // Location Group
  // test_area_stops_only("l_g_s_1", {"S1", "S2", "S3"});
  // test_area_stops_only("l_g_s_2", {"S4", "S5", "S6", "S7"});
  // test_area_stops_only("l_g_s_3", {"S8", "S2"});
  // // Location Group Stop
}
