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

  auto const geojson = read_location_geojson(
      src, tt, files.get_file(kLocationGeojsonFile).data());

  tz_map timezones;

  auto const stops = read_stops(source_idx_t{0}, tt, timezones,
                                files.get_file(kStopFile).data(),
                                files.get_file(kTransfersFile).data(), 0U);

  auto const areas = read_areas(src, tt, files.get_file(kStopAreasFile).data(),
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
              tt.area_idx_to_location_idxs_.at(area_idx.location_);
          ASSERT_EQ(actual_stops.size(), expected_stop_ids.size());
          for (auto i = 0; i < expected_stop_ids.size(); ++i) {
            EXPECT_EQ(actual_stops.at(i),
                      tt.locations_.location_id_to_idx_.at(
                          location_id{expected_stop_ids.at(i), src}));
          }
        }

        if (area_idx.location_geojson_ != kInvalidAreaIndex) {
          auto const& actual_geojsons =
              tt.area_idx_to_location_geojson_idxs_.at(
                  area_idx.location_geojson_);
          ASSERT_EQ(actual_geojsons.size(),
                    expected_location_geojson_ids.size());
          for (auto i = 0; i < expected_location_geojson_ids.size(); ++i) {
            EXPECT_EQ(
                actual_geojsons.at(i),
                tt.location_id_to_location_geojson_idx_.at(location_geojson_id{
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

TEST(gtfs, rtree) {
  timetable tt;
  source_idx_t src = source_idx_t{0};

  auto const files = example_files();

  auto const geojson = read_location_geojson(
      src, tt, files.get_file(kRtreeLocationGeojsonFile).data());

  tz_map timezones;

  auto const stops = read_stops(src, tt, timezones,
                                files.get_file(kRtreeStopFile).data(), "", 0U);

  auto areas =
      read_areas(src, tt, files.get_file(kRtreeLocationGroupFile).data());

  // Points only
  geo::latlng t1_p1 = {48.86392712004471, 10.268574187361935};
  geo::latlng t1_p2 = {48.78106035877934, 9.174858910926702};

  auto matches = tt.lookup_td_stops(t1_p1);
  EXPECT_TRUE(matches.empty());
  matches = tt.lookup_td_stops(t1_p2);
  ASSERT_EQ(matches.size(), 1);
  ASSERT_NO_THROW({ EXPECT_EQ(matches[0].area_idxs_, areas.at("l_g_3")); });
  EXPECT_TRUE(matches[0].location_geojsons_.empty());
  ASSERT_EQ(matches[0].locations_.size(), 1);
  ASSERT_NO_THROW(
      { EXPECT_EQ(matches[0].locations_[0], stops.at("Stuttgart")); });

  // Polygon and Points
  geo::latlng t2_p1 = {51.228381181627526, 6.772220890150692};
  geo::latlng t2_p2 = {51.43395142107519, 5.276914890042207};

  matches = tt.lookup_td_stops(t2_p1);
  ASSERT_EQ(matches.size(), 1);
  auto l_geo_2_it = areas.find("l_g_2");
  ASSERT_NE(l_geo_2_it, end(areas));
  auto const& t2_expected1 =
      area_idx{kInvalidAreaIndex, l_geo_2_it->second.location_geojson_};

  EXPECT_EQ(matches[0].area_idxs_, t2_expected1);
  EXPECT_TRUE(matches[0].locations_.empty());
  ASSERT_EQ(matches[0].location_geojsons_.size(), 1);
  ASSERT_NO_THROW({
    EXPECT_EQ(matches[0].location_geojsons_[0],
              geojson.at("Duesseldorf-Umgebung"));
  });

  matches = tt.lookup_td_stops(t2_p2);
  EXPECT_TRUE(matches.empty());

  // Polygon with hole
  geo::latlng t3_p1 = {52.516380054478844, 13.393337943006543};
  geo::latlng t3_p2 = {50.879957734721586, 12.080196883746197};
  geo::latlng t3_p3 = {52.13535399323928, 11.634545207813858};

  matches = tt.lookup_td_stops(t3_p1);
  EXPECT_TRUE(matches.empty());
  matches = tt.lookup_td_stops(t3_p2);
  EXPECT_TRUE(matches.empty());
  matches = tt.lookup_td_stops(t3_p3);
  ASSERT_EQ(matches.size(), 1);
  ASSERT_NO_THROW({ EXPECT_EQ(matches[0].area_idxs_, areas.at("l_g_1")); });
  ASSERT_EQ(matches[0].location_geojsons_.size(), 1);
  ASSERT_NO_THROW({
    EXPECT_EQ(matches[0].location_geojsons_[0], geojson.at("Brandenburg"));
  });

  // Multipolygon
  geo::latlng t4_p1 = {47.83229005918568, 15.289364972187087};
  geo::latlng t4_p2 = {48.393347831713896, 16.206412961908768};
  geo::latlng t4_p3 = {48.04235031088112, 14.416873929905307};
  geo::latlng t4_p4 = {48.05994334965749, 14.598990352133825};

  matches = tt.lookup_td_stops(t4_p1);
  ASSERT_EQ(matches.size(), 2);

  ASSERT_NO_THROW({ EXPECT_EQ(matches[0].area_idxs_, areas.at("l_g_4")); });
  EXPECT_TRUE(matches[0].locations_.empty());
  ASSERT_EQ(matches[0].location_geojsons_.size(), 1);
  ASSERT_NO_THROW({
    EXPECT_EQ(matches[0].location_geojsons_[0], geojson.at("Wien-Umgebung"));
  });

  ASSERT_NO_THROW({ EXPECT_EQ(matches[1].area_idxs_, areas.at("l_g_5")); });
  EXPECT_TRUE(matches[0].locations_.empty());
  ASSERT_EQ(matches[1].location_geojsons_.size(), 1);
  ASSERT_NO_THROW({
    EXPECT_EQ(matches[1].location_geojsons_[0], geojson.at("Wien-Umgebung2"));
  });

  matches = tt.lookup_td_stops(t4_p2);
  ASSERT_EQ(matches.size(), 1);
  ASSERT_NO_THROW({ EXPECT_EQ(matches[0].area_idxs_, areas.at("l_g_4")); });
  ASSERT_EQ(matches[0].location_geojsons_.size(), 1);
  EXPECT_TRUE(matches[0].locations_.empty());
  ASSERT_NO_THROW({
    EXPECT_EQ(matches[0].location_geojsons_[0], geojson.at("Wien-Umgebung"));
  });

  matches = tt.lookup_td_stops(t4_p3);
  EXPECT_TRUE(matches.empty());

  matches = tt.lookup_td_stops(t4_p4);
  ASSERT_EQ(matches.size(), 1);
  ASSERT_NO_THROW({ EXPECT_EQ(matches[0].area_idxs_, areas.at("l_g_4")); });
  ASSERT_EQ(matches[0].location_geojsons_.size(), 1);
  EXPECT_TRUE(matches[0].locations_.empty());
  ASSERT_NO_THROW({
    EXPECT_EQ(matches[0].location_geojsons_[0], geojson.at("Wien-Umgebung"));
  });
}