#include <nigiri/loader/gtfs/booking_rule.h>

#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/location_geojson.h"
#include "nigiri/loader/gtfs/parse_date.h"
#include "nigiri/loader/gtfs/parse_time.h"
#include "nigiri/timetable.h"

#include "./test_data.h"

using namespace nigiri;
using namespace nigiri::loader::gtfs;

TEST(gtfs, location_geojson) {
  // timetable tt;
  // source_idx_t src = source_idx_t{0};
  //
  // auto const files = example_files();
  //
  // auto const geojson = read_location_geojson(
  //     src, tt, files.get_file(kLocationGeojsonFile).data());
  //
  // auto const test_geojson = [&](std::string key, tg_geom_type expected_type,
  //                               tg_geom* expected_geom) {
  //   ASSERT_NO_THROW({
  //     auto const l_idx = geojson.at(key);
  //     auto const actual_type = tt.location_geojson_types_.at(l_idx.v_());
  //     auto const* actual_geo =
  //     &tt.locations_geojson_geometries_.at(l_idx.v_());
  //
  //     // EXPECT_EQ(actual_type, expected_type);
  //     if (actual_type == expected_type) {
  //       switch (actual_type) {
  //         case TG_POINT:
  //           auto const actual_point = tg_geom_point(**actual_geo);
  //           // ASSERT_FALSE(tg_geom_error(actual_point));
  //           auto const expected_point = tg_geom_point(expected_geom);
  //           // EXPECT_EQ(actual_point.x, expected_point.x);
  //           // EXPECT_EQ(actual_point.y, expected_point.y);
  //           break;
  //         case TG_POLYGON:
  //           EXPECT_TRUE(tg_geom_equals(**actual_geo, expected_geom));
  //           break;
  //         case TG_MULTIPOINT:
  //           auto const numPolys = tg_geom_num_polys(**actual_geo);
  //           for (int i = 0; i < numPolys; i++) {
  //             auto const actual_poly = tg_geom_poly_at(**actual_geo, i);
  //             auto const expected_poly = tg_geom_poly_at(expected_geom, i);
  //             ASSERT_TRUE(actual_poly != nullptr);
  //             ASSERT_TRUE(expected_poly != nullptr);
  //             EXPECT_TRUE(tg_geom_equals((tg_geom*)actual_poly,
  //                                        (tg_geom*)expected_geom));
  //           }
  //           break;
  //         default:;
  //       };
  //     }
  //   });
  // };
  //
  // auto const* expected_multipolygon =
  //     &tg_point{.x = 0, .y = 0};  // TODO Solution Multipolygon
  // test_geojson("l_geo_1", TG_POINT, (tg_geom*)expected_multipolygon);
  //
  // auto const* expected_polygon =
  //     &tg_point{.x = 0, .y = 0};  // TODO Solution Polygon
  // test_geojson("l_geo_2", TG_POLYGON, (tg_geom*)expected_polygon);
  //
  // auto const* expected_point = &tg_point{.x = 100.0, .y = 0.0};
  // test_geojson("l_geo_3", TG_MULTIPOLYGON, (tg_geom*)expected_point);
}
