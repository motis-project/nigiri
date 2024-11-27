#include <nigiri/loader/gtfs/booking_rule.h>

#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/location_geojson.h"
#include "nigiri/loader/gtfs/parse_date.h"
#include "nigiri/loader/gtfs/parse_time.h"
#include "nigiri/geometry.h"
#include "nigiri/timetable.h"

#include "./test_data.h"

using namespace nigiri;
using namespace nigiri::loader::gtfs;

using p = point;
using r = ring;
using pol = polygon;
using mul = multipolgyon;

TEST(gtfs, location_geojson) {
  timetable tt;
  source_idx_t src = source_idx_t{0};

  auto const files = example_files();

  auto const geojson = read_location_geojson(
      src, tt, files.get_file(kLocationGeojsonFile).data());

  auto const test_geojson = [&](std::string const& key,
                                tg_geom* expected_geom) {
    // ASSERT_NO_THROW({
    auto const l_idx = geojson.at(key);
    auto& multipolygon_container = tt.location_geojson_geometries.at(l_idx);
    auto actual_type = multipolygon_container.original_type_;
    auto expected_type = tg_geom_typeof(expected_geom);

    ASSERT_TRUE(actual_type == TG_POINT || actual_type == TG_POLYGON ||
                actual_type == TG_MULTIPOLYGON);
    ASSERT_EQ(actual_type, expected_type);
    switch (actual_type) {
      case TG_POINT: {
        auto const point = point_from_multipolygon(multipolygon_container);
        auto const actual_point = create_tg_point(point);
        auto const expected_point = tg_geom_point(expected_geom);
        EXPECT_EQ(actual_point.x, expected_point.x);
        EXPECT_EQ(actual_point.y, expected_point.y);
        break;
      }
      case TG_POLYGON: {
        auto const poly = polygon_from_multipolygon(multipolygon_container);
        auto* actual_polygon = create_tg_poly(poly);
        EXPECT_TRUE(tg_geom_covers(reinterpret_cast<tg_geom*>(actual_polygon),
                                   expected_geom));
        EXPECT_TRUE(tg_geom_covers(expected_geom,
                                   reinterpret_cast<tg_geom*>(actual_polygon)));
        break;
      }
      case TG_MULTIPOLYGON: {
        auto* actual_multipolygon = create_tg_multipoly(multipolygon_container);
        EXPECT_TRUE(tg_geom_covers(actual_multipolygon, expected_geom));
        EXPECT_TRUE(tg_geom_covers(expected_geom, actual_multipolygon));
        break;
      }
    };
    // });
  };
  //

  auto const multipoly =
      mul{{pol{r{p{102.0, 2.0}, p{103.0, 2.0}, p{103.0, 3.0}, p{102.0, 3.0},
                 p{102.0, 2.0}},
               {}},
           pol{r{p{100.0, 0.0}, p{101.0, 0.0}, p{101.0, 1.0}, p{100.0, 1.0},
                 point{100.0, 0.0}},
               {r{p{100.2, 0.2}, p{100.2, 0.8}, p{100.8, 0.8}, p{100.8, 0.2},
                  p{100.2, 0.2}}}}}};
  //
  auto expected_multipolygon = create_tg_multipoly(multipoly);
  test_geojson("l_geo_1", expected_multipolygon);

  auto poly = pol{r{p{100.0, 0.0}, p{101.0, 0.0}, p{101.0, 1.0}, p{100.0, 1.0},
                    p{100.0, 0.0}},
                  {}};

  auto expected_polygon = create_tg_poly(poly);
  test_geojson("l_geo_2", (tg_geom*)expected_polygon);

  auto expected_point = tg_geom_new_point(tg_point{.x = 100.0, .y = 0.0});
  test_geojson("l_geo_3", expected_point);
}
