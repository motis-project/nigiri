#include <nigiri/loader/gtfs/booking_rule.h>

#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/location_geojson.h"

#include <nigiri/loader/gtfs/agency.h>
#include <nigiri/loader/gtfs/route.h>
#include <nigiri/loader/gtfs/stop_time.h>
#include <nigiri/loader/gtfs/trip.h>
#include <nigiri/loader/loader_interface.h>

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

  auto const files = example_files();

  auto const geojson =
      read_location_geojson(tt, files.get_file(kLocationGeojsonFile).data());

  auto const test_geojson = [&](std::string const& key,
                                tg_geom* expected_geom) {
    ASSERT_NO_THROW({
      auto const l_idx = geojson.at(key);
      auto& multipolygon_container = tt.geometry_.at(l_idx);
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
          EXPECT_TRUE(tg_geom_covers(
              expected_geom, reinterpret_cast<tg_geom*>(actual_polygon)));
          break;
        }
        case TG_MULTIPOLYGON: {
          auto* actual_multipolygon =
              create_tg_multipoly(multipolygon_container);
          EXPECT_EQ(tg_geom_num_polys(actual_multipolygon), 2);
          EXPECT_TRUE(tg_geom_covers(actual_multipolygon, expected_geom));
          EXPECT_TRUE(tg_geom_covers(expected_geom, actual_multipolygon));
          break;
        }
      };
    });
  };

  auto const multipoly =
      mul{{pol{r{p{102.0, 2.0}, p{103.0, 2.0}, p{103.0, 3.0}, p{102.0, 3.0},
                 p{102.0, 2.0}},
               {}},
           pol{r{p{100.0, 0.0}, p{101.0, 0.0}, p{101.0, 1.0}, p{100.0, 1.0},
                 point{100.0, 0.0}},
               {r{p{100.2, 0.2}, p{100.2, 0.8}, p{100.8, 0.8}, p{100.8, 0.2},
                  p{100.2, 0.2}}}}}};

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

TEST(gtfs, rtree) {
  auto const outside_hamburg =
      geo::latlng{53.707225991711624, 9.979755852932868};
  auto const inside_hamburg = geo::latlng{53.57490926352469, 9.95961326465499};

  auto const outside_brandenburg =
      geo::latlng{52.539517832998655, 13.415153651454148};
  auto const inside_brandenburg =
      geo::latlng{52.627580321336865, 13.257278815958188};

  auto const outside_duesseldorf_dortmund =
      geo::latlng{51.349468342102455, 7.065163768329711};
  auto const inside_dortmund =
      geo::latlng{51.510917931309194, 7.4491141156070455};
  auto const inside_duesseldorf =
      geo::latlng{51.179558217001215, 6.7252039269921795};

  auto const outside_frankfurt_and_mainz =
      geo::latlng{50.062363831385056, 8.387321779425122};
  auto const inside_frankfurt =
      geo::latlng{50.08479722333715, 8.73476821579439};
  auto const inside_mainz = geo::latlng{49.97632859907455, 8.22431177390385};
  auto const inside_frankfurt_and_mainz =
      geo::latlng{50.003150195331585, 8.516451158935439};

  timetable tt;

  auto const geojson = read_location_geojson(
      tt, example_files().get_file(kRtreeLocationGeojsonFile).data());

  match_t matches;

  // Single polygon
  matches = tt.lookup_td_stops(outside_hamburg);
  EXPECT_TRUE(matches.empty());
  matches = tt.lookup_td_stops(inside_hamburg);
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], geojson.at("Hamburg"));

  // Polygon with hole
  matches = tt.lookup_td_stops(outside_brandenburg);
  EXPECT_TRUE(matches.empty());
  matches = tt.lookup_td_stops(inside_brandenburg);
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], geojson.at("Brandenburg"));

  // Multipolygon
  matches = tt.lookup_td_stops(outside_duesseldorf_dortmund);
  EXPECT_TRUE(matches.empty());
  matches = tt.lookup_td_stops(inside_dortmund);
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], geojson.at("Duesseldorf-Dortmund"));
  matches = tt.lookup_td_stops(inside_duesseldorf);
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], geojson.at("Duesseldorf-Dortmund"));

  // Intersecting Polygons
  matches = tt.lookup_td_stops(outside_frankfurt_and_mainz);
  EXPECT_TRUE(matches.empty());
  matches = tt.lookup_td_stops(inside_frankfurt);
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], geojson.at("Frankfurt"));
  matches = tt.lookup_td_stops(inside_mainz);
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0], geojson.at("Mainz"));
  matches = tt.lookup_td_stops(inside_frankfurt_and_mainz);
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0], geojson.at("Frankfurt"));
  EXPECT_EQ(matches[1], geojson.at("Mainz"));
}

TEST(gtfs, register_locations_in_geometries) {
  timetable tt;
  tz_map timezones;

  auto const files = example_files();

  auto const stops = read_stops(
      source_idx_t{0}, tt, timezones,
      files.get_file(kLocationsWithinGeometriesStopFile).data(), "", 0U);

  auto const geojson = read_location_geojson(
      tt, files.get_file(kLocationsWithinGeometriesGeojsonFile).data());

  auto const outside_berlin =
      geo::latlng{52.610329253088594, 13.20597275574309};
  auto const inside_berlin = geo::latlng{52.52382076496181, 13.403639322418002};
  auto const edge_berlin = geo::latlng{52.406589559298396, 13.340254133857854};
  auto const way_outside_berlin =
      geo::latlng{52.51106243823082, 13.72059314902458};

  auto const within_hole_hannover =
      geo::latlng{52.30395481022279, 9.648938978164438};
  auto const inside_hannover =
      geo::latlng{52.070567698004766, 10.499473611847066};
  auto const hole_edge_hannover =
      geo::latlng{52.2668775457241, 10.160283681204987};
  auto const outside_hannover =
      geo::latlng{51.740426496286176, 9.032883031459278};
  auto const way_outside_hannover =
      geo::latlng{51.52928071175947, 9.934063502721187};

  struct pointer_rtree {
    pointer_rtree(timetable& tt) { tt_ = &tt; }

    timetable* tt_;

    void find(geo::box const& b,
              std::function<void(geo::latlng const&, location_idx_t const)> fn)
        const {
      for (auto i = 0; i < tt_->locations_.coordinates_.size(); ++i) {
        auto const idx = location_idx_t{i};
        if (b.contains(tt_->locations_.coordinates_[idx])) {
          fn(tt_->locations_.coordinates_[idx], idx);
        }
      }
    }
  };

  std::unique_ptr<pointer_rtree> rtree = std::make_unique<pointer_rtree>(tt);

  tt.register_locations_in_geometries(std::move(rtree));

  auto berlin_idx = geojson.at("Berlin");

  auto hannover_idx = geojson.at("Hannover-Umgebung");

  ASSERT_EQ(tt.geometry_locations_within_[berlin_idx].size(), 1);
  EXPECT_EQ(tt.geometry_locations_within_[berlin_idx][0],
            stops.at("inside_berlin"));
  // EXPECT_EQ(tt.geometry_locations_within_[berlin_idx][1],
  //           stops.at("edge_berlin"));

  ASSERT_EQ(tt.geometry_locations_within_[hannover_idx].size(), 2);
  EXPECT_EQ(tt.geometry_locations_within_[hannover_idx][0],
            stops.at("inside_hannover"));
  EXPECT_EQ(tt.geometry_locations_within_[hannover_idx][1],
            stops.at("hole_edge_hannover"));
}

// TEST(gtfs, calculate_duration) {
//   auto const files = example_files();
//
//   auto const src = source_idx_t{0};
//
//   timetable tt;
//   tt.date_range_ = interval{date::sys_days{date::July / 1 / 2006},
//                             date::sys_days{date::August / 1 / 2006}};
//   tz_map timezones;
//
//   auto const config = loader::loader_config{};
//   auto agencies =
//       read_agencies(tt, timezones, files.get_file(kAgencyFile).data());
//   auto const routes = read_routes(tt, timezones, agencies,
//                                   files.get_file(kRoutesFile).data(), "CET");
//   auto const dates =
//       read_calendar_date(files.get_file(kCalendarDatesFile).data());
//   auto const calendar = read_calendar(files.get_file(kCalenderFile).data());
//   auto const services =
//       merge_traffic_days(tt.internal_interval_days(), calendar, dates);
//   auto trip_data =
//       read_trips(tt, routes, services, {}, files.get_file(kTripsFile).data(),
//                  config.bikes_allowed_default_);
//
//   auto const geometries =
//       read_location_geojson(tt, files.get_file(kLocationGeojsonFile).data());
//
//   read_stop_times(
//       tt, src, trip_data, geometries, locations_map{}, booking_rule_map_t{},
//       files.get_file(kCalculateDurationStopTimesFile).data(), false);
//
//   auto const kNumGeometries = tt.geometry_idx_to_trip_idxs_.size();
//
//   hash_map<std::pair<geo::latlng, geo::latlng>, duration_t> durations;
//   std::vector<geo::latlng> points{};
//   for (auto i = 0; i < kNumGeometries; ++i) {
//     auto geo_idx = geometry_idx_t{i};
//     auto& geo = tt.geometry_.at(geo_idx);
//     auto center = geo.get_center();
//     points.push_back(center);
//   }
//
//   std::vector<utl::cstr> hhmm = {"00:05:00", "00:10:00", "00:15:00"};
//
//   for (auto i = 0; i < points.size(); ++i) {
//     auto p1 = points.at(i);
//     std::for_each(points.begin(), points.end(), [&](auto p2) {
//       if (p1 == p2) {
//         durations.emplace(std::make_pair(p1, p2), duration_t::zero());
//       } else {
//         auto duration = hhmm_to_min(hhmm[i]);
//         durations.emplace(std::make_pair(p1, p2), duration);
//       }
//     });
//   }
//
//   auto func = [&](auto p1, auto p2) {
//     return durations.at(std::make_pair(p1, p2));
//   };
//   tt.calculate_geometry_durations(func);
//
//   ASSERT_EQ(kNumGeometries, tt.geometry_duration_.size());
//   for (auto i = 0; i < kNumGeometries; ++i) {
//     auto key1 = geometry_idx_t{i};
//     ASSERT_EQ(kNumGeometries, tt.geometry_duration_[key1].size());
//     auto point1 = tt.geometry_[key1].get_center();
//     for (auto j = 0; j < kNumGeometries; ++j) {
//       auto key2 = geometry_idx_t{j};
//       auto point2 = tt.geometry_[key2].get_center();
//       ASSERT_TRUE(durations.contains({point1, point2}));
//       EXPECT_EQ(tt.geometry_duration_[key1].at(j),
//                 durations.at({point1, point2}));
//     }
//   }
// }
