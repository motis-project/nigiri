#include "nigiri/loader/gtfs/location_geojson.h"

#include <utl/raii.h>

#include <tg.h>
#include <boost/json.hpp>

#include <nigiri/timetable.h>

namespace nigiri::loader::gtfs {
location_geojson_map_t read_location_geojson(source_idx_t src,
                                             timetable& tt,
                                             std::string_view file_content) {
  // constexpr char* const kGeojsonFeaturesKey = "features";
  // constexpr char* const kGeojsonIDKey = "id";
  // constexpr char* const kGeojsonGeometryKey = "geometry";
  // constexpr char* const kGeojsonTypeKey = "type";
  // constexpr char* const kGeojsonPointValue = "point";
  // constexpr char* const kGeojsonPolygonValue = "polygon";
  // constexpr char* const kGeojsonMultipolygonValue = "multipolygon";

  location_geojson_map_t location_geojson{};

  boost::system::error_code ec;
  const auto json = boost::json::parse(file_content, ec);
  if (ec) {
    // TODO log error
    return location_geojson;
  }

  auto featureCollection = json.as_object();
  auto const features = featureCollection["features"].as_array();
  for (auto feature = features.begin(); feature != features.end(); ++feature) {
    auto const feature_object = feature->as_object();
    auto const id = std::string(feature_object.at("id").as_string());
    auto const geometry_object = feature_object.at("geometry").as_object();

    const auto feature_content = feature->as_string();
    auto type = TG_GEOMETRYCOLLECTION;
    auto const type_name = std::string(geometry_object.at("type").as_string());
    if (type_name == "point") {
      type = TG_POINT;
    } else if (type_name == "polygon") {
      type = TG_POLYGON;
    } else if (type_name == "multipolygon") {
      type = TG_MULTIPOLYGON;
    } else {
      // TODO log error
    }

    // const auto ptr =
    //     utl::raii(tg_parse_geojson(feature_content.c_str()), tg_geom_free);
    const auto idx = tt.register_location_geojson(
        src, id, type, tg_parse_geojson(feature_content.c_str()));
    location_geojson.emplace(id, idx);
  }
  return location_geojson;
}

}  // namespace nigiri::loader::gtfs