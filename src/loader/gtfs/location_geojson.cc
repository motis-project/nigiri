#include "nigiri/loader/gtfs/location_geojson.h"

#include <utl/raii.h>

#include <tg.h>
#include <boost/json.hpp>

#include <nigiri/timetable.h>

namespace nigiri::loader::gtfs {
location_geojson_map_t read_location_geojson(source_idx_t src,
                                             timetable& tt,
                                             std::string_view file_content) {
  std::string constexpr kGeojsonFeaturesKey = "features";
  std::string constexpr kGeojsonIDKey = "id";
  std::string constexpr kGeojsonGeometryKey = "geometry";
  std::string constexpr kGeojsonTypeKey = "type";
  std::string constexpr kGeojsonPointValue = "point";
  std::string constexpr kGeojsonPolygonValue = "polygon";
  std::string constexpr kGeojsonMultipolygonValue = "multipolygon";

  location_geojson_map_t location_geojson{};

  boost::system::error_code ec;
  const auto json = boost::json::parse(file_content, ec);
  if (ec) {
    // TODO log error
    return location_geojson;
  }

  auto featureCollection = json.as_object();
  const auto features = featureCollection[kGeojsonFeaturesKey].as_array();
  for (auto feature = features.begin(); feature != features.end(); ++feature) {
    const auto id = feature[kGeojsonIDKey].as_string();
    const auto geometry_object = feature[kGeojsonGeometryKey].as_object();

    const auto feature_content = feature->as_string();
    auto type = TG_GEOMETRYCOLLECTION;
    switch (geometry_object[kGeojsonTypeKey].as_string()) {
      case kGeojsonPointValue: type = TG_POINT; break;
      case kGeojsonPolygonValue: type = TG_POLYGON; break;
      case kGeojsonMultipolygonValue: type = TG_MULTIPOLYGON; break;
    }

    const auto ptr =
        utl::raii(tg_parse_geojson(feature_content.c_str()), tg_geom_free);
    const auto idx = tt.register_location_geojson(id, type, ptr);
    location_geojson.emplace(id, idx);
  }
  return location_geojson;
}

}  // namespace nigiri::loader::gtfs