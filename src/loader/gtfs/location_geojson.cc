#include "nigiri/loader/gtfs/location_geojson.h"

#include <nigiri/timetable.h>

#include "boost/json.hpp"
#include "tg.h"

namespace nigiri::loader::gtfs {
location_geojson_map_t read_location_geojson(timetable& tt,
                                             std::string_view file_content) {
  const std::string kGeojsonFeaturesKey = "features";
  const std::string kGeojsonIDKey = "id";
  const std::string kGeojsonGeometryKey = "geometry";
  const std::string kGeojsonTypeKey = "type";
  const std::string kGeojsonPointValue = "point";
  const std::string kGeojsonPolygonValue = "polygon";
  const std::string kGeojsonMultipolygonValue = "multipolygon";

  location_geojson_map_t location_geojson{};

  try {
    boost::system::error_code ec;
    const auto json = boost::json::parse(file_content, ec);
    if (ec) {
      // TODO log error
    } else {
      auto featureCollection = json.as_object();
      const auto features = featureCollection[kGeojsonFeaturesKey].as_array();
      for (auto feature = features.begin(); feature != features.end();
           ++feature) {
        const auto id = feature[kGeojsonIDKey].as_string();
        const auto geometry_object = feature[kGeojsonGeometryKey].as_object();

        const auto feature_content = feature->as_string();
        auto type = TG_GEOMETRYCOLLECTION;
        switch (geometry_object[kGeojsonTypeKey].as_string()) {
          case kGeojsonPointValue: type = TG_POINT; break;
          case kGeojsonPolygonValue: type = TG_POLYGON; break;
          case kGeojsonMultipolygonValue: type = TG_MULTIPOLYGON; break;
        }

        tg_geom* area = tg_parse_geojson(feature_content.c_str());
        const auto ptr = std::make_shared<tg_geom*>(area, tg_geom_free);
        const auto idx = tt.register_location_geojson(id, type, ptr);
        location_geojson.emplace(id, idx);
      }
    }
    return location_geojson;
  }

  catch (std::bad_alloc const& e) {
    // TODO log error
  }
}

}  // namespace nigiri::loader::gtfs