#include "nigiri/loader/gtfs-flex/td_locationGeojson.h"

#include <tg.h>
#include "nlohmann/json.hpp"
#include "boost/algorithm/string.hpp"

namespace nigiri::loader::gtfs_flex {


  td_location_geojson_map_t read_td_location_geojson(std::string_view file_content) {
    const std::string GEOJSON_FEATURE_KEYWORD = "features";
    const std::string GEOJSON_ID_KEYWORD = "id";
    const std::string GEOJSON_TYPE_KEYWORD = "type";
    const std::string GEOJSON_POLYGON_VALUE = "polygon";
    const std::string GEOJSON_MULTIPOLYGON_VALUE = "multipolygon";


    const auto locations_geojson = nlohmann::json::parseparse(file_content);
    td_location_geojson_map_t location_geojson_map{};
    for(nlohmann::json::iterator feature = locations_geojson.begin(); feature != locations_geojson.end(); ++feature) {
        tg_geom *location = tg_parse_geojson(*feature.c_str());
        if(tg_geom_error(location)) {
          auto err_msg = tg_geom_error(location);
          printf("%s\n", err_msg);
          tg_geom_free(location);
          assert(false);
        }
        switch (boost::to_lower(feature[GEOJSON_TYPE_KEYWORD])) {
          case GEOJSON_POLYGON_VALUE:
            location_geojson_map.emplace( feature[GEOJSON_ID_KEYWORD],
                                          std::make_pair(GEOMETRY_TYPE::POLYGON, location));
          break;
          case GEOJSON_MULTIPOLYGON_VALUE:
            location_geojson_map.emplace( feature[GEOJSON_ID_KEYWORD],
                                          std::make_pair(GEOMETRY_TYPE::MULTIPOLYGON, location));
          break;
          default:
            printf("Unsupported feature type in location.geojson!");
        }
    }
    return location_geojson_map;
  }

}  // namespace nigiri::loader::gtfs_flex