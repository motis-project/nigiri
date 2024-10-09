#pragma once

#include <nigiri/types.h>

struct tg_geom;

namespace nigiri::loader::gtfs_flex {
  enum class GEOMETRY_TYPE {
    POINT,
    POLYGON,
    MULTIPOLYGON,
    UNKOWN
  };

  using td_location_geojson_map_t = hash_map<std::string, std::pair<GEOMETRY_TYPE, tg_geom*>>;

  td_location_geojson_map_t read_td_location_geojson(std::string_view file_content);

}  // namespace nigiri::loader::gtfs_flex