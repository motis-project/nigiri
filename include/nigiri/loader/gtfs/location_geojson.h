#pragma once

#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader::gtfs {
using location_geojson_map_t = hash_map<std::string, location_geojson_idx_t>;

location_geojson_map_t read_location_geojson(source_idx_t src,
                                             timetable& tt,
                                             std::string_view file_content);
}  // namespace nigiri::loader::gtfs
