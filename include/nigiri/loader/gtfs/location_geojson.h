#pragma once

#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader::gtfs {
using location_geojson_map_t =
    hash_map<std::string_view, location_geojson_idx_t>;

location_geojson_map_t read_location_geojson(timetable& tt,
                                             std::string_view file_content);
}  // namespace nigiri::loader::gtfs
