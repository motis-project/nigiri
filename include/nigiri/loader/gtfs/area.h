#pragma once

#include "nigiri/types.h"
#include "location_geojson.h"
#include "stop.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader::gtfs {
using area_map_t =
    hash_map<std::string, std::pair<location_id_type, area_idx_t>>;

area_map_t read_areas(const source_idx_t src,
                      timetable& tt,
                      const locations_map& location_id_to_idx,
                      const location_geojson_map_t& geojson_id_to_idx,
                      std::string_view stop_areas_content,
                      std::string_view location_groups_content,
                      std::string_view location_group_stops_content);

area_map_t read_areas(source_idx_t src,
                      timetable& tt,
                      const locations_map& location_id_to_idx,
                      const location_geojson_map_t& geojson_id_to_idx,
                      std::string_view file_content);
}  // namespace nigiri::loader::gtfs