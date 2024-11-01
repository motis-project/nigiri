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

area_map_t read_areas(source_idx_t const stop_areas_src,
                      source_idx_t const location_groups_src,
                      source_idx_t const location_groups_stops_src,
                      source_idx_t const location_src,
                      source_idx_t const geojson_src,
                      timetable& tt,
                      locations_map const& location_id_to_idx,
                      location_geojson_map_t const& geojson_id_to_idx,
                      std::string_view const stop_areas_content,
                      std::string_view const location_groups_content,
                      std::string_view const location_group_stops_content);

area_map_t read_areas(source_idx_t const area_src,
                      source_idx_t const location_src,
                      source_idx_t const geojson_src,
                      timetable& tt,
                      locations_map const& location_id_to_idx,
                      location_geojson_map_t const& geojson_id_to_idx,
                      std::string_view const file_content);
}  // namespace nigiri::loader::gtfs