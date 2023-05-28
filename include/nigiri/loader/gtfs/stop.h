#pragma once

#include <string>

#include "nigiri/loader/gtfs/tz_map.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader::gtfs {

using locations_map = hash_map<std::string, location_idx_t>;

locations_map read_stops(source_idx_t,
                         timetable&,
                         tz_map&,
                         std::string_view stops_file_content,
                         std::string_view transfers_file_content,
                         unsigned link_stop_distance);

}  // namespace nigiri::loader::gtfs
