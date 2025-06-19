#pragma once

#include <string>

#include "nigiri/loader/gtfs/tz_map.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader::gtfs {

using seated_transfers_map =
    hash_map<std::string /* from_trip_id */, std::vector<std::string>>;

using stops_map_t = hash_map<std::string, location_idx_t>;

std::pair<stops_map_t, seated_transfers_map> read_stops(
    source_idx_t,
    timetable&,
    tz_map&,
    std::string_view stops_file_content,
    std::string_view transfers_file_content,
    unsigned link_stop_distance);

}  // namespace nigiri::loader::gtfs
