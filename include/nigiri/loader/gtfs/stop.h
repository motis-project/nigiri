#pragma once

#include <string>

#include "nigiri/loader/gtfs/translations.h"
#include "nigiri/loader/gtfs/tz_map.h"
#include "nigiri/loader/register.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader::gtfs {

using seated_transfers_map_t =
    hash_map<std::string /* from_trip_id */,
             std::vector<std::string> /* to trip ids */>;

using stops_map_t = hash_map<std::string, location_idx_t>;

std::pair<stops_map_t, seated_transfers_map_t> read_stops(
    source_idx_t,
    timetable&,
    translator&,
    tz_map&,
    std::string_view stops_file_content,
    std::string_view transfers_file_content,
    unsigned link_stop_distance,
    script_runner const& = script_runner{});

}  // namespace nigiri::loader::gtfs
