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

using stops_map_t = hash_map<std::string, location_idx_t>;

using location_accessible_map_t = hash_map<location_idx_t, bool>;

std::pair<stops_map_t, location_accessible_map_t> read_stops(
    source_idx_t,
    timetable&,
    translator&,
    tz_map&,
    std::string_view stops_file_content,
    unsigned link_stop_distance,
    duration_t default_transfer_time = duration_t{2},
    script_runner const& = script_runner{});

}  // namespace nigiri::loader::gtfs
