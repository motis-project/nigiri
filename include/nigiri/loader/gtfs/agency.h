#pragma once

#include <string_view>

#include "nigiri/loader/gtfs/translations.h"
#include "nigiri/loader/gtfs/tz_map.h"
#include "nigiri/loader/register.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader::gtfs {

using agency_map_t = hash_map<std::string, provider_idx_t>;
using agency_ticketing_map_t = hash_map<provider_idx_t, std::string>;

std::pair<agency_map_t, agency_ticketing_map_t> read_agencies(
    source_idx_t,
    timetable&,
    translator&,
    tz_map&,
    std::string_view file_content,
    std::string_view default_tz,
    script_runner const& = script_runner{});

}  // namespace nigiri::loader::gtfs
