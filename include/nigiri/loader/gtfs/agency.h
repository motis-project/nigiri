#pragma once

#include "nigiri/loader/register.h"

#include <string_view>

#include "nigiri/loader/gtfs/tz_map.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader::gtfs {

using agency_map_t = hash_map<std::string, provider_idx_t>;

agency_map_t read_agencies(source_idx_t,
                           timetable&,
                           tz_map&,
                           std::string_view file_content,
                           script_runner const& = script_runner{});

}  // namespace nigiri::loader::gtfs
