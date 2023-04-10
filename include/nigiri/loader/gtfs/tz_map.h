#pragma once

#include <string>

#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader::gtfs {

using tz_map = hash_map<std::string, timezone_idx_t>;

timezone_idx_t get_tz_idx(timetable& tt, tz_map&, std::string_view tz_name);

}  // namespace nigiri::loader::gtfs
