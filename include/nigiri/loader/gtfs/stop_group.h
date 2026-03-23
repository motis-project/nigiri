#pragma once

#include <string_view>

#include "nigiri/loader/gtfs/stop.h"
#include "nigiri/timetable.h"

namespace nigiri::loader::gtfs {

void add_stop_groups(timetable&,
                     std::string_view stop_group_elements_file_content,
                     stops_map_t const&);

}  // namespace nigiri::loader::gtfs
