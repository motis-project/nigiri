#pragma once

#include <map>
#include <string>

#include "nigiri/loader/gtfs/trip.h"

namespace nigiri::loader::gtfs {

void read_stop_times(timetable&,
                     trip_data&,
                     locations_map const&,
                     std::string_view file_content);

}  // namespace nigiri::loader::gtfs
