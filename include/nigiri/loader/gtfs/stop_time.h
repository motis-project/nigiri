#pragma once

#include <map>
#include <string>

#include "nigiri/loader/gtfs/flex.h"
#include "nigiri/loader/gtfs/trip.h"

namespace nigiri::loader::gtfs {

void read_stop_times(timetable&,
                     trip_data&,
                     stops_map_t const&,
                     flex_areas_t const&,
                     booking_rules_t const&,
                     location_groups_t const&,
                     std::string_view file_content,
                     bool);

}  // namespace nigiri::loader::gtfs
