#pragma once

#include "booking_rule.h"

#include "nigiri/loader/gtfs/trip.h"

namespace nigiri::loader::gtfs {

void read_stop_times(timetable& tt,
                     trip_data& trips,
                     locations_map const& stops,
                     std::string_view file_content,
                     bool);

void read_stop_times(timetable&,
                     source_idx_t,
                     trip_data&,
                     location_geojson_map_t const&,
                     locations_map const&,
                     booking_rule_map_t const&,
                     std::string_view file_content,
                     bool);

}  // namespace nigiri::loader::gtfs
