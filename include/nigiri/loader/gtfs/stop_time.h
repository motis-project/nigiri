#pragma once

#include "booking_rule.h"

#include "nigiri/loader/gtfs/trip.h"

namespace nigiri::loader::gtfs {

struct geometry_trip_data {
  std::vector<pickup_dropoff_type> pickup_types_;
  std::vector<booking_rule_idx_t> pickup_booking_rules_;
  std::vector<pickup_dropoff_type> dropoff_types;
  std::vector<booking_rule_idx_t> dropoff_booking_rules_;
  std::vector<stop_window> stop_windows_;
};

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
