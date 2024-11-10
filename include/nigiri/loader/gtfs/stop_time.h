#pragma once

#include "booking_rule.h"

#include "nigiri/loader/gtfs/trip.h"

namespace nigiri::loader::gtfs {

enum PickupDropoffType : std::uint8_t {
  kRegularType = 0,
  kUnavailableType = 1,
  kPhoneAgencyType = 2,
  kCoordinateWithDriverType = 3
};

void read_stop_times(timetable& tt,
                     trip_data& trips,
                     locations_map const& stops,
                     std::string_view file_content);

void read_stop_times(source_idx_t src,
                     timetable&,
                     trip_data&,
                     locations_map const&,
                     booking_rule_map_t&,
                     std::string_view file_content);

}  // namespace nigiri::loader::gtfs
