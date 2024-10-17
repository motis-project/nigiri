#pragma once

#include "booking_rule.h"

#include "nigiri/loader/gtfs/trip.h"

namespace nigiri::loader::gtfs {

constexpr auto kPickupDropoffRegularType = 0;
constexpr auto kPickupDropoffUnavailableType = 1;
constexpr auto kPickupDropoffPhoneAgencyType = 2;
constexpr auto kPickupDropoffCoordinateWithDriverType = 3;

void read_stop_times(timetable&,
                     trip_data&,
                     locations_map const&,
                     booking_rule_map_t&,
                     std::string_view file_content);

}  // namespace nigiri::loader::gtfs
