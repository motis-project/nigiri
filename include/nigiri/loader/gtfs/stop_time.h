#pragma once

#include "booking_rule.h"

#include <map>
#include <string>

#include "nigiri/loader/gtfs/trip.h"

namespace nigiri::loader::gtfs {

constexpr auto PICKUP_DROPOFF_TYPE_REGULAR = 0;
constexpr auto PICKUP_DROPOFF_TYPE_UNAVAILABLE = 1;
constexpr auto PICKUP_DROPOFF_TYPE_PHONE_AGENCY = 2;
constexpr auto PICKUP_DROPOFF_TYPE_COORDINATE_WITH_DRIVER = 3;


void read_stop_times(timetable&,
                     trip_data&,
                     locations_map const&,
                     booking_rule_map_t&,
                     std::string_view file_content);

}  // namespace nigiri::loader::gtfs
