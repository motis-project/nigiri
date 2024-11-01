#pragma once

#include "nigiri/types.h"
#include "services.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader::gtfs {

enum Booking_type : std::uint8_t {
  kRealTimeBooking = 0,
  kSameDayBooking = 1,
  kPriorDaysBooking = 2
};

using booking_rule_map_t = hash_map<std::string_view, booking_rule_idx_t>;

booking_rule_map_t read_booking_rules(traffic_days_t const& services,
                                      timetable& tt,
                                      std::string_view file_content);
}  // namespace nigiri::loader::gtfs
