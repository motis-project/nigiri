#pragma once

#include "nigiri/types.h"
#include "agency.h"
#include "services.h"

#include <bitset>

namespace nigiri {
struct timetable;
}

namespace nigiri::loader::gtfs {
constexpr auto kRealTimeBooking = 0;
constexpr auto kSameDayBooking = 1;
constexpr auto kPriorDaysBooking = 2;

using booking_rule_map_t = hash_map<std::string_view, booking_rule_idx_t>;

booking_rule_map_t read_booking_rules(traffic_days_t const& services,
                                      timetable& tt,
                                      std::string_view file_content);
}  // namespace nigiri::loader::gtfs
