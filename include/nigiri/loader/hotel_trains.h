#pragma once

#include <span>

#include "geo/latlng.h"

#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader {

struct import_context;

// Stationary night "trains" (hotel stays): every station served by rail or
// coach with a hotel within kHotelRadius gets a kNight route that departs
// there at 22:00 local time and arrives at the same station at 08:00 the next
// day, so night-aware criteria can sleep in place instead of on a train.
// Registered as a new source; call before finalize().
constexpr auto const kHotelRadius = 300.0;  // [m]
constexpr auto const kHotelDeparture = duration_t{22 * 60};
constexpr auto const kHotelDuration = duration_t{10 * 60};

void add_hotel_night_trains(timetable&,
                            import_context&,
                            std::span<geo::latlng const> hotels);

}  // namespace nigiri::loader
