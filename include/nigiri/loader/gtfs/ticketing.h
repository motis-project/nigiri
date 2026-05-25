#pragma once

#include "nigiri/types.h"

namespace nigiri {
struct timetable;
struct dir;
struct trip_data;
struct route_map_t;
struct stops_map_t;
struct agency_ticketing_map_t;
}  // namespace nigiri

namespace nigiri::loader::gtfs {

void load_ticketing(timetable& tt,
                    dir const& d,
                    agency_ticketing_map_t const& agency_ticketing,
                    stops_map_t const& stops,
                    route_map_t const& routes,
                    trip_data const& trips,
                    source_idx_t const src);

}  // namespace nigiri::loader::gtfs