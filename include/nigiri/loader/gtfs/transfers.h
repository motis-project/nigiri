#pragma once

#include <string_view>

#include "nigiri/loader/gtfs/stop.h"
#include "nigiri/loader/gtfs/trip.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader::gtfs {

void read_transfers(source_idx_t,
                    timetable&,
                    std::string_view file_content,
                    stops_map_t const&,
                    trip_data&,
                    bool adjust_footpaths);

}  // namespace nigiri::loader::gtfs
