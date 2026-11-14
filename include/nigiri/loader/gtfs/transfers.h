#pragma once

#include <string_view>

#include "nigiri/loader/gtfs/stop.h"
#include "nigiri/loader/gtfs/trip.h"
#include "nigiri/timetable.h"

namespace nigiri::loader::gtfs {

void read_transfers(source_idx_t,
                    timetable&,
                    import_context const&,
                    std::string_view file_content,
                    stops_map_t const&,
                    trip_data&,
                    bool adjust_footpaths);

}  // namespace nigiri::loader::gtfs
