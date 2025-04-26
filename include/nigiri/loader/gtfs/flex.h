#pragma once

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/services.h"
#include "nigiri/loader/gtfs/stop.h"
#include "nigiri/types.h"

namespace nigiri::loader::gtfs {

void load_flex(timetable& tt,
               dir const&,
               traffic_days_t const&,
               locations_map const&,
               hash_map<bitfield, bitfield_idx_t>&);

}  // namespace nigiri::loader::gtfs