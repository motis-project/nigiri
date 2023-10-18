#pragma once

#include "nigiri/loader/dir.h"
#include "nigiri/loader/gtfs/agency.h"
#include "nigiri/loader/gtfs/tz_map.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include <filesystem>

namespace nigiri::loader::netex {

void load_timetable(source_idx_t, dir const&, timetable&);

}  // namespace nigiri::loader::netex