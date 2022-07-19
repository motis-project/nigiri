#pragma once

#include "nigiri/section_db.h"
#include "nigiri/timetable.h"

namespace nigiri::loader {

void build_footpaths(info_db& db, timetable& tt);

}  // namespace nigiri::loader