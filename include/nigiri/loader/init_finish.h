#pragma once

#include <cinttypes>
#include <limits>

#include "nigiri/loader/build_footpaths.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::loader {

void register_special_stations(timetable&);
void finalize(timetable&, finalize_options = {});

}  // namespace nigiri::loader