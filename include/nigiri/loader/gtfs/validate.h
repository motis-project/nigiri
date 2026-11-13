#pragma once

#include <vector>

#include "nigiri/loader/load.h"

namespace nigiri::loader::gtfs {

std::vector<timetable_source> validate_sources(
    std::vector<timetable_source> const&);

}  // namespace nigiri::loader::gtfs