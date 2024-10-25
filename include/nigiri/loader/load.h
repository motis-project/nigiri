#pragma once

#include <filesystem>
#include <vector>

#include "date/date.h"

#include "nigiri/loader/build_footpaths.h"
#include "nigiri/common/interval.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri {
struct shapes_storage;
}

namespace nigiri::loader {

struct assistance_times;
struct loader_config;

timetable load(std::vector<std::pair<std::string, loader_config>> const&,
               finalize_options const&,
               interval<date::sys_days> const&,
               assistance_times* = nullptr,
               shapes_storage* = nullptr,
               bool ignore = false);

}  // namespace nigiri::loader