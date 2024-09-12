#pragma once

#include <filesystem>
#include <vector>

#include "date/date.h"

#include "nigiri/loader/assistance.h"
#include "nigiri/common/interval.h"
#include "nigiri/timetable.h"
#include "nigiri/shape.h"
#include "nigiri/types.h"

namespace nigiri::loader {

struct loader_config;

timetable load(std::vector<std::filesystem::path> const&,
               loader_config const&,
               interval<date::sys_days> const&,
               shapes_storage&,
               assistance_times*,
               bool ignore = false);

}  // namespace nigiri::loader