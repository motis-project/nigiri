#pragma once

#include <filesystem>
#include <vector>

#include "date/date.h"

#include "nigiri/common/interval.h"
#include "nigiri/timetable.h"

namespace nigiri::loader {

struct loader_config;

timetable load(std::vector<std::filesystem::path> const&,
               loader_config const&,
               interval<date::sys_days> const&);

}  // namespace nigiri::loader