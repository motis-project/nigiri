#pragma once

#include <filesystem>
#include <optional>
#include <vector>

#include "date/date.h"

#include "nigiri/loader/assistance.h"
#include "nigiri/common/interval.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::loader {

struct loader_config;

timetable load(std::vector<std::filesystem::path> const&,
               loader_config const&,
               interval<date::sys_days> const&,
               std::optional<shape_vecvec_t>&,
               assistance_times*,
               bool ignore = false);

}  // namespace nigiri::loader