#pragma once

#include "nigiri/footpath.h"
#include "nigiri/stop.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include <algorithm>
#include <ranges>
#include <vector>

namespace nigiri {

void permutate_locations(timetable& tt, std::uint32_t const first_idx);

}  // namespace nigiri