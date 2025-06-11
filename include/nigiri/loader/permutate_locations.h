#pragma once

#include <algorithm>
#include <ranges>
#include <vector>

#include "nigiri/footpath.h"
#include "nigiri/stop.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri {

void permutate_locations(timetable& tt);

}  // namespace nigiri