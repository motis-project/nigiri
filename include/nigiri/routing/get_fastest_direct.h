#pragma once

#include "nigiri/routing/query.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing {

duration_t get_fastest_direct(timetable const&, query const&, direction const);

}  // namespace nigiri::routing
