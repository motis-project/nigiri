#pragma once

#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::rt {

void resolve_conflicts(timetable const&, rt_timetable&);

}  // namespace nigiri::rt
