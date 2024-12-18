#pragma once

#include "nigiri/common/interval.h"
#include "nigiri/types.h"

namespace nigiri {

struct timetable;

timetable slice(timetable const&, interval<date::sys_days>);

}  // namespace nigiri