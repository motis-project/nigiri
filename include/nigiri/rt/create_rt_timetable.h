#pragma once

#include "date/date.h"

namespace nigiri {
struct timetable;
struct rt_timetable;
}  // namespace nigiri

namespace nigiri::rt {

rt_timetable create_rt_timetable(timetable const& tt, date::sys_days);

}  // namespace nigiri::rt