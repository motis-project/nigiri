#pragma once

#include "nigiri/rt/frun.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

namespace nigiri {

rt::frun resolve(timetable const& tt,
                 rt_timetable const* rtt,
                 std::string_view trip_id,
                 std::string_view start_date = "",
                 std::string_view start_time = "");

}  // namespace nigiri
