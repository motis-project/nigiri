#pragma once

#include <cstdint>

#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::rt {

void resolve_conflicts(timetable const&, rt_timetable&);

std::uint64_t validate_transfer_times(timetable const&,
                                      rt_timetable const&,
                                      interval<unixtime_t>);

}  // namespace nigiri::rt
