#pragma once

#include <optional>

#include "nigiri/routing/journey.h"
#include "nigiri/routing/query.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

void fix_transfer_durations(timetable const&,
                            rt_timetable const*,
                            query const&,
                            journey&,
                            std::optional<journey::run_enter_exit> prev = {},
                            std::optional<journey::run_enter_exit> next = {});

}  // namespace nigiri::routing
