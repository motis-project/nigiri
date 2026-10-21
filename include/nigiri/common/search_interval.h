#pragma once

#include <variant>

#include "nigiri/common/interval.h"
#include "nigiri/types.h"

namespace nigiri {

interval<unixtime_t> start_time_interval(
    std::variant<unixtime_t, interval<unixtime_t>> const&);

interval<unixtime_t> reachable_events(direction,
                                      interval<unixtime_t> const& start_itv,
                                      duration_t max_reach);

}  // namespace nigiri
