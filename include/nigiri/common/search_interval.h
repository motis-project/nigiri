#pragma once

#include <variant>

#include "nigiri/common/interval.h"
#include "nigiri/types.h"

namespace nigiri {

// A start time given either as a single point or as an interval, normalized
// to a half-open interval that is never empty: a single point becomes one
// minute.
interval<unixtime_t> start_time_interval(
    std::variant<unixtime_t, interval<unixtime_t>> const&);

// Every event time a search over the (half-open) start interval `start_itv`
// can touch: the start times themselves, widened by `max_reach` in search
// direction. Half-open, so the widening is `max_reach + 1`.
//
// Computed in 64 bit and clamped: `unixtime_t` is 32 bit minutes and adding
// `max_reach` to a start time at the end of its range would overflow.
// Clamping only ever widens the result.
interval<unixtime_t> reachable_events(direction,
                                      interval<unixtime_t> const& start_itv,
                                      duration_t max_reach);

}  // namespace nigiri
