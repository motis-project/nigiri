#pragma once

#include "nigiri/types.h"

// the position of the query day in the day offset
#define QUERY_DAY_SHIFT 5

namespace nigiri::routing {

static constexpr auto const maxASTravelTime = 1_days;
constexpr auto const kASMaxTravelTimeDays = 1U;
constexpr auto const kASMaxDayOffset =
    std::int8_t{kASMaxTravelTimeDays +
                kTimetableOffset / 1_days};  // +5 for timetable offset
}  // namespace nigiri::routing