#pragma once

#include <array>
#include <optional>

#include "date/date.h"

#include "nigiri/loader/gtfs/agency.h"
#include "nigiri/common/interval.h"
#include "nigiri/types.h"

namespace nigiri::loader::gtfs {

// Provides the noon offset in minutes
// for each GTFS local service day for all timezones
// referenced in agencies.txt
using noon_offset_hours_t =
    vector_map<timezone_idx_t, std::optional<std::array<duration_t, kMaxDays>>>;

duration_t get_noon_offset(date::local_days const days,
                           date::time_zone const* tz);

noon_offset_hours_t precompute_noon_offsets(timetable const& tt,
                                            agency_map_t const& agencies);

}  // namespace nigiri::loader::gtfs