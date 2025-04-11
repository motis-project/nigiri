#pragma once

#include <optional>
#include <string_view>

#include "nigiri/rt/rt_timetable.h"
#include "nigiri/rt/run.h"
#include "nigiri/timetable.h"

#include "gtfsrt/gtfs-realtime.pb.h"

namespace nigiri::rt {

std::pair<date::days, duration_t> split_rounded(duration_t);

std::pair<run, trip_idx_t> gtfsrt_resolve_run(
    date::sys_days const today,
    timetable const&,
    rt_timetable const*,
    source_idx_t,
    transit_realtime::TripDescriptor const&,
    std::optional<std::string_view> trip_id = std::nullopt);

}  // namespace nigiri::rt