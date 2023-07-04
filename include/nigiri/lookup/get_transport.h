#pragma once

#include <optional>
#include <string>

#include "date/date.h"

#include "nigiri/types.h"

namespace nigiri {

struct timetable;

std::optional<std::pair<transport, interval<std::uint16_t>>> get_ref_transport(
    timetable const& tt,
    trip_id const& id,
    date::year_month_day const day,
    bool const gtfs_local_day);

}  // namespace nigiri