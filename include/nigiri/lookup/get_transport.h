#pragma once

#include <optional>
#include <string>

#include "date/date.h"

#include "nigiri/types.h"

namespace nigiri {

struct timetable;

std::optional<transport> get_transport(timetable const&,
                                       std::string_view trip_idx,
                                       date::year_month_day const day);

}  // namespace nigiri