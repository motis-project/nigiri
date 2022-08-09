#pragma once

#include <iosfwd>

#include "nigiri/timetable.h"

namespace nigiri {

void print_transport(timetable const&, std::ostream&, trip_idx_t, day_idx_t);

}  // namespace nigiri