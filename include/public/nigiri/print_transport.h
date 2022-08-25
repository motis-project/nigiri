#pragma once

#include <iosfwd>

#include "nigiri/timetable.h"

namespace nigiri {

void print_transport(timetable const&,
                     std::ostream&,
                     transport,
                     bool with_debug = false);

void print_transport(timetable const&,
                     std::ostream&,
                     transport,
                     interval<unsigned> stop_range,
                     unsigned indent = 0U,
                     bool with_debug = false);

}  // namespace nigiri