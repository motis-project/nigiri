#pragma once

#include <iosfwd>

#include "nigiri/common/interval.h"
#include "nigiri/types.h"

namespace nigiri {

struct timetable;
struct rt_timetable;

void print_transport(timetable const&,
                     rt_timetable const*,
                     std::ostream&,
                     transport,
                     interval<stop_idx_t> stop_range,
                     unsigned indent = 0U,
                     bool with_debug = false);

void print_transport(timetable const&,
                     rt_timetable const*,
                     std::ostream&,
                     transport,
                     bool with_debug = false);

void print_transport(timetable const&,
                     rt_timetable const*,
                     std::ostream&,
                     rt_transport_idx_t,
                     interval<stop_idx_t> stop_range,
                     unsigned indent = 0U,
                     bool with_debug = false);

void print_transport(timetable const&,
                     rt_timetable const*,
                     std::ostream&,
                     rt_transport_idx_t,
                     bool with_debug = false);

}  // namespace nigiri
