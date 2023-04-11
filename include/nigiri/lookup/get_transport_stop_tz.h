#pragma once

#include "nigiri/types.h"

namespace nigiri {

struct timetable;

timezone_idx_t get_transport_stop_tz(timetable const&,
                                     transport_idx_t,
                                     location_idx_t);

}  // namespace nigiri