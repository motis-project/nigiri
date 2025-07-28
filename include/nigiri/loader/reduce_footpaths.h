#pragma once

#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri {
struct timatable;
}

namespace nigiri::loader {

vecvec<location_idx_t, footpath> reduce_footpaths(
    timetable&, vecvec<location_idx_t, footpath> const&, std::size_t n);

}