#pragma once

#include <vector>

#include "geo/polyline.h"

#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri {

geo::polyline get_shape(trip_idx_t const,
                        timetable const&,
                        shape_vecvec_t const&);

geo::polyline get_shape(shape_idx_t const, shape_vecvec_t const&);

}  // namespace nigiri