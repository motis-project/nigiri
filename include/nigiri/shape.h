#pragma once

#include "geo/polyline.h"

#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri {

geo::polyline get_shape(trip_idx_t trip_idx,
                        timetable const&,
                        shape_vecvec_t const* const);

geo::polyline get_shape(shape_idx_t const, shape_vecvec_t const* const);

}  // namespace nigiri