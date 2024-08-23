#pragma once

#include "geo/polyline.h"

#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri {

geo::polyline get_shape(
    trip_idx_t trip_idx, timetable const&,
    mm_vecvec<shape_idx_t, geo::latlng> const* const);

geo::polyline get_shape(
    shape_idx_t const,
    mm_vecvec<shape_idx_t, geo::latlng> const* const);

}