#pragma once

#include <span>

#include "geo/latlng.h"

#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri {

std::span<geo::latlng const> get_shape(timetable const&,
                                       shape_vecvec_t const&,
                                       trip_idx_t const);

std::span<geo::latlng const> get_shape(shape_vecvec_t const&,
                                       shape_idx_t const);

}  // namespace nigiri