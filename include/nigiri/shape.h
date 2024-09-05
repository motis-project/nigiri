#pragma once

#include <span>

#include "geo/latlng.h"

#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri {

std::span<geo::latlng const> get_shape(timetable const&,
                                       shapes_storage_t const&,
                                       trip_idx_t);

std::span<geo::latlng const> get_shape(shapes_storage_t const&, shape_idx_t);

}  // namespace nigiri