#pragma once

#include <filesystem>
#include <span>

#include "geo/latlng.h"

#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri {

shapes_storage_t create_shapes_storage(std::filesystem::path const&);

std::span<geo::latlng const> get_shape(timetable const&,
                                       shapes_storage_t const&,
                                       trip_idx_t);

std::span<geo::latlng const> get_shape(shapes_storage_t const&, shape_idx_t);

}  // namespace nigiri