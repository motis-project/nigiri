#include "nigiri/shape.h"

namespace nigiri {

std::span<geo::latlng const> get_shape(timetable const& tt,
                                       shapes_storage_t const& shapes,
                                       trip_idx_t const trip_idx) {
  if (trip_idx == trip_idx_t::invalid() ||
      trip_idx >= tt.trip_shape_indices_.size()) {
    return {};
  }
  return get_shape(shapes, tt.trip_shape_indices_[trip_idx]);
}

std::span<geo::latlng const> get_shape(shapes_storage_t const& shapes,
                                       shape_idx_t const shape_idx) {
  if (shape_idx == shape_idx_t::invalid() || shape_idx >= shapes.size()) {
    return {};
  }
  auto const bucket = shapes[shape_idx];
  return {begin(bucket), end(bucket)};
}

}  // namespace nigiri