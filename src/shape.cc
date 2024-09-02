#include "nigiri/shape.h"

namespace nigiri {

std::span<geo::latlng const> get_shape(timetable const& tt,
                                       shape_vecvec_t const& shapes,
                                       trip_idx_t const trip_idx) {
  if (trip_idx == trip_idx_t::invalid() ||
      trip_idx >= tt.trip_shape_indices_.size()) {
    return {};
  }
  return get_shape(shapes, tt.trip_shape_indices_[trip_idx]);
}

std::span<geo::latlng const> get_shape(shape_vecvec_t const& shapes,
                                       shape_idx_t const shape_idx) {
  if (shape_idx == shape_idx_t::invalid() || shape_idx >= shapes.size()) {
    return {};
  }
  auto const& bucket = shapes[shape_idx];
  return {bucket.begin(), bucket.end()};
}

}  // namespace nigiri