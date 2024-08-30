#include "nigiri/shape.h"

namespace nigiri {

geo::polyline get_shape(trip_idx_t const trip_idx,
                        timetable const& tt,
                        shape_vecvec_t const& shapes) {
  if (trip_idx == trip_idx_t::invalid() ||
      trip_idx >= tt.trip_shape_indices_.size()) {
    return {};
  }
  return get_shape(tt.trip_shape_indices_[trip_idx], shapes);
}

geo::polyline get_shape(shape_idx_t const shape_idx,
                        shape_vecvec_t const& shapes) {
  if (shape_idx == shape_idx_t::invalid() || shape_idx >= shapes.size()) {
    return {};
  }
  auto const& bucket = shapes[shape_idx];
  return geo::polyline(bucket.begin(), bucket.end());
}

}  // namespace nigiri