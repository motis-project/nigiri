#include "nigiri/shape.h"
#include "nigiri/timetable.h"

namespace nigiri {

geo::polyline get_shape(trip_idx_t trip_idx,
                        timetable const& tt,
                        shape_vecvec_t const& shape) {
  if (trip_idx == trip_idx_t::invalid()) {
    return {};
  }
  return (trip_idx < tt.trip_shape_indices_.size())
    ? get_shape(tt.trip_shape_indices_[trip_idx], shape)
    : geo::polyline{};
}

geo::polyline get_shape(shape_idx_t const shape_idx,
                        shape_vecvec_t const& shape) {
  if (shape_idx == shape_idx_t::invalid()) {
    return {};
  }
  auto const& bucket = shape.at(shape_idx);
  return geo::polyline(bucket.begin(), bucket.end());
}

}  // namespace nigiri