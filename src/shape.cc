#include "nigiri/shape.h"
#include "nigiri/timetable.h"

namespace nigiri {

geo::polyline get_shape(
    trip_idx_t trip_idx, timetable const& tt,
    mm_vecvec<shape_idx_t, geo::latlng> const* const shape_vecvec) {
if (shape_vecvec == nullptr || trip_idx == trip_idx_t::invalid()) {
    return {};
}
return get_shape(tt.trip_shape_indices_.at(trip_idx), shape_vecvec);
}

geo::polyline get_shape(
    shape_idx_t const shape_idx,
    mm_vecvec<shape_idx_t, geo::latlng> const* const shape_vecvec) {
  if (shape_vecvec == nullptr || shape_idx == shape_idx_t::invalid()) {
    return {};
  }
  auto const& bucket = shape_vecvec->at(shape_idx);
  return geo::polyline(bucket.begin(), bucket.end());
}

}