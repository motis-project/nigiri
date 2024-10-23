#pragma once

#include <filesystem>
#include <optional>
#include <span>

#include "cista/containers/pair.h"

#include "geo/box.h"
#include "geo/latlng.h"

#include "nigiri/types.h"

namespace nigiri {

struct shapes_storage {
  explicit shapes_storage(
      std::filesystem::path const&,
      cista::mmap::protection = cista::mmap::protection::WRITE);
  std::span<geo::latlng const> get_shape(shape_idx_t) const;
  std::span<geo::latlng const> get_shape(trip_idx_t) const;
  std::span<geo::latlng const> get_shape(trip_idx_t,
                                         interval<stop_idx_t> const&) const;
  shape_offset_idx_t add_offsets(std::vector<shape_offset_t> const&);
  void add_trip_shape_offsets(
      trip_idx_t, cista::pair<shape_idx_t, shape_offset_idx_t> const&);
  geo::box get_bounding_box(route_idx_t) const;
  std::optional<geo::box> get_bounding_box(route_idx_t, std::size_t) const;

  mm_vecvec<shape_idx_t, geo::latlng> data_;
  mm_vecvec<shape_offset_idx_t, shape_offset_t> offsets_;
  mm_vec_map<trip_idx_t, cista::pair<shape_idx_t, shape_offset_idx_t>>
      trip_offset_indices_;
  mm_vecvec<route_idx_t, geo::box> boxes_;
};

}  // namespace nigiri