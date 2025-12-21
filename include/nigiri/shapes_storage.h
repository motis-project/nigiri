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
  shapes_storage(std::filesystem::path, cista::mmap::protection);

  cista::mmap mm(char const* file);

  void add(shapes_storage* other);
  std::span<geo::latlng const> get_shape(shape_idx_t) const;
  std::span<geo::latlng const> get_shape(trip_idx_t) const;
  std::span<geo::latlng const> get_shape(trip_idx_t,
                                         interval<stop_idx_t> const&) const;
  shape_offset_idx_t add_offsets(std::vector<shape_offset_t> const&);
  void add_trip_shape_offsets(
      trip_idx_t, cista::pair<shape_idx_t, shape_offset_idx_t> const&);
  geo::box get_bounding_box(route_idx_t) const;
  std::optional<geo::box> get_bounding_box(route_idx_t,
                                           std::size_t segment) const;

  cista::mmap::protection mode_;
  std::filesystem::path p_;

  mm_paged_vecvec<shape_idx_t, geo::latlng> data_;
  mm_vecvec<shape_offset_idx_t, shape_offset_t, std::uint64_t> offsets_;
  mm_vec_map<trip_idx_t, cista::pair<shape_idx_t, shape_offset_idx_t>>
      trip_offset_indices_;
  mm_vec_map<route_idx_t, geo::box> route_bboxes_;
  mm_vecvec<route_idx_t, geo::box, std::uint64_t> route_segment_bboxes_;
};

}  // namespace nigiri