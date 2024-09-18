#pragma once

#include <filesystem>
#include <span>

#include "cista/containers/mmap_vec.h"

#include "geo/latlng.h"

#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri {

struct shapes_storage {
  explicit shapes_storage(
      std::filesystem::path const&,
      cista::mmap::protection = cista::mmap::protection::WRITE);
  std::span<geo::latlng const> get_shape(shape_idx_t) const;
  std::span<geo::latlng const> get_shape(timetable const&, trip_idx_t) const;
  std::span<geo::latlng const> get_shape(timetable const&,
                                         trip_idx_t,
                                         interval<stop_idx_t> const&) const;
  shape_offset_idx_t add_offsets(std::vector<shape_offset_t> const&);
  void register_trip(trip_idx_t, shape_offset_idx_t);
  mm_vecvec<shape_idx_t, geo::latlng> data_;
  mm_vecvec<shape_offset_idx_t, shape_offset_t> offsets_;
  cista::basic_mmap_vec<shape_offset_idx_t, trip_idx_t> trip_offset_indices_;
};

shape_idx_t get_shape_index(timetable const&, trip_idx_t);

}  // namespace nigiri