#pragma once

#include <filesystem>
#include <span>

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
  void add_offsets(trip_idx_t, std::vector<shape_offset_t> const&);
  void duplicate_offsets(trip_idx_t from, trip_idx_t to);
  mm_vecvec<shape_idx_t, geo::latlng> data_;
  mm_vecvec<trip_idx_t, shape_offset_t> offsets_;
};

shape_idx_t get_shape_index(timetable const&, trip_idx_t);

}  // namespace nigiri