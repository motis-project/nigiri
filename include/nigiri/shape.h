#pragma once

#include <filesystem>
#include <memory>
#include <span>

#include "geo/latlng.h"

#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri {

// using shapes_storage_t = mm_vecvec<shape_idx_t, geo::latlng>;

struct shapes_storage {
  using shape_offsets_storage_t = mm_vecvec<trip_idx_t, shape_offset_t>;
  shapes_storage() = default;
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
  std::unique_ptr<shapes_storage_t> data_;
  std::unique_ptr<shape_offsets_storage_t> offsets_;
};

constexpr shape_idx_t get_shape_index(timetable const&, trip_idx_t);

}  // namespace nigiri