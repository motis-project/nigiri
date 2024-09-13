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

struct shapes_storage {
  using shape_offsets_storage_t = mm_vecvec<trip_idx_t, shape_offset_t>;
  shapes_storage() = default;
  explicit shapes_storage(
      std::filesystem::path const&,
      cista::mmap::protection = cista::mmap::protection::WRITE);
  // TODO Use interval<stop_idx_t> instead
  std::span<geo::latlng const> get_shape(shape_idx_t) const;
  std::span<geo::latlng const> get_shape(timetable const&, trip_idx_t) const;
  std::span<geo::latlng const> get_shape(
      timetable const&, trip_idx_t, interval<stop_idx_t> const&) const;
  // std::span<geo::latlng const> get_shape(
  //     timetable const&, trip_idx_t, interval<geo::latlng const> const&) const;
  // void add_offset(trip_idx_t, shape_offset_t);
  // void create_offsets(std::size_t);
  void add_offsets(trip_idx_t, std::vector<shape_offset_t> const&);
  std::unique_ptr<shapes_storage_t> data_;
  std::unique_ptr<shape_offsets_storage_t> offsets_;
};

shape_idx_t get_shape_index(timetable const&, trip_idx_t);

}  // namespace nigiri