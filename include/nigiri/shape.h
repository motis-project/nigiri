#pragma once

#include <array>
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
  shapes_storage() = default;
  explicit shapes_storage(
      std::filesystem::path const&,
      cista::mmap::protection = cista::mmap::protection::WRITE);
  // TODO Use interval<stop_idx_t> instead
  std::span<geo::latlng const> get_shape(shape_idx_t) const;
  std::span<geo::latlng const> get_shape(timetable const&, trip_idx_t) const;
  std::span<geo::latlng const> get_shape(
      timetable const&, trip_idx_t, interval<geo::latlng const> const&) const;
  std::span<geo::latlng const> make_span(interval<geo::latlng const> const&) const;
  std::unique_ptr<shapes_storage_t> data_;
  mutable std::array<geo::latlng, 2> no_shape_cache_;
};

shape_idx_t get_shape_index(timetable const&, trip_idx_t);

}  // namespace nigiri