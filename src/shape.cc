#include "nigiri/shape.h"

#include "fmt/core.h"

#include "nigiri/timetable.h"

namespace nigiri {

shapes_storage_t create_shapes_storage(std::filesystem::path const& path) {
  constexpr auto const kMode = cista::mmap::protection::WRITE;
  return {
      cista::basic_mmap_vec<geo::latlng, std::uint64_t>{cista::mmap{
          fmt::format("{}_data.bin", path.generic_string()).c_str(), kMode}},
      cista::basic_mmap_vec<cista::base_t<shape_idx_t>, std::uint64_t>{
          cista::mmap{fmt::format("{}_idx.bin", path.generic_string()).c_str(),
                      kMode}}};
}

std::span<geo::latlng const> get_shape(timetable const& tt,
                                       shapes_storage_t const& shapes,
                                       trip_idx_t const trip_idx) {
  if (trip_idx == trip_idx_t::invalid() ||
      trip_idx >= tt.trip_shape_indices_.size()) {
    return {};
  }
  return get_shape(shapes, tt.trip_shape_indices_[trip_idx]);
}

std::span<geo::latlng const> get_shape(shapes_storage_t const& shapes,
                                       shape_idx_t const shape_idx) {
  if (shape_idx == shape_idx_t::invalid() || shape_idx >= shapes.size()) {
    return {};
  }
  auto const bucket = shapes[shape_idx];
  return {begin(bucket), end(bucket)};
}

}  // namespace nigiri