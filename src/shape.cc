#include "nigiri/shape.h"

#include <cstdint>

#include "fmt/core.h"

#include "utl/verify.h"

#include "nigiri/types.h"

namespace nigiri {

template <typename Value, typename Size = std::uint64_t>
cista::basic_mmap_vec<Value, Size> create_storage_vector(
    std::string_view const path, cista::mmap::protection const mode) {
  return cista::basic_mmap_vec<Value, Size>{cista::mmap{path.data(), mode}};
}

template <typename Key, typename Value>
mm_vecvec<Key, Value> create_storage(std::filesystem::path const& path,
                                     std::string_view prefix,
                                     cista::mmap::protection const mode) {
  return {
      create_storage_vector<Value>(
          fmt::format("{}_{}_data.bin", path.generic_string(), prefix), mode),
      create_storage_vector<cista::base_t<Key>>(
          fmt::format("{}_{}_idx.bin", path.generic_string(), prefix), mode)};
}

std::pair<std::span<geo::latlng const>, shape_offset_idx_t> get_shape(
    shapes_storage const& storage, trip_idx_t const trip_idx) {
  if (trip_idx == trip_idx_t::invalid() ||
      trip_idx >= storage.trip_offset_indices_.size()) {
    return {};
  }
  auto const [shape_idx, offset_idx] = storage.trip_offset_indices_[trip_idx];
  assert((shape_idx == shape_idx_t::invalid()) ==
         (offset_idx == shape_offset_idx_t::invalid()));
  if (offset_idx == shape_offset_idx_t::invalid()) {
    return {};
  }
  return std::pair{storage.get_shape(shape_idx), offset_idx};
}

shapes_storage::shapes_storage(std::filesystem::path const& path,
                               cista::mmap::protection const mode)
    : data_{create_storage<shape_idx_t, geo::latlng>(path, "points", mode)},
      offsets_{create_storage<shape_offset_idx_t, shape_offset_t>(
          path, "offsets", mode)},
      trip_offset_indices_{
          create_storage_vector<cista::pair<shape_idx_t, shape_offset_idx_t>,
                                trip_idx_t>(
              fmt::format("{}_offset_indices.bin", path.generic_string()),
              mode)},
      boxes_{create_storage<route_idx_t, geo::box>(path, "boxes", mode)} {}

std::span<geo::latlng const> shapes_storage::get_shape(
    shape_idx_t const shape_idx) const {
  if (shape_idx == shape_idx_t::invalid() || shape_idx > data_.size()) {
    return {};
  }
  auto const shape = data_[shape_idx];
  return {begin(shape), end(shape)};
}

std::span<geo::latlng const> shapes_storage::get_shape(
    trip_idx_t const trip_idx) const {
  auto const [shape, _] = nigiri::get_shape(*this, trip_idx);
  return shape;
}

std::span<geo::latlng const> shapes_storage::get_shape(
    trip_idx_t const trip_idx, interval<stop_idx_t> const& range) const {
  auto const [shape, offset_idx] = nigiri::get_shape(*this, trip_idx);
  if (shape.empty()) {
    return shape;
  }
  auto const offsets = offsets_[offset_idx];
  auto const from = static_cast<unsigned>(offsets[range.from_]);
  auto const to = static_cast<unsigned>(offsets[range.to_ - 1]);
  return shape.subspan(from, to - from + 1);
}

shape_offset_idx_t shapes_storage::add_offsets(
    std::vector<shape_offset_t> const& offsets) {
  auto const index = shape_offset_idx_t{offsets_.size()};
  offsets_.emplace_back(offsets);
  return index;
}

void shapes_storage::add_trip_shape_offsets(
    [[maybe_unused]] trip_idx_t const trip_idx,
    cista::pair<shape_idx_t, shape_offset_idx_t> const& offset_idx) {
  assert(trip_idx == trip_offset_indices_.size());
  trip_offset_indices_.emplace_back(offset_idx);
}

geo::box shapes_storage::get_bounding_box(route_idx_t const route_idx) const {
  utl::verify(route_idx < boxes_.size(), "Route index {} is out of bounds",
              route_idx);
  // 0: bounding box for trip
  return boxes_[route_idx][0];
}

geo::box shapes_storage::get_bounding_box_or_else(
    nigiri::route_idx_t const route_idx,
    std::size_t const segment,
    std::function<geo::box()> const& callback) const {

  utl::verify(route_idx < boxes_.size(), "Route index {} is out of bounds",
              route_idx);
  auto const& boxes = boxes_[route_idx];
  // 1-N: bounding box for segment
  return segment + 1 < boxes.size() ? boxes[segment + 1] : callback();
}

}  // namespace nigiri