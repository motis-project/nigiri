#include "nigiri/shape.h"

#include <cstdint>
#include <format>

#include "fmt/core.h"

#include "nigiri/types.h"

namespace nigiri {

template <typename Value, typename Size = uint64_t>
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

shapes_storage::shapes_storage(std::filesystem::path const& path,
                               cista::mmap::protection const mode)
    : data_{create_storage<shape_idx_t, geo::latlng>(path, "points", mode)},
      offsets_{create_storage<shape_offset_idx_t, shape_offset_t>(
          path, "offsets", mode)},
      trip_offset_indices_{
          create_storage_vector<cista::pair<shape_idx_t, shape_offset_idx_t>,
                                trip_idx_t>(
              std::format("{}_offset_indices.bin", path.generic_string()),
              mode)} {}

std::span<geo::latlng const> shapes_storage::get_shape(
    shape_idx_t const shape_index) const {
  if (shape_index == shape_idx_t::invalid() || shape_index > data_.size()) {
    return {};
  }
  auto const shape = data_.at(shape_index);
  return {begin(shape), end(shape)};
}

std::span<geo::latlng const> shapes_storage::get_shape(
    trip_idx_t const trip_index) const {
  if (trip_index == trip_idx_t::invalid() ||
      trip_index >= trip_offset_indices_.size()) {
    return {};
  }
  auto const& [shape_index, _] = trip_offset_indices_[trip_index];
  return get_shape(shape_index);
}

std::span<geo::latlng const> shapes_storage::get_shape(
    trip_idx_t const trip_index, interval<stop_idx_t> const& range) const {
  if (trip_index == trip_idx_t::invalid() ||
      trip_index >= trip_offset_indices_.size()) {
    return {};
  }
  auto const& [shape_index, offset_index] = trip_offset_indices_[trip_index];
  // Reminder: shape_index is checked by 'get_shape(shape_index)'
  if (offset_index == shape_offset_idx_t::invalid()) {
    return {};
  }
  auto const shape = get_shape(shape_index);
  if (shape.empty()) {
    return shape;
  }
  auto const& offsets = offsets_.at(offset_index);
  auto const from_offset = static_cast<unsigned>(offsets.at(range.from_));
  auto const to_offset = static_cast<unsigned>(offsets.at(range.to_));
  return shape.subspan(from_offset, to_offset - from_offset + 1);
}

shape_offset_idx_t shapes_storage::add_offsets(
    std::vector<shape_offset_t> const& offsets) {
  auto const index = shape_offset_idx_t{offsets_.size()};
  offsets_.emplace_back(offsets);
  return index;
}

void shapes_storage::add_trip_shape_offsets(
    [[maybe_unused]] trip_idx_t const trip_index,
    cista::pair<shape_idx_t, shape_offset_idx_t> const& offset_index) {
  assert(trip_index == trip_offset_indices_.size());
  trip_offset_indices_.emplace_back(offset_index);
}

}  // namespace nigiri