#include "nigiri/shape.h"

#include <cstdint>

#include "fmt/core.h"

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
    shapes_storage const& storage, trip_idx_t const trip_index) {
  if (trip_index == trip_idx_t::invalid() ||
      trip_index >= storage.trip_offset_indices_.size()) {
    return {};
  }
  auto const [shape_index, offset_index] =
      storage.trip_offset_indices_[trip_index];
  // Reminder: shape_index is checked by 'storage.get_shape(shape_index)'
  if (offset_index == shape_offset_idx_t::invalid()) {
    return {};
  }
  return std::pair{storage.get_shape(shape_index), offset_index};
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
              mode)} {}

std::span<geo::latlng const> shapes_storage::get_shape(
    shape_idx_t const shape_index) const {
  if (shape_index == shape_idx_t::invalid() || shape_index > data_.size()) {
    return {};
  }
  auto const shape = data_[shape_index];
  return {begin(shape), end(shape)};
}

std::span<geo::latlng const> shapes_storage::get_shape(
    trip_idx_t const trip_index) const {
  auto const [shape, _] = nigiri::get_shape(*this, trip_index);
  return shape;
}

std::span<geo::latlng const> shapes_storage::get_shape(
    trip_idx_t const trip_index, interval<stop_idx_t> const& range) const {
  auto const [shape, offset_index] = nigiri::get_shape(*this, trip_index);
  if (shape.empty()) {
    return shape;
  }
  auto const offsets = offsets_[offset_index];
  auto const from = static_cast<unsigned>(offsets[range.from_]);
  auto const to = static_cast<unsigned>(offsets[range.to_ - 1]);
  return shape.subspan(from, to - from + 1);
}

std::pair<std::span<geo::latlng const>, unsigned>
shapes_storage::get_shape_with_stop_count(trip_idx_t const trip_index,
                                          stop_idx_t const from) const {
  auto const [shape, offset_index] = nigiri::get_shape(*this, trip_index);
  if (shape.empty()) {
    return {};
  }
  auto const offsets = offsets_[offset_index];
  auto const offset = static_cast<unsigned>(offsets[from]);
  return std::pair{shape.subspan(offset), offsets.size() - from};
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