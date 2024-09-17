#include "nigiri/shape.h"

#include "fmt/core.h"

#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri {

template <typename Key, typename Value>
mm_vecvec<Key, Value> create_storage(std::filesystem::path const& path,
                                     std::string_view prefix,
                                     cista::mmap::protection const mode) {
  return {
      cista::basic_mmap_vec<Value, std::uint64_t>{cista::mmap{
          fmt::format("{}_{}_data.bin", path.generic_string(), prefix).c_str(),
          mode}},
      cista::basic_mmap_vec<cista::base_t<Key>, std::uint64_t>{cista::mmap{
          fmt::format("{}_{}_idx.bin", path.generic_string(), prefix).c_str(),
          mode}}};
}

shapes_storage::shapes_storage(std::filesystem::path const& path,
                               cista::mmap::protection const mode)
    : data_{std::make_unique<shapes_storage_t>(
          create_storage<shape_idx_t, geo::latlng>(path, "points", mode))},
      offsets_{std::make_unique<shape_offsets_storage_t>(
          create_storage<trip_idx_t, shape_offset_t>(path, "offsets", mode))} {}

std::span<geo::latlng const> shapes_storage::get_shape(
    shape_idx_t const shape_index) const {
  if (!data_) {
    return {};
  }
  if (shape_index == shape_idx_t::invalid() || shape_index > data_->size()) {
    return {};
  }
  auto const shape = data_->at(shape_index);
  return {begin(shape), end(shape)};
}

std::span<geo::latlng const> shapes_storage::get_shape(
    timetable const& tt, trip_idx_t const trip_index) const {
  if (!data_) {
    return {};
  }
  return get_shape(get_shape_index(tt, trip_index));
}

std::span<geo::latlng const> shapes_storage::get_shape(
    timetable const& tt,
    trip_idx_t const trip_index,
    interval<stop_idx_t> const& range) const {
  auto const shape = get_shape(tt, trip_index);
  if (shape.empty()) {
    return shape;
  }
  auto const& offsets = offsets_->at(trip_index);
  auto const from_offset = static_cast<unsigned>(offsets.at(range.from_));
  auto const to_offset = static_cast<unsigned>(offsets.at(range.to_));
  return shape.subspan(from_offset, to_offset - from_offset + 1);
}

void shapes_storage::add_offsets(trip_idx_t,
                                 std::vector<shape_offset_t> const& offsets) {
  if (!offsets_) {
    return;
  }
  offsets_->emplace_back(offsets);
}

void shapes_storage::duplicate_offsets(trip_idx_t const from,
                                       [[maybe_unused]] trip_idx_t const to) {
  if (!offsets_) {
    return;
  }
  assert(from < to);
  assert(to == offsets_->size());

  auto const& duplicate = offsets_->at(from);
  offsets_->emplace_back(std::vector(begin(duplicate), end(duplicate)));
}

shape_idx_t get_shape_index(timetable const& tt, trip_idx_t const trip_index) {
  if (trip_index == trip_idx_t::invalid() ||
      trip_index >= tt.trip_shape_indices_.size()) {
    return shape_idx_t::invalid();
  }
  return tt.trip_shape_indices_.at(trip_index);
}

}  // namespace nigiri