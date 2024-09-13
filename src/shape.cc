#include "nigiri/shape.h"

#include "fmt/core.h"

#include "geo/polyline.h"

#include "nigiri/timetable.h"

namespace nigiri {

template <typename Key, typename Value>
mm_vecvec<Key, Value> create_storage(std::filesystem::path const& path,
                                     std::string_view prefix,
                                     cista::mmap::protection const mode) {
  return {cista::basic_mmap_vec<Value, std::uint64_t>{cista::mmap{
              fmt::format("{}_{}_data.bin", path.generic_string(), prefix).c_str(), mode}},
          cista::basic_mmap_vec<cista::base_t<Key>, std::uint64_t>{cista::mmap{
              fmt::format("{}_{}_idx.bin", path.generic_string(), prefix).c_str(), mode}}};
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
  // std::cout << "REQ TRIP INDEX: " << trip_index << "\n";
  auto const& offsets = offsets_->at(trip_index);
  // std::cout << "AAAAA" << offsets.size() << " / " << range << "\n";
  auto const from_offset = static_cast<unsigned>(offsets.at(range.from_));
  // std::cout << "BBBBB\n";
  auto const to_offset = static_cast<unsigned>(offsets.at(range.to_));
  // std::cout << "BUCKET: ";
  // for (auto& x : offsets) std::cout << x << ", ";
  // std::cout << "\nSTORED: " << from_offset << ", " << to_offset << " / " << to_offset - from_offset + 1 << "\n";
  return {begin(shape) + from_offset, to_offset - from_offset + 1};
  // return get_subshape(shape, range.from_, range.to_);
}
// std::span<geo::latlng const> shapes_storage::get_shape(
//     timetable const& tt,
//     trip_idx_t const trip_index,
//     interval<geo::latlng const> const& range) const {
//   auto const shape = get_shape(tt, trip_index);
//   if (shape.empty()) {
//     return shape;
//   }
//   return get_subshape(shape, range.from_, range.to_);
// }

// void shapes_storage::create_offsets(std::size_t const size) {
//   if (!offsets_) {
//     return;
//   }
//   offsets_->resize(size);
// }

// void shapes_storage::add_offsets(trip_idx_t const trip_index, std::vector<shape_offset_t> const& offsets) {
void shapes_storage::add_offsets(trip_idx_t, std::vector<shape_offset_t> const& offsets) {
  if (!offsets_) {
    return;
  }
  offsets_->emplace_back(offsets);
  // // assert(trip_index == offset_->size());
  // auto bucket = offsets_->add_back_sized(offsets.size());
  // for (auto const& offset : offsets) {
  //   bucket.push_back(offset);
  // }
}
// void shapes_storage::add_offset(trip_idx_t const trip_index, shape_offset_t const offset) {
//   if (!offsets_) {
//     return;
//   }
//   (*offsets_)[trip_index].push_back(offset);
// }

shape_idx_t get_shape_index(timetable const& tt, trip_idx_t const trip_index) {
  if (trip_index == trip_idx_t::invalid() ||
      trip_index >= tt.trip_shape_indices_.size()) {
    return shape_idx_t::invalid();
  }
  return tt.trip_shape_indices_.at(trip_index);
}

}  // namespace nigiri