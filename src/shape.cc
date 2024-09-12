#include "nigiri/shape.h"

#include "fmt/core.h"

#include "geo/polyline.h"

#include "nigiri/timetable.h"

namespace nigiri {

template <typename Key, typename Value>
mm_vecvec<Key, Value> create_storage(std::filesystem::path const& path,
                                       cista::mmap::protection const mode) {
  return {
      cista::basic_mmap_vec<Value, std::uint64_t>{cista::mmap{
          fmt::format("{}_data.bin", path.generic_string()).c_str(), mode}},
      cista::basic_mmap_vec<cista::base_t<Key>, std::uint64_t>{
          cista::mmap{fmt::format("{}_idx.bin", path.generic_string()).c_str(),
                      mode}}};
}

template <std::ranges::range Range>
std::span<geo::latlng const> get_subshape(Range const shape, geo::latlng const& from, geo::latlng const& to) {
  auto const best_from = geo::distance_to_polyline(from, shape);
  auto const subshape_from = begin(shape) + static_cast<decltype(shape)::difference_type>(best_from.segment_idx_);
  auto const remaining_shape = std::span{subshape_from, end(shape)};
  auto const best_to = geo::distance_to_polyline(to, remaining_shape);
  return {subshape_from, best_to.segment_idx_ + 1};
}

shapes_storage::shapes_storage(
        std::filesystem::path const& path,
        cista::mmap::protection const mode) : data_{std::make_unique<shapes_storage_t>(create_storage<shape_idx_t, geo::latlng>(path, mode))} {}

std::span<geo::latlng const> shapes_storage::get_shape(timetable const& tt, trip_idx_t const trip_index) const {
  if (data_->empty()) {
    return {};
  }
  auto const shape_index = get_shape_index(tt, trip_index);
  if (shape_index == shape_idx_t::invalid() || shape_index > data_->size()) {
    return {};
  }
  auto const shape = data_->at(shape_index);
  return {begin(shape), end(shape)};
}

std::span<geo::latlng const> shapes_storage::get_shape(timetable const& tt, trip_idx_t const trip_index, interval<geo::latlng const> const& range) const {
  auto const shape = get_shape(tt, trip_index);
  if (shape.empty()) {
    return shape;
  }
  return get_subshape(shape, range.from_, range.to_);
}

std::span<geo::latlng const> shapes_storage::make_span(interval<geo::latlng const> const& range) {
  no_shape_cache_ = {range.from_, range.to_};
  return {no_shape_cache_};
}

shape_idx_t get_shape_index(timetable const& tt, trip_idx_t const trip_index) {
  if (trip_index == trip_idx_t::invalid() || trip_index >= tt.trip_shape_indices_.size()) {
    return shape_idx_t::invalid();
  }
  return tt.trip_shape_indices_.at(trip_index);
}

// ---- TODO Remove ----

shapes_storage_t create_shapes_storage(std::filesystem::path const& path,
                                       cista::mmap::protection const mode) {
  return create_storage<shape_idx_t, geo::latlng>(path, mode);
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