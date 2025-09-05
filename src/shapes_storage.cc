#include "nigiri/shapes_storage.h"

#include <cstdint>

#include "cista/strong.h"

#include "fmt/core.h"

#include "utl/verify.h"

#include "nigiri/types.h"

namespace fs = std::filesystem;

namespace nigiri {

shapes_storage::shapes_storage(std::filesystem::path path,
                               cista::mmap::protection const mode)
    : mode_{mode},
      p_{[&]() {
        fs::create_directories(path);
        return std::move(path);
      }()},
      data_{mm_paged_vecvec_helper<shape_idx_t, geo::latlng>::data_t{
                mm_vec<geo::latlng>{mm("shapes_data.bin")}},
            mm_vec<cista::page<std::uint64_t, std::uint32_t>>{
                mm("shapes_idx.bin")}},
      offsets_{mm_vec<shape_offset_t>{mm("shape_offsets_data.bin")},
               mm_vec<std::uint64_t>{mm("shape_offsets_idx.bin")}},
      trip_offset_indices_{mm("shape_trip_offsets.bin")},
      route_bboxes_{mm("shape_route_bboxes.bin")},
      route_segment_bboxes_{
          mm_vec<geo::box>{mm("shape_route_segment_bboxes_data.bin")},
          mm_vec<std::uint64_t>{mm("shape_route_segment_bboxes_idx.bin")}} {}

cista::mmap shapes_storage::mm(char const* file) {
  return cista::mmap{(p_ / file).generic_string().c_str(), mode_};
}

void shapes_storage::add(shapes_storage* other) {
  for (auto i = 0U; i < other->data_.size(); ++i) {
    auto const idx = shape_idx_t{i};
    this->data_.emplace_back(other->data_[idx]);
  }
  for (auto const& e : other->offsets_) {
    this->offsets_.emplace_back(e);
  }
  for (auto const& e : other->trip_offset_indices_) {
    this->trip_offset_indices_.emplace_back(e);
  }
  for (auto const& e : other->route_bboxes_) {
    this->route_bboxes_.emplace_back(e);
  }
  for (auto const& e : other->route_segment_bboxes_) {
    this->route_segment_bboxes_.emplace_back(e);
  }
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

std::span<geo::latlng const> shapes_storage::get_shape(
    shape_idx_t const shape_idx) const {
  if (shape_idx == shape_idx_t::invalid() ||
      static_cast<std::size_t>(to_idx(shape_idx)) > data_.size()) {
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
  utl::verify(cista::to_idx(route_idx) < route_bboxes_.size(),
              "Route index {} is out of bounds", route_idx);
  return route_bboxes_[route_idx];
}

std::optional<geo::box> shapes_storage::get_bounding_box(
    nigiri::route_idx_t const route_idx, std::size_t const segment) const {
  utl::verify(cista::to_idx(route_idx) < route_segment_bboxes_.size(),
              "Route index {} is out of bounds", route_idx);
  auto const& bboxes = route_segment_bboxes_[route_idx];
  return segment < bboxes.size() ? bboxes[segment]
                                 : std::optional<geo::box>{std::nullopt};
}

}  // namespace nigiri