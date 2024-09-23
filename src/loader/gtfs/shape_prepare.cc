#include "nigiri/loader/gtfs/shape_prepare.h"

#include <algorithm>

#include "geo/latlng.h"
#include "geo/polyline.h"

#include "utl/get_or_create.h"
#include "utl/progress_tracker.h"
#include "utl/zip.h"

#include "nigiri/stop.h"
#include "nigiri/types.h"

namespace nigiri::loader::gtfs {

auto get_closest(geo::latlng coordinate,
                 std::span<geo::latlng const> const shape) {
  if (shape.size() < 2) {
    return 0U;
  }
  auto const best = geo::distance_to_polyline(coordinate, shape);
  auto const segment = best.segment_idx_;
  auto const segment_from = shape[segment];
  auto const segment_to = shape[segment + 1];
  auto const offset = geo::distance(coordinate, segment_from) <=
                              geo::distance(coordinate, segment_to)
                          ? segment
                          : segment + 1;
  return static_cast<unsigned>(offset);
}

std::vector<shape_offset_t> split_shape(
    timetable const& tt,
    std::span<geo::latlng const> const shape,
    stop_seq_t const& stops) {
  if (shape.empty()) {
    return {};
  }
  auto offsets = std::vector<shape_offset_t>(stops.size());
  auto offset = shape_offset_t{0};

  auto index = 0U;
  for (auto const& location_index : stops) {
    if (index == 0U) {
      offsets[0] = shape_offset_t{0};
    } else if (index == stops.size() - 1U) {
      offsets[index] = shape_offset_t{shape.size() - 1U};
    } else {
      auto const location =
          tt.locations_.get(stop(location_index).location_idx());
      offsets[index] = offset += get_closest(
          location.pos_, shape.subspan(static_cast<std::size_t>(offset.v_)));
    }
    ++index;
  }

  return offsets;
}

void calculate_shape_offsets(timetable const& tt,
                             shapes_storage& shapes_data,
                             vector_map<gtfs_trip_idx_t, trip> const& trips) {
  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Calculating shape offsets")
      .out_bounds(98.F, 100.F)
      .in_high(trips.size());

  auto shape_offsets_cache =
      hash_map<std::pair<shape_idx_t, stop_seq_t const*>,
               cista::pair<shape_idx_t, shape_offset_idx_t>,
               decltype([](std::pair<shape_idx_t, stop_seq_t const*> const&
                               pair) noexcept {
                 auto h = cista::BASE_HASH;
                 h = cista::hash_combine(
                     h, cista::hashing<shape_idx_t>{}(pair.first));
                 h = cista::hash_combine(
                     h, cista::hashing<stop_seq_t>{}(*pair.second));
                 return h;
               }),
               decltype([](std::pair<shape_idx_t, stop_seq_t const*> const& lhs,
                           std::pair<shape_idx_t, stop_seq_t const*> const&
                               rhs) noexcept {
                 return (lhs.first == rhs.first) &&
                        (*lhs.second == *rhs.second);
               })>{};
  for (auto const& trip : trips) {
    progress_tracker->increment();
    auto const trip_index = trip.trip_idx_;
    auto const shape_index = trip.shape_idx_;
    if (shape_index == shape_idx_t::invalid() || trip.stop_seq_.size() < 2U) {
      shapes_data.add_trip_shape_offsets(
          trip_index,
          cista::pair{shape_idx_t::invalid(), shape_offset_idx_t::invalid()});
    } else {
      auto const shape_offset_indices = utl::get_or_create(
          shape_offsets_cache, std::make_pair(trip.shape_idx_, &trip.stop_seq_),
          [&]() {
            auto const shape = shapes_data.get_shape(shape_index);
            auto const offsets = split_shape(tt, shape, trip.stop_seq_);
            return cista::pair{shape_index, shapes_data.add_offsets(offsets)};
          });
      shapes_data.add_trip_shape_offsets(trip_index, shape_offset_indices);
    }
  }
}

}  // namespace nigiri::loader::gtfs