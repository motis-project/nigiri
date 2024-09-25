#include "nigiri/loader/gtfs/shape_prepare.h"

#include <algorithm>
#include <iterator>

#include "geo/latlng.h"
#include "geo/polyline.h"

#include "utl/enumerate.h"
#include "utl/get_or_create.h"
#include "utl/helpers/algorithm.h"
#include "utl/pairwise.h"
#include "utl/progress_tracker.h"

#include "nigiri/stop.h"
#include "nigiri/types.h"

namespace nigiri::loader::gtfs {

auto get_closest(geo::latlng const& coordinate,
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

  for (auto const [index, location_index] : utl::enumerate(stops)) {
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
  }

  return offsets;
}

std::vector<shape_offset_t> split_shape_by_offsets(
    std::vector<double> const& stops, std::vector<double> const& shape) {
  auto offsets = std::vector<shape_offset_t>{};
  offsets.reserve(stops.size());
  auto remaining_shape_begin = begin(shape);
  auto offset = decltype(offsets)::iterator::difference_type{};
  for (auto const& stop : stops) {
    auto const candidate =
        std::lower_bound(remaining_shape_begin, end(shape), stop);
    if (candidate == end(shape)) {
      offset = static_cast<decltype(offset)>(shape.size() - 1);
    } else if (candidate == remaining_shape_begin) {
      offset = std::distance(begin(shape), candidate);
    } else if (stop - *(candidate - 1) < *candidate - stop) {
      offset = std::distance(begin(shape), candidate - 1);
    } else {
      offset = std::distance(begin(shape), candidate);
    }
    offsets.push_back(shape_offset_t{offset});
  }
  return offsets;
}

bool is_monotonic_distances(std::ranges::range auto const& distances) {
  if (distances.empty()) {
    return false;
  }
  auto const first = *begin(distances);
  if (first != 0.0) {
    return false;
  }
  auto const pairs = utl::pairwise(distances);
  return utl::all_of(pairs, [](auto const pair) {
    auto const [previous, next] = pair;
    return previous < next;
  });
};

auto get_shape_distance_map(shape_id_map_t const& shape_states) {
  auto shape_distances =
      hash_map<shape_idx_t,
               decltype(shape_id_map_t::value_type::second_type::distances_)
                   const*>{};
  for (auto const& [_, state] : shape_states) {
    if (is_monotonic_distances(state.distances_)) {
      shape_distances[state.index_] = &state.distances_;
    }
  }
  return shape_distances;
}

void calculate_shape_offsets(timetable const& tt,
                             shapes_storage& shapes_data,
                             vector_map<gtfs_trip_idx_t, trip> const& trips,
                             shape_id_map_t const& shape_states) {
  auto const shapes_distances = get_shape_distance_map(shape_states);

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Calculating shape offsets")
      .out_bounds(98.F, 100.F)
      .in_high(trips.size());

  auto const key_hash =
      [](std::pair<shape_idx_t, stop_seq_t const*> const& pair) noexcept {
        auto h = cista::BASE_HASH;
        h = cista::hash_combine(h, cista::hashing<shape_idx_t>{}(pair.first));
        h = cista::hash_combine(h, cista::hashing<stop_seq_t>{}(*pair.second));
        return h;
      };
  auto const key_compare =
      [](std::pair<shape_idx_t, stop_seq_t const*> const& lhs,
         std::pair<shape_idx_t, stop_seq_t const*> const& rhs) noexcept {
        return (lhs.first == rhs.first) && (*lhs.second == *rhs.second);
      };
  auto shape_offsets_cache =
      hash_map<std::pair<shape_idx_t, stop_seq_t const*>, shape_offset_idx_t,
               decltype(key_hash), decltype(key_compare)>{};
  for (auto const& trip : trips) {
    progress_tracker->increment();
    auto const trip_index = trip.trip_idx_;
    auto const shape_index = trip.shape_idx_;
    auto const shape_distances = shapes_distances.find(shape_index);
    if (shape_distances != end(shapes_distances) &&
        is_monotonic_distances(trip.distance_traveled_)) {
      auto const offsets = split_shape_by_offsets(trip.distance_traveled_,
                                                  *shape_distances->second);
      auto const shape_offset_index = shapes_data.add_offsets(offsets);
      shapes_data.add_trip_shape_offsets(
          trip_index, cista::pair{shape_index, shape_offset_index});
    } else if (shape_index == shape_idx_t::invalid() ||
               trip.stop_seq_.size() < 2U) {
      shapes_data.add_trip_shape_offsets(
          trip_index,
          cista::pair{shape_idx_t::invalid(), shape_offset_idx_t::invalid()});
    } else {
      auto const shape_offset_index = utl::get_or_create(
          shape_offsets_cache, std::make_pair(trip.shape_idx_, &trip.stop_seq_),
          [&]() {
            auto const shape = shapes_data.get_shape(shape_index);
            auto const offsets = split_shape(tt, shape, trip.stop_seq_);
            return shapes_data.add_offsets(offsets);
          });
      shapes_data.add_trip_shape_offsets(
          trip_index, cista::pair{shape_index, shape_offset_index});
    }
  }
}

}  // namespace nigiri::loader::gtfs