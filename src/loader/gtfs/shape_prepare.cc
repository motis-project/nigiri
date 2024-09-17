#include "nigiri/loader/gtfs/shape_prepare.h"

#include "geo/latlng.h"
#include "geo/polyline.h"

#include "utl/progress_tracker.h"

#include "nigiri/shape.h"
#include "nigiri/stop.h"
#include "nigiri/types.h"

namespace nigiri::loader::gtfs {

auto get_closest(geo::latlng coordinate,
                 std::span<geo::latlng const> const& shape) {
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

std::vector<location_idx_t> get_interior_locations(stop_seq_t const& stops) {
  auto const length = stops.length();
  assert(length >= 2U);
  auto locations = std::vector<location_idx_t>(length - 2);
  for (auto i = 0U; i < locations.size(); ++i) {
    locations[i] = stop(stops[i + 1]).location_idx();
  }
  return locations;
}

trip_idx_t find_in_cache(
    hash_map<
        shape_idx_t,
        std::vector<std::pair<std::vector<location_idx_t>, trip_idx_t>>> const&
        offsets_cache,
    shape_idx_t shape_index,
    std::vector<location_idx_t> const& locations) {
  auto const cached_locations = offsets_cache.find(shape_index);
  if (cached_locations != offsets_cache.end()) {
    for (auto const& [known_locations, trip_index] : cached_locations->second) {
      if (known_locations == locations) {
        return trip_index;
      }
    }
  }
  return trip_idx_t::invalid();
}

std::vector<shape_offset_t> split_shape(
    timetable const& tt,
    std::span<geo::latlng const> shape,
    std::vector<location_idx_t> const& locations) {
  if (shape.empty()) {
    return {};
  }
  auto offsets = std::vector<shape_offset_t>(locations.size() + 2U);
  offsets.front() = shape_offset_t{0};
  offsets.back() = shape_offset_t{shape.size() - 1U};
  auto offset = shape_offset_t{0};
  auto index = 0U;
  for (auto const location_index : locations) {
    auto const location = tt.locations_.get(location_index);
    offset += get_closest(location.pos_,
                          shape.subspan(static_cast<std::size_t>(offset.v_)));
    offsets[++index] = offset;
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
  auto offsets_cache = hash_map<
      shape_idx_t,
      std::vector<std::pair<std::vector<location_idx_t>, trip_idx_t>>>{};
  for (auto const& trip : trips) {
    progress_tracker->update_fn();
    auto const trip_index = trip.trip_idx_;
    if (trip.stop_seq_.size() < 2U) {
      shapes_data.add_offsets(trip_index, {});
      continue;
    }
    auto const shape_index = trip.shape_idx_;
    auto const locations = get_interior_locations(trip.stop_seq_);
    if (auto cached_trip_index =
            find_in_cache(offsets_cache, shape_index, locations);
        cached_trip_index != trip_idx_t::invalid()) {
      shapes_data.duplicate_offsets(cached_trip_index, trip_index);
    } else {
      auto const shape = shapes_data.get_shape(shape_index);
      auto const offsets = split_shape(tt, shape, locations);
      shapes_data.add_offsets(trip_index, offsets);
      if (auto const it = offsets_cache.find(shape_index);
          it == offsets_cache.end()) {
        offsets_cache.emplace_hint(
            it, shape_index,
            std::vector{std::make_pair(locations, trip_index)});
      } else {
        it->second.emplace_back(std::make_pair(locations, trip_index));
      }
    }
  }
}

}  // namespace nigiri::loader::gtfs