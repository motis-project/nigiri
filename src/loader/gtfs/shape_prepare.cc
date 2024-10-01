#include "nigiri/loader/gtfs/shape_prepare.h"

#include <algorithm>
#include <iterator>

#include "geo/latlng.h"
#include "geo/polyline.h"

#include "utl/enumerate.h"
#include "utl/get_or_create.h"
#include "utl/helpers/algorithm.h"
#include "utl/pairwise.h"
#include "utl/pipes/all.h"
#include "utl/pipes/remove_if.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"
#include "utl/progress_tracker.h"

#include "nigiri/stop.h"
#include "nigiri/types.h"

namespace nigiri::loader::gtfs {

std::size_t get_closest(geo::latlng const& pos,
                        std::span<geo::latlng const> shape) {
  if (shape.size() < 2U) {
    return 0U;
  }
  auto const best = geo::distance_to_polyline(pos, shape);
  auto const from = shape[best.segment_idx_];
  auto const to = shape[best.segment_idx_ + 1];
  return geo::distance(pos, from) <= geo::distance(pos, to)
             ? best.segment_idx_
             : best.segment_idx_ + 1;
}

std::vector<shape_offset_t> split_shape(timetable const& tt,
                                        std::span<geo::latlng const> shape,
                                        stop_seq_t const& stop_seq) {
  if (shape.empty()) {
    return {};
  }

  auto offsets = std::vector<shape_offset_t>(stop_seq.size());
  auto remaining_start = cista::base_t<shape_offset_t>{0U};

  for (auto const [i, s] : utl::enumerate(stop_seq)) {
    if (i == 0U) {
      offsets[0] = shape_offset_t{0U};
    } else if (i == stop_seq.size() - 1U) {
      offsets[i] = shape_offset_t{shape.size() - 1U};
    } else {
      auto const pos = tt.locations_.coordinates_[stop{s}.location_idx()];
      remaining_start += get_closest(pos, shape.subspan(remaining_start));
      offsets[i] = shape_offset_t{remaining_start};
    }
  }

  return offsets;
}

std::vector<shape_offset_t> split_shape_by_dist_traveled(
    std::vector<double> const& dist_traveled_stops_times,
    std::vector<double> const& dist_traveled_shape) {
  auto offsets = std::vector<shape_offset_t>{};
  offsets.reserve(dist_traveled_stops_times.size());
  auto remaining_shape_begin = begin(dist_traveled_shape);
  for (auto const& distance : dist_traveled_stops_times) {
    remaining_shape_begin = std::lower_bound(
        remaining_shape_begin, end(dist_traveled_shape), distance);
    offsets.push_back(
        shape_offset_t{remaining_shape_begin - begin(dist_traveled_shape)});
  }
  return offsets;
}

bool is_monotonic_distances(std::vector<double> const& distances) {
  return utl::all_of(utl::pairwise(distances), [](auto&& pair) {
    auto const [previous, next] = pair;
    return previous < next;
  });
}

void calculate_shape_offsets(timetable const& tt,
                             shapes_storage& shapes_data,
                             vector_map<gtfs_trip_idx_t, trip> const& trips,
                             shape_id_map_t const& shape_states) {
  auto const shapes_distances =
      utl::all(shape_states)  //
      | utl::remove_if([](auto&& x) {
          auto const& [_, state] = x;
          return !is_monotonic_distances(state.distances_);
        })  //
      | utl::transform([](auto&& x) {
          auto const& [_, state] = x;
          return std::pair{state.index_, &state.distances_};
        })  //
      | utl::to<hash_map<shape_idx_t, std::vector<double> const*>>();

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

    auto const shape_offset_index = utl::get_or_create(
        shape_offsets_cache, std::pair{trip.shape_idx_, &trip.stop_seq_},
        [&]() {
          if (shape_index == shape_idx_t::invalid() ||
              trip.stop_seq_.size() < 2U) {
            return shape_offset_idx_t::invalid();
          }
          auto const shape_distances = shapes_distances.find(shape_index);
          if (shape_distances != end(shapes_distances) &&
              is_monotonic_distances(trip.distance_traveled_)) {
            auto const offsets = split_shape_by_dist_traveled(
                trip.distance_traveled_, *shape_distances->second);
            return shapes_data.add_offsets(offsets);
          }
          auto const shape = shapes_data.get_shape(shape_index);
          auto const offsets = split_shape(tt, shape, trip.stop_seq_);
          return shapes_data.add_offsets(offsets);
        });
    shapes_data.add_trip_shape_offsets(
        trip_index, cista::pair{shape_index, shape_offset_index});
  }
}

}  // namespace nigiri::loader::gtfs