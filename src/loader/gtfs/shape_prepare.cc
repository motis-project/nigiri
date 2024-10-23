#include "nigiri/loader/gtfs/shape_prepare.h"

#include <algorithm>
#include <ranges>
#include <span>
#include <type_traits>

#include "geo/latlng.h"
#include "geo/polyline.h"

#include "utl/enumerate.h"
#include "utl/get_or_create.h"
#include "utl/progress_tracker.h"

#include "nigiri/rt/frun.h"
#include "nigiri/shape.h"
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

std::vector<shape_offset_t> get_offsets_by_stops(
    timetable const& tt,
    std::span<geo::latlng const> shape,
    stop_seq_t const& stop_seq) {
  auto offsets = std::vector<shape_offset_t>(stop_seq.size());
  auto remaining_start = cista::base_t<shape_offset_t>{1U};
  // Reserve space to map each stop to a different point
  auto max_width = shape.size() - stop_seq.size();

  for (auto const [i, s] : utl::enumerate(stop_seq)) {
    if (i == 0U) {
      offsets[0] = shape_offset_t{0U};
    } else if (i == stop_seq.size() - 1U) {
      offsets[i] = shape_offset_t{shape.size() - 1U};
    } else {
      auto const pos = tt.locations_.coordinates_[stop{s}.location_idx()];
      auto const offset =
          get_closest(pos, shape.subspan(remaining_start, max_width + 1U));
      offsets[i] = shape_offset_t{remaining_start + offset};
      remaining_start += offset + 1U;
      max_width -= offset;
    }
  }

  return offsets;
}

template <typename DoubleRange>
  requires std::ranges::range<DoubleRange> &&
           std::is_same_v<std::ranges::range_value_t<DoubleRange>, double>
std::vector<shape_offset_t> get_offsets_by_dist_traveled(
    std::vector<double> const& dist_traveled_stops_times,
    DoubleRange const& dist_traveled_shape_edges) {
  auto offsets = std::vector<shape_offset_t>{};
  offsets.reserve(dist_traveled_stops_times.size());
  auto remaining_shape_begin = begin(dist_traveled_shape_edges);
  for (auto const& distance : dist_traveled_stops_times) {
    remaining_shape_begin = std::lower_bound(
        remaining_shape_begin, end(dist_traveled_shape_edges), distance);
    offsets.push_back(shape_offset_t{remaining_shape_begin -
                                     begin(dist_traveled_shape_edges)});
  }
  return offsets;
}

void calculate_shape_offsets(timetable const& tt,
                             shapes_storage& shapes_data,
                             vector_map<gtfs_trip_idx_t, trip> const& trips,
                             shape_loader_state const& shape_states) {
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
    auto const trip_idx = trip.trip_idx_;
    auto const shape_idx = trip.shape_idx_;

    auto const shape_offset_idx = utl::get_or_create(
        shape_offsets_cache, std::pair{shape_idx, &trip.stop_seq_}, [&]() {
          if (shape_idx == shape_idx_t::invalid() ||
              trip.stop_seq_.size() < 2U) {
            return shape_offset_idx_t::invalid();
          }
          auto const& shape_distances =
              shape_states.distances_[shape_idx - shape_states.index_offset_];
          if (!shape_distances.empty() && !trip.distance_traveled_.empty()) {
            auto const offsets = get_offsets_by_dist_traveled(
                trip.distance_traveled_, shape_distances);
            return shapes_data.add_offsets(offsets);
          }
          auto const shape = shapes_data.get_shape(shape_idx);
          if (shape.size() < trip.stop_seq_.size()) {
            return shape_offset_idx_t::invalid();  // >= 1 shape/point required
          }
          auto const offsets = get_offsets_by_stops(tt, shape, trip.stop_seq_);
          return shapes_data.add_offsets(offsets);
        });
    shapes_data.add_trip_shape_offsets(
        trip_idx, cista::pair{shape_offset_idx == shape_offset_idx_t::invalid()
                                  ? shape_idx_t::invalid()
                                  : shape_idx,
                              shape_offset_idx});
  }
}

void calculate_shape_boxes(timetable const& tt, shapes_storage& shapes_data) {
  auto shape_segment_boxes =
      hash_map<cista::pair<shape_idx_t, shape_offset_idx_t>,
               std::vector<geo::box>>{};
  // Create bounding boxes for all shape segments
  for (auto const key : shapes_data.trip_offset_indices_) {
    if (key.first == shape_idx_t::invalid() ||
        key.second == shape_offset_idx_t::invalid()) {
      continue;
    }
    utl::get_or_create(shape_segment_boxes, key, [&]() {
      auto const shape = shapes_data.get_shape(key.first);
      auto const& offsets = shapes_data.offsets_[key.second];
      auto segment_boxes = std::vector<geo::box>(offsets.size() - 1);
      for (auto const [i, pair] : utl::enumerate(utl::pairwise(offsets))) {
        auto& box = segment_boxes[i];
        auto const& [from, to] = pair;
        for (auto const point :
             shape.subspan(cista::to_idx(from),
                           cista::to_idx(to) - cista::to_idx(from) + 1)) {
          box.extend(point);
        }
      }
      return segment_boxes;
    });
  }
  // Create bounding boxes for all routes not already added
  for (auto const r : tt.transport_route_ |
                          std::views::filter([&](route_idx_t const route_idx) {
                            return route_idx >= shapes_data.boxes_.size();
                          })) {
    auto const seq = tt.route_location_seq_[r];
    assert(seq.size() > 0U);
    auto segment_boxes = std::vector<geo::box>(seq.size());
    auto last_extend = 0U;
    // 0: bounding box for trip,  1-N: bounding box for segment
    auto& bounding_box = segment_boxes[0U];
    auto const stop_indices =
        interval{stop_idx_t{0U}, static_cast<stop_idx_t>(seq.size())};
    for (auto const transport_idx : tt.route_transport_ranges_[r]) {
      auto const frun = rt::frun{tt, nullptr,
                                 rt::run{.t_ = transport{transport_idx},
                                         .stop_range_ = stop_indices,
                                         .rt_ = rt_transport_idx_t::invalid()}};
      frun.for_each_trip([&](trip_idx_t const trip_idx,
                             interval<stop_idx_t> const absolute_range) {
        auto shape_boxes = static_cast<std::vector<geo::box> const*>(nullptr);
        auto it = shape_segment_boxes.find(
            shapes_data.trip_offset_indices_[trip_idx]);
        if (it != end(shape_segment_boxes)) {
          shape_boxes = &it->second;
        }
        auto prev_pos = tt.locations_.coordinates_.at(
            stop{seq[absolute_range.from_]}.location_idx());
        bounding_box.extend(prev_pos);
        for (auto const [from, to] : utl::pairwise(absolute_range)) {
          auto const next_pos =
              tt.locations_.coordinates_.at(stop{seq[to]}.location_idx());
          auto& box = segment_boxes[cista::to_idx(to)];
          bounding_box.extend(next_pos);
          box.extend(prev_pos);
          box.extend(next_pos);
          if (shape_boxes != nullptr) {
            auto const& shape_box = (*shape_boxes)[static_cast<std::size_t>(
                cista::to_idx(from) - cista::to_idx(absolute_range.from_))];
            if (!box.contains(shape_box)) {
              bounding_box.extend(shape_box);
              box.extend(shape_box);
              last_extend = std::max(last_extend, from + 1U);
            }
          }
          prev_pos = next_pos;
        }
      });
    }
    // 0: bounding box for trip,  1-N: bounding box for segment
    segment_boxes.resize(last_extend + 1);
    shapes_data.boxes_.emplace_back(segment_boxes);
  }
}

}  // namespace nigiri::loader::gtfs