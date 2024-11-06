#include "nigiri/loader/gtfs/shape_prepare.h"

#include <algorithm>
#include <mutex>
#include <ranges>
#include <span>
#include <type_traits>

#include "cista/strong.h"

#include "geo/latlng.h"
#include "geo/polyline.h"

#include "utl/enumerate.h"
#include "utl/helpers/algorithm.h"
#include "utl/parallel_for.h"
#include "utl/progress_tracker.h"
#include "utl/verify.h"

#include "nigiri/loader/gtfs/shape.h"
#include "nigiri/rt/frun.h"
#include "nigiri/shapes_storage.h"
#include "nigiri/stop.h"
#include "nigiri/timetable.h"

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
  // Ensure final offset maps to valid shape point
  auto const shape_end =
      begin(dist_traveled_shape_edges) +
      (end(dist_traveled_shape_edges) - begin(dist_traveled_shape_edges)) - 1;
  for (auto const& distance : dist_traveled_stops_times) {
    remaining_shape_begin =
        std::lower_bound(remaining_shape_begin, shape_end, distance);
    offsets.push_back(shape_offset_t{remaining_shape_begin -
                                     begin(dist_traveled_shape_edges)});
  }
  return offsets;
}

shape_prepare::shape_results::result::result(
    stop_seq_t const* stop_seq, std::vector<double> const* distances)
    : stop_seq_{stop_seq},
      distances_{distances},
      offset_idx_{shape_offset_idx_t::invalid()},
      trip_box_{},
      segment_boxes_{} {}

shape_prepare::shape_prepare(shape_loader_state const& states,
                             vector_map<gtfs_trip_idx_t, trip> const& trips,
                             shapes_storage& shapes_data)
    : index_offset_{states.index_offset_},
      shape_results_(states.id_map_.size()),
      shapes_{shapes_data},
      results_ready_{false} {
  for (auto i = 0U; i < shape_results_.size(); ++i) {
    shape_results_[i].shape_idx_ = static_cast<shape_idx_t>(index_offset_ + i);
  }
  for (auto const& trip : trips) {
    if (trip.shape_idx_ == shape_idx_t::invalid()) {
      continue;
    }
    auto const idx = cista::to_idx(trip.shape_idx_ - index_offset_);
    auto& results = shape_results_[idx].results_;
    if (utl::all_of(results, [&](shape_results::result const& it) {
          return *it.stop_seq_ != trip.stop_seq_;
        })) {
      auto const distances =
          trip.distance_traveled_.empty() ? nullptr : &trip.distance_traveled_;
      results.emplace_back(&trip.stop_seq_, std::move(distances));
    }
  }
}

void shape_prepare::calculate_results(timetable const& tt,
                                      shape_loader_state const& states) {
  auto m = std::mutex{};
  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Calculating shape offsets")
      .out_bounds(98.F, 99.F)
      .in_high(shape_results_.size());
  utl::parallel_for(shape_results_, [&](shape_results& results) {
    progress_tracker->increment();
    if (results.results_.empty()) {
      return;
    }
    auto const shape = shapes_.get_shape(results.shape_idx_);
    auto const& shape_distances =
        states.distances_[cista::to_idx(results.shape_idx_ - index_offset_)];
    for (auto& result : results.results_) {

      auto const offsets = [&]() {
        if (!shape_distances.empty() && result.distances_ != nullptr) {
          return get_offsets_by_dist_traveled(*result.distances_,
                                              shape_distances);
        }
        if (shape.size() < result.stop_seq_->size()) {
          return std::vector<shape_offset_t>{};
        }

        return get_offsets_by_stops(tt, shape, *result.stop_seq_);
      }();

      if (!offsets.empty()) {
        auto const guard = std::lock_guard<decltype(m)>{m};
        result.offset_idx_ = shapes_.add_offsets(offsets);
      } else {
        result.offset_idx_ = shape_offset_idx_t::invalid();
      }

      if (!offsets.empty()) {
        result.segment_boxes_ = std::vector<geo::box>(offsets.size() - 1);
        auto box_count = 0UL;
        for (auto const [i, pair] : utl::enumerate(utl::pairwise(offsets))) {
          auto& segment_box = result.segment_boxes_[i];
          auto const& [from, to] = pair;
          for (auto const point :
               shape.subspan(cista::to_idx(from),
                             cista::to_idx(to) - cista::to_idx(from) + 1)) {
            result.trip_box_.extend(point);
            segment_box.extend(point);
          }
          auto const from_l =
              tt.locations_
                  .coordinates_[stop{(*result.stop_seq_)[i]}.location_idx()];
          auto const to_l =
              tt.locations_.coordinates_[stop{(*result.stop_seq_)[i + 1]}
                                             .location_idx()];
          auto const stop_box = geo::make_box({from_l, to_l});
          if (!stop_box.contains(segment_box)) {
            box_count = i + 1U;
          }
        }
        result.segment_boxes_.resize(box_count);
      } else {
        for (auto const s : *result.stop_seq_) {
          result.trip_box_.extend(
              tt.locations_.coordinates_[stop{s}.location_idx()]);
        }
      }
    }
  });
  results_ready_ = true;
}

void shape_prepare::create_trip_shape_offsets(
    vector_map<gtfs_trip_idx_t, trip> const& trips) const {
  utl::verify(results_ready_, "Operation requires calculated results");
  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Storing trip offsets")
      .out_bounds(99.F, 100.F)
      .in_high(trips.size());
  for (auto const& trip : trips) {
    progress_tracker->increment();
    auto const trip_idx = trip.trip_idx_;
    auto const shape_idx = trip.shape_idx_;
    auto const offset_idx = [&]() {
      if (shape_idx == shape_idx_t::invalid()) {
        return shape_offset_idx_t::invalid();
      }
      auto const& results =
          shape_results_[cista::to_idx(shape_idx - index_offset_)].results_;
      auto const result =
          utl::find_if(results, [&](shape_results::result const& res) {
            return *res.stop_seq_ == trip.stop_seq_;
          });
      assert(result != end(results));
      return result->offset_idx_;
    }();
    shapes_.add_trip_shape_offsets(trip_idx,
                                   cista::pair{shape_idx, offset_idx});
  }
}

void shape_prepare::create_route_boxes(timetable const& tt) const {
  utl::verify(results_ready_, "Operation requires calculated results");
  auto const route_offset = shapes_.route_boxes_.size();
  auto const routes_count = tt.route_transport_ranges_.size() - route_offset;
  auto route_boxes = std::vector<geo::box>(routes_count);
  auto route_segment_boxes = std::vector<std::vector<geo::box>>(routes_count);
  utl::parallel_for(std::views::iota(0U, routes_count), [&](std::size_t const
                                                                box_idx) {
    auto const r = static_cast<route_idx_t>(box_idx + route_offset);
    auto const seq = tt.route_location_seq_[r];
    assert(seq.size() > 0U);
    auto bounding_box = geo::box{};
    auto segment_boxes = std::vector<geo::box>(seq.size() - 1);
    auto box_count = 0UL;
    auto const stop_indices =
        interval{stop_idx_t{0U}, static_cast<stop_idx_t>(seq.size())};
    for (auto const transport_idx : tt.route_transport_ranges_[r]) {
      auto const frun = rt::frun{tt, nullptr,
                                 rt::run{.t_ = transport{transport_idx},
                                         .stop_range_ = stop_indices,
                                         .rt_ = rt_transport_idx_t::invalid()}};
      frun.for_each_trip([&](trip_idx_t const trip_idx,
                             interval<stop_idx_t> const absolute_range) {
        auto const [shape_idx, offset_idx] =
            shapes_.trip_offset_indices_[trip_idx];
        if (shape_idx == shape_idx_t::invalid() ||
            offset_idx == shape_offset_idx_t::invalid()) {
          for (auto const idx : absolute_range) {
            bounding_box.extend(
                tt.locations_.coordinates_[stop{seq[idx]}.location_idx()]);
          }
        } else {
          auto const& result = utl::find_if(
              shape_results_[cista::to_idx(shape_idx - index_offset_)].results_,
              [&](shape_results::result const& res) {
                return res.offset_idx_ == offset_idx;
              });
          bounding_box.extend(result->trip_box_);
          for (auto i = 0U; i < result->segment_boxes_.size(); ++i) {
            segment_boxes[i + cista::to_idx(absolute_range.from_)] =
                result->segment_boxes_[i];
          }
          box_count = std::max(box_count, static_cast<unsigned long>(
                                              result->segment_boxes_.size() +
                                              absolute_range.from_));
        }
      });
    }
    segment_boxes.resize(box_count);
    route_boxes[box_idx] = std::move(bounding_box);
    route_segment_boxes[box_idx] = std::move(segment_boxes);
  });
  for (auto const [route_box, segment_boxes] :
       utl::zip(route_boxes, route_segment_boxes)) {
    shapes_.route_boxes_.emplace_back(route_box);
    shapes_.segment_boxes_.emplace_back(segment_boxes);
  }
}
}  // namespace nigiri::loader::gtfs