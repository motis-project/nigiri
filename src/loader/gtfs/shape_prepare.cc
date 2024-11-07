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
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"
#include "utl/progress_tracker.h"
#include "utl/verify.h"

#include "nigiri/loader/gtfs/shape.h"
#include "nigiri/loader/gtfs/trip.h"
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
      trip_bbox_{},
      segment_bboxes_{} {}

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
        result.segment_bboxes_ = std::vector<geo::box>(offsets.size() - 1);
        auto bbox_count = 0UL;
        for (auto const [i, pair] : utl::enumerate(utl::pairwise(offsets))) {
          auto& segment_bbox = result.segment_bboxes_[i];
          auto const& [from, to] = pair;
          for (auto const point :
               shape.subspan(cista::to_idx(from),
                             cista::to_idx(to) - cista::to_idx(from) + 1)) {
            result.trip_bbox_.extend(point);
            segment_bbox.extend(point);
          }
          auto const from_l =
              tt.locations_
                  .coordinates_[stop{(*result.stop_seq_)[i]}.location_idx()];
          auto const to_l =
              tt.locations_.coordinates_[stop{(*result.stop_seq_)[i + 1]}
                                             .location_idx()];
          auto const stop_bbox = geo::make_box({from_l, to_l});
          if (!stop_bbox.contains(segment_bbox)) {
            bbox_count = i + 1U;
          }
        }
        result.segment_bboxes_.resize(bbox_count);
      } else {
        for (auto const s : *result.stop_seq_) {
          result.trip_bbox_.extend(
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

void shape_prepare::create_route_bounding_boxes(timetable const& tt) const {
  utl::verify(results_ready_, "Operation requires calculated results");
  auto const route_offset = shapes_.route_bboxes_.size();
  auto const routes_count = tt.route_transport_ranges_.size() - route_offset;
  auto route_bboxes = std::vector<geo::box>(routes_count);
  auto route_segment_bboxes = std::vector<std::vector<geo::box>>(routes_count);
  utl::parallel_for_run(route_bboxes.size(), [&](std::size_t const bbox_idx) {
    auto const r = static_cast<route_idx_t>(bbox_idx + route_offset);
    auto const seq = tt.route_location_seq_[r];
    assert(seq.size() > 0U);
    auto bounding_box = geo::box{};
    auto segment_bboxes = std::vector<geo::box>(seq.size() - 1);
    auto bbox_count = 0UL;
    auto processed =
        std::vector<cista::pair<shape_idx_t, shape_offset_idx_t>>{};
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
          bounding_box.extend(result->trip_bbox_);
          for (auto i = 0U; i < result->segment_bboxes_.size(); ++i) {
            segment_bboxes[i + cista::to_idx(absolute_range.from_)] =
                result->segment_bboxes_[i];
          }
          bbox_count = std::max(bbox_count, static_cast<unsigned long>(
                                                result->segment_bboxes_.size() +
                                                absolute_range.from_));
        }
      });
    }
    segment_bboxes.resize(bbox_count);
    route_bboxes[bbox_idx] = std::move(bounding_box);
    route_segment_bboxes[bbox_idx] = std::move(segment_bboxes);
  });
  for (auto const [route_bbox, segment_bboxes] :
       utl::zip(route_bboxes, route_segment_bboxes)) {
    shapes_.route_bboxes_.emplace_back(route_bbox);
    shapes_.route_segment_bboxes_.emplace_back(segment_bboxes);
  }
}

struct offset_task {
  struct results {
    struct result {
      std::vector<shape_offset_t> offsets_;
      geo::box trip_bbox;
      std::vector<geo::box> segment_bboxes_;
    };
    // shape_idx_t shape_idx_;
    std::vector<result> results_;
  };
  std::function<std::optional<results>()> task_;
  std::optional<results> results_;
};

std::vector<offset_task> create_offset_tasks(
    timetable const& tt,
    shapes_storage& shapes_data,
    vector_map<gtfs_trip_idx_t, trip> const& trips,
    shape_loader_state const& states) {
  using input_pair = cista::pair<stop_seq_t const*, std::vector<double> const*>;
  struct input_data {
    shape_idx_t shape_idx_;
    std::vector<input_pair> inputs_;
  };

  auto const index_offset = cista::to_idx(states.index_offset_);
  auto inputs = std::vector<input_data>(states.id_map_.size());
  for (auto [idx, input] : utl::enumerate(inputs)) {
    input.shape_idx_ = static_cast<shape_idx_t>(idx + index_offset);
  }
  for (auto const& trip : trips) {
    if (trip.shape_idx_ == shape_idx_t::invalid()) {
      continue;
    }
    auto const idx = cista::to_idx(trip.shape_idx_ - index_offset);
    auto& shape_inputs = inputs[idx].inputs_;
    auto const it = std::ranges::lower_bound(
        shape_inputs, trip.stop_seq_,
        [&](stop_seq_t const& a, stop_seq_t const& b) { return a < b; },
        [](input_pair const& p) { return *p.first; });
    if (it != end(shape_inputs) && *it->first == trip.stop_seq_) {
      continue;
    }
    auto const distances =
        trip.distance_traveled_.empty() ? nullptr : &trip.distance_traveled_;
    shape_inputs.emplace(it, &trip.stop_seq_, std::move(distances));
  }

  return utl::all(std::move(inputs))  //
         |
         utl::transform([&](input_data const& in) {
           return offset_task{
               .task_ =
                   [&, input = in, index_offset]() {
                     auto const shape = shapes_data.get_shape(input.shape_idx_);
                     auto const& shape_distances =
                         states.distances_[cista::to_idx(input.shape_idx_ -
                                                         index_offset)];
                     return std::make_optional<offset_task::results>({
                         .results_ =
                             utl::all(input.inputs_)  //
                             |
                             utl::transform(
                                 [&](input_pair const& pair)
                                     -> offset_task::results::result {
                                   auto const& [stop_seq, distances] = pair;
                                   // Calculate offsets
                                   auto const offsets = [&]() {
                                     if (!shape_distances.empty() &&
                                         distances != nullptr) {
                                       return get_offsets_by_dist_traveled(
                                           *distances, shape_distances);
                                     }
                                     if (shape.size() < stop_seq->size()) {
                                       return std::vector<shape_offset_t>{};
                                     }

                                     return get_offsets_by_stops(tt, shape,
                                                                 *stop_seq);
                                   }();
                                   // Calculate bounding boxes
                                   auto trip_bbox = geo::box{};
                                   auto segment_bboxes =
                                       std::vector<geo::box>{};
                                   if (!offsets.empty()) {
                                     segment_bboxes.resize(
                                         (offsets.size() - 1));
                                     auto bbox_count = 0UL;
                                     for (auto const [i, segment] :
                                          utl::enumerate(
                                              utl::pairwise(offsets))) {
                                       auto& segment_bbox = segment_bboxes[i];
                                       auto const& [from, to] = segment;
                                       for (auto const point : shape.subspan(
                                                cista::to_idx(from),
                                                cista::to_idx(to) -
                                                    cista::to_idx(from) + 1)) {
                                         trip_bbox.extend(point);
                                         segment_bbox.extend(point);
                                       }
                                       auto const from_l =
                                           tt.locations_.coordinates_
                                               [stop{(*stop_seq)[i]}
                                                    .location_idx()];
                                       auto const to_l =
                                           tt.locations_.coordinates_
                                               [stop{(*stop_seq)[i + 1]}
                                                    .location_idx()];
                                       auto const stop_bbox =
                                           geo::make_box({from_l, to_l});
                                       if (!stop_bbox.contains(segment_bbox)) {
                                         bbox_count = i + 1U;
                                       }
                                     }
                                     segment_bboxes.resize(bbox_count);
                                   } else {
                                     for (auto const s : *stop_seq) {
                                       trip_bbox.extend(
                                           tt.locations_.coordinates_
                                               [stop{s}.location_idx()]);
                                     }
                                   }
                                   //
                                   return offset_task::results::result{
                                       .offsets_ = std::move(offsets),
                                       .trip_bbox = std::move(trip_bbox),
                                       .segment_bboxes_ =
                                           std::move(segment_bboxes),
                                   };
                                 })  //
                             | utl::vec(),
                     });
                   },
               .results_ = std::nullopt,
           };
         })  //
         | utl::vec();
}


void calculate_shape_offsets_and_bboxes(
    timetable const& tt,
    shapes_storage& shapes_data,
    vector_map<gtfs_trip_idx_t, trip> const& trips,
    shape_loader_state const& shape_states) {
  auto offset_tasks = create_offset_tasks(tt, shapes_data, trips, shape_states);
  utl::parallel_for(offset_tasks,
                    [](offset_task& task) { task.results_ = task.task_(); });
}
}  // namespace nigiri::loader::gtfs