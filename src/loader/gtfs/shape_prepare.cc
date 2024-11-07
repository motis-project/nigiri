#include "nigiri/loader/gtfs/shape_prepare.h"

#include <algorithm>
#include <optional>
#include <ranges>
#include <span>
#include <type_traits>

#include "cista/strong.h"

#include "geo/box.h"
#include "geo/latlng.h"
#include "geo/polyline.h"

#include "utl/enumerate.h"
#include "utl/parallel_for.h"
#include "utl/pipes/all.h"
#include "utl/pipes/transform.h"
#include "utl/pipes/vec.h"
#include "utl/progress_tracker.h"

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

struct offset_task {
  struct result {
    stop_seq_t const* stop_seq_;
    std::vector<shape_offset_t> offsets_;
    geo::box trip_bbox_;
    std::vector<geo::box> segment_bboxes_;
  };
  std::function<std::optional<std::vector<result>>()> task_;
  std::optional<std::vector<result>> results_;
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
                     return std::make_optional<
                         std::vector<offset_task::result>>({
                         utl::all(input.inputs_)  //
                             |
                             utl::transform([&](input_pair const& pair)
                                                -> offset_task::result {
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
                               auto segment_bboxes = std::vector<geo::box>{};
                               if (!offsets.empty()) {
                                 segment_bboxes.resize((offsets.size() - 1));
                                 auto bbox_count = 0UL;
                                 for (auto const [i, segment] :
                                      utl::enumerate(utl::pairwise(offsets))) {
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
                                       tt.locations_
                                           .coordinates_[stop{(*stop_seq)[i]}
                                                             .location_idx()];
                                   auto const to_l =
                                       tt.locations_
                                           .coordinates_[stop{
                                               (*stop_seq)[i + 1]}
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
                               // Create result
                               return offset_task::result{
                                   .stop_seq_ = pair.first,
                                   .offsets_ = std::move(offsets),
                                   .trip_bbox_ = std::move(trip_bbox),
                                   .segment_bboxes_ = std::move(segment_bboxes),
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

struct bbox_and_offset_data {
  struct trip {
    stop_seq_t const* stop_seq_;
    shape_offset_idx_t shape_offset_idx_;
    geo::box trip_bbox_;
    std::vector<geo::box> segment_bboxes_;
  };
  std::vector<trip> trips_;
};

std::vector<bbox_and_offset_data> assign_shape_offsets(
    shapes_storage& shapes_data,
    vector_map<gtfs_trip_idx_t, trip> const& trips,
    std::vector<offset_task>&& tasks,
    shape_idx_t const index_offset) {
  assert(utl::all_of(tasks, [](offset_task const& task) {
    return task.results_ != std::nullopt;
  }));
  auto const entries =
      utl::all(std::move(tasks))  //
      | utl::transform([&](offset_task const& task) {
          return bbox_and_offset_data{
              .trips_ =
                  utl::all(*task.results_)  //
                  | utl::transform([&](offset_task::result const& result) {
                      return bbox_and_offset_data::trip{
                          .stop_seq_ = result.stop_seq_,
                          .shape_offset_idx_ =
                              result.offsets_.empty()
                                  ? shape_offset_idx_t::invalid()
                                  : shapes_data.add_offsets(result.offsets_),
                          .trip_bbox_ = result.trip_bbox_,
                          .segment_bboxes_ = result.segment_bboxes_,
                      };
                    })  //
                  | utl::vec(),
          };
        }) |
      utl::vec();
  for (auto const& trip : trips) {
    auto const trip_idx = trip.trip_idx_;
    auto const shape_idx = trip.shape_idx_;
    if (shape_idx == shape_idx_t::invalid()) {
      shapes_data.add_trip_shape_offsets(
          trip_idx, cista::pair{shape_idx, shape_offset_idx_t::invalid()});
    } else {
      auto const& results =
          entries[cista::to_idx(shape_idx - index_offset)].trips_;
      auto const it = std::ranges::lower_bound(
          results, trip.stop_seq_,
          [&](stop_seq_t const& a, stop_seq_t const& b) { return a < b; },
          [](bbox_and_offset_data::trip const& entry) {
            return *entry.stop_seq_;
          });
      shapes_data.add_trip_shape_offsets(
          trip_idx, cista::pair{shape_idx, it->shape_offset_idx_});
    }
  }
  return entries;
}

struct bbox_task {
  struct result {
    geo::box route_bbox_;
    std::vector<geo::box> route_segment_bboxes_;
  };
  std::function<std::optional<result>()> task_;
  std::optional<result> result_;
};

std::vector<bbox_task> create_bbox_tasks(
    timetable const& tt,
    shapes_storage& shapes_data,
    std::vector<bbox_and_offset_data> const& bbox_data,
    std::size_t const route_offset,
    shape_idx_t const shape_offset) {
  assert(utl::all_of(tasks, [](offset_task const& task) {
    return task.results_ != std::nullopt;
  }));
  auto const new_routes =
      interval{static_cast<route_idx_t>(route_offset),
               static_cast<route_idx_t>(tt.route_transport_ranges_.size() -
                                        route_offset)};
  return utl::all(new_routes)  //
         |
         utl::transform([&](route_idx_t const& r) {
           return bbox_task{
               .task_ =
                   [&, r, shape_offset]() {
                     auto const seq = tt.route_location_seq_[r];
                     assert(seq.size() > 0U);
                     auto bounding_box = geo::box{};
                     auto segment_bboxes = std::vector<geo::box>{};
                     segment_bboxes.resize(seq.size() - 1);
                     auto bbox_count = 0UL;
                     auto const stop_indices = interval{
                         stop_idx_t{0U}, static_cast<stop_idx_t>(seq.size())};

                     for (auto const transport_idx :
                          tt.route_transport_ranges_[r]) {
                       auto const frun = rt::frun{
                           tt, nullptr,
                           rt::run{.t_ = transport{transport_idx},
                                   .stop_range_ = stop_indices,
                                   .rt_ = rt_transport_idx_t::invalid()}};
                       frun.for_each_trip(
                           [&](trip_idx_t const trip_idx,
                               interval<stop_idx_t> const absolute_range) {
                             auto const [shape_idx, offset_idx] =
                                 shapes_data.trip_offset_indices_[trip_idx];
                             if (shape_idx == shape_idx_t::invalid() ||
                                 offset_idx == shape_offset_idx_t::invalid()) {
                               for (auto const idx : absolute_range) {
                                 bounding_box.extend(
                                     tt.locations_.coordinates_
                                         [stop{seq[idx]}.location_idx()]);
                               }
                             } else {
                               auto const result = std::ranges::lower_bound(
                                   bbox_data[cista::to_idx(shape_idx -
                                                           shape_offset)]
                                       .trips_,
                                   offset_idx,
                                   [](shape_offset_idx_t const a,
                                      shape_offset_idx_t const b) {
                                     return a < b;
                                   },
                                   [](bbox_and_offset_data::trip const& trip) {
                                     return trip.shape_offset_idx_;
                                   });
                               bounding_box.extend(result->trip_bbox_);
                               for (auto i = 0U;
                                    i < result->segment_bboxes_.size(); ++i) {
                                 segment_bboxes[i + cista::to_idx(
                                                        absolute_range.from_)] =
                                     result->segment_bboxes_[i];
                               }
                               bbox_count =
                                   std::max(bbox_count,
                                            static_cast<unsigned long>(
                                                result->segment_bboxes_.size() +
                                                absolute_range.from_));
                             }
                           });
                     }

                     return std::make_optional(bbox_task::result{
                         std::move(bounding_box), std::move(segment_bboxes)});
                   },
               .result_ = std::nullopt,
           };
         }) |
         utl::vec();
}

void assign_bounding_boxes(shapes_storage& shapes_data,
                           std::vector<bbox_task> const&& tasks) {
  for (auto const& task : tasks) {
    shapes_data.route_bboxes_.emplace_back(task.result_->route_bbox_);
    shapes_data.route_segment_bboxes_.emplace_back(
        task.result_->route_segment_bboxes_);
  }
}

void calculate_shape_offsets_and_bboxes(
    timetable const& tt,
    shapes_storage& shapes_data,
    vector_map<gtfs_trip_idx_t, trip> const& trips,
    shape_loader_state const& shape_states) {
  auto offset_tasks = create_offset_tasks(tt, shapes_data, trips, shape_states);
  auto const offset_progress_tracker = utl::get_active_progress_tracker();
  offset_progress_tracker->status("Creating trip offsets")
      .out_bounds(98.F, 99.F)
      .in_high(offset_tasks.size());
  utl::parallel_for(offset_tasks,
                    [&offset_progress_tracker](offset_task& task) {
                      offset_progress_tracker->increment();
                      task.results_ = task.task_();
                    });
  auto bbox_data = assign_shape_offsets(
      shapes_data, trips, std::move(offset_tasks), shape_states.index_offset_);

  auto bbox_tasks = create_bbox_tasks(tt, shapes_data, bbox_data,
                                      shapes_data.route_bboxes_.size(),
                                      shape_states.index_offset_);
  auto const bbox_progress_tracker = utl::get_active_progress_tracker();
  bbox_progress_tracker->status("Creating bounding boxes")
      .out_bounds(99.F, 100.F)
      .in_high(bbox_tasks.size());
  utl::parallel_for(bbox_tasks, [&bbox_progress_tracker](bbox_task& task) {
    bbox_progress_tracker->increment();
    task.result_ = task.task_();
  });
  assign_bounding_boxes(shapes_data, std::move(bbox_tasks));
}

}  // namespace nigiri::loader::gtfs