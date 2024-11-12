#include "nigiri/loader/gtfs/shape_prepare.h"

#include <algorithm>
#include <ranges>
#include <span>
#include <type_traits>

#include "cista/strong.h"

#include "geo/box.h"
#include "geo/latlng.h"
#include "geo/polyline.h"

#include "utl/enumerate.h"
#include "utl/helpers/algorithm.h"
#include "utl/insert_sorted.h"
#include "utl/pairwise.h"
#include "utl/parallel_for.h"
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

struct stop_seq_dist {
  bool operator<(stop_seq_dist const& o) const {
    return *stop_seq_ < *o.stop_seq_;
  }
  bool operator==(stop_seq_dist const& o) const {
    return *stop_seq_ == *o.stop_seq_;
  }
  bool operator!=(stop_seq_dist const& o) const {
    return *stop_seq_ != *o.stop_seq_;
  }

  stop_seq_t const* stop_seq_;
  std::vector<double> const* dist_traveled_;
  struct result {
    shape_offset_idx_t shape_offset_idx_{shape_offset_idx_t::invalid()};
    std::vector<shape_offset_t> offsets_;
    geo::box trip_bbox_;
    std::vector<geo::box> segment_bboxes_;
  } result_{};
};

using task = std::vector<stop_seq_dist>;

vector_map<relative_shape_idx_t, task> create_offset_tasks(
    vector_map<gtfs_trip_idx_t, trip> const& trips,
    shape_loader_state const& states) {
  auto tasks = vector_map<relative_shape_idx_t, task>{};
  tasks.resize(static_cast<shape_idx_t::value_t>(states.id_map_.size()));
  for (auto const& trip : trips) {
    if (trip.shape_idx_ == shape_idx_t::invalid()) {
      continue;
    }
    utl::insert_sorted(
        tasks[states.get_relative_idx(trip.shape_idx_)],
        {&trip.stop_seq_,
         trip.distance_traveled_.empty() ? nullptr : &trip.distance_traveled_});
  }
  return tasks;
}

void process_task(timetable const& tt,
                  shapes_storage const& shapes_data,
                  shape_loader_state const& shape_states,
                  relative_shape_idx_t const i,
                  task& t) {
  auto const shape = shapes_data.get_shape(shape_states.get_shape_idx(i));
  auto const& shape_distances = shape_states.distances_[i];
  for (auto& x : t) {
    auto& r = x.result_;
    auto const& [stop_seq, distances] = std::tie(x.stop_seq_, x.dist_traveled_);

    // Calculate offsets
    r.offsets_ = (shape.size() < stop_seq->size())
                     ? std::vector<shape_offset_t>{}
                 : (!shape_distances.empty() && distances != nullptr)
                     ? get_offsets_by_dist_traveled(*distances, shape_distances)
                     : get_offsets_by_stops(tt, shape, *stop_seq);

    // Calculate bounding boxes
    if (r.offsets_.empty()) {
      for (auto const s : *stop_seq) {
        r.trip_bbox_.extend(tt.locations_.coordinates_[stop{s}.location_idx()]);
      }
    } else {
      r.segment_bboxes_.resize(r.offsets_.size() - 1);
      auto is_trivial = true;
      for (auto const [segment_idx, segment] :
           utl::enumerate(utl::pairwise(r.offsets_))) {
        auto const& [from, to] = segment;
        for (auto const point :
             shape.subspan(cista::to_idx(from), cista::to_idx(to - from + 1))) {
          r.trip_bbox_.extend(point);
          r.segment_bboxes_[segment_idx].extend(point);
        }
        auto const from_l =
            tt.locations_
                .coordinates_[stop{(*stop_seq)[segment_idx]}.location_idx()];
        auto const to_l =
            tt.locations_.coordinates_[stop{(*stop_seq)[segment_idx + 1]}
                                           .location_idx()];
        auto const stop_bbox = geo::make_box({from_l, to_l});
        if (!stop_bbox.contains(r.segment_bboxes_[segment_idx])) {
          is_trivial = false;
        }
      }
      if (is_trivial) {
        r.segment_bboxes_.clear();
      }
    }
  }
}

void assign_shape_offsets(shapes_storage& shapes_data,
                          vector_map<gtfs_trip_idx_t, trip> const& trips,
                          vector_map<relative_shape_idx_t, task>& tasks,
                          shape_loader_state const& states) {
  for (auto& shape_tasks : tasks) {
    for (auto& task : shape_tasks) {
      auto& r = task.result_;
      if (!r.offsets_.empty()) {
        r.shape_offset_idx_ = shapes_data.add_offsets(std::move(r.offsets_));
      }
    }
  }
  for (auto const& trip : trips) {
    auto const trip_idx = trip.trip_idx_;
    auto const shape_idx = trip.shape_idx_;
    if (shape_idx == shape_idx_t::invalid()) {
      shapes_data.add_trip_shape_offsets(
          trip_idx, cista::pair{shape_idx, shape_offset_idx_t::invalid()});
    } else {
      auto const shape_tasks = tasks[states.get_relative_idx(shape_idx)];
      auto const task = std::ranges::lower_bound(
          shape_tasks, trip.stop_seq_,
          [&](stop_seq_t const& a, stop_seq_t const& b) { return a < b; },
          [](stop_seq_dist const& s) { return *s.stop_seq_; });
      shapes_data.add_trip_shape_offsets(
          trip_idx, cista::pair{shape_idx, task->result_.shape_offset_idx_});
    }
  }
}

void assign_bounding_boxes(timetable const& tt,
                           shapes_storage& shapes_data,
                           vector_map<relative_shape_idx_t, task>& tasks,
                           shape_loader_state const& shape_states) {
  auto const new_routes =
      interval{static_cast<route_idx_t>(shapes_data.route_bboxes_.size()),
               static_cast<route_idx_t>(tt.route_transport_ranges_.size())};
  for (auto const r : new_routes) {
    auto const seq = tt.route_location_seq_[r];
    assert(seq.size() > 0U);
    auto bounding_box = geo::box{};
    auto segment_bboxes = std::vector<geo::box>{};
    segment_bboxes.resize(seq.size() - 1);
    auto const stop_indices =
        interval{stop_idx_t{0U}, static_cast<stop_idx_t>(seq.size())};

    // Create basic bounding boxes, span by stops
    for (auto const [i, s] : utl::enumerate(stop_indices)) {
      auto const pos = tt.locations_.coordinates_[stop{seq[s]}.location_idx()];
      bounding_box.extend(pos);
      if (i > 0U) {
        segment_bboxes[i - 1U].extend(pos);
      }
      if (i < segment_bboxes.size()) {
        segment_bboxes[i].extend(pos);
      }
    }

    auto is_trivial = true;

    for (auto const transport_idx : tt.route_transport_ranges_[r]) {
      auto const frun = rt::frun{tt, nullptr,
                                 rt::run{.t_ = transport{transport_idx},
                                         .stop_range_ = stop_indices,
                                         .rt_ = rt_transport_idx_t::invalid()}};
      frun.for_each_trip([&](trip_idx_t const trip_idx,
                             interval<stop_idx_t> const absolute_range) {
        auto const [shape_idx, offset_idx] =
            shapes_data.trip_offset_indices_[trip_idx];
        if (shape_idx == shape_idx_t::invalid() ||
            offset_idx == shape_offset_idx_t::invalid()) {
          return;
        }
        auto const& shape_tasks =
            tasks[shape_states.get_relative_idx(shape_idx)];
        auto const it = utl::find_if(shape_tasks, [&](stop_seq_dist const& s) {
          return s.result_.shape_offset_idx_ == offset_idx;
        });

        auto const& res = it->result_;
        bounding_box.extend(res.trip_bbox_);
        auto const& bboxes = res.segment_bboxes_;
        if (!bboxes.empty()) {
          for (auto const [i, bbox] : utl::enumerate(bboxes)) {
            segment_bboxes[i + cista::to_idx(absolute_range.from_)].extend(
                bbox);
          }
          is_trivial = false;
        }
      });
    }

    if (is_trivial) {
      segment_bboxes.clear();
    }

    shapes_data.route_bboxes_.emplace_back(std::move(bounding_box));
    shapes_data.route_segment_bboxes_.emplace_back(std::move(segment_bboxes));
  }
}

void calculate_shape_offsets_and_bboxes(
    timetable const& tt,
    shapes_storage& shapes_data,
    shape_loader_state const& shape_states,
    vector_map<gtfs_trip_idx_t, trip> const& trips) {
  utl::get_active_progress_tracker()
      ->status("Creating trip offsets")
      .out_bounds(98.F, 100.F);
  auto tasks = create_offset_tasks(trips, shape_states);
  utl::parallel_for_run(tasks.size(), [&](std::size_t const i) {
    auto const s = relative_shape_idx_t{i};
    process_task(tt, shapes_data, shape_states, s, tasks[s]);
  });

  assign_shape_offsets(shapes_data, trips, tasks, shape_states);
  assign_bounding_boxes(tt, shapes_data, tasks, shape_states);
}

}  // namespace nigiri::loader::gtfs