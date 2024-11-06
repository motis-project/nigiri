#pragma once

#include "geo/box.h"

#include "nigiri/loader/gtfs/trip.h"
#include "nigiri/types.h"

namespace nigiri {
struct shape_loader_state;
struct shapes_storage;
struct timetable;
}  // namespace nigiri

namespace nigiri::loader::gtfs {

struct shape_prepare {
  struct shape_results {
    struct result {
      explicit result(stop_seq_t const* stop_seq,
                      std::vector<double> const* distances);
      stop_seq_t const* stop_seq_;
      std::vector<double> const* distances_;
      shape_offset_idx_t offset_idx_;
      geo::box trip_bbox_;
      std::vector<geo::box> segment_bboxes_;
    };

    shape_idx_t shape_idx_;
    std::vector<result> results_;
  };

  shape_prepare(shape_loader_state const&,
                vector_map<gtfs_trip_idx_t, trip> const&,
                shapes_storage&);
  void calculate_results(timetable const&, shape_loader_state const&);
  void create_trip_shape_offsets(
      vector_map<gtfs_trip_idx_t, trip> const&) const;
  void create_route_bounding_boxes(timetable const&) const;
  shape_idx_t index_offset_;

  std::vector<shape_results> shape_results_;
  shapes_storage& shapes_;
  bool results_ready_;
};

}  // namespace nigiri::loader::gtfs
