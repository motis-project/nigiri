#pragma once

#include "geo/box.h"

#include "nigiri/loader/gtfs/shape.h"
#include "nigiri/loader/gtfs/trip.h"
#include "nigiri/types.h"

namespace nigiri {
struct shapes_storage;
struct timetable;
}  // namespace nigiri

namespace nigiri::loader::gtfs {

struct shape_segment {
  explicit shape_segment(stop_seq_t const* stop_seq,
                         std::vector<double> const* distances);
  stop_seq_t const* stop_seq_;
  shape_offset_idx_t offset_idx_;
  std::vector<double> const* distances_;
  std::vector<geo::box> boxes_;
};

struct shape_segments {
  shape_idx_t shape_idx_;
  std::vector<shape_segment> offsets_;
};

// TODO Rename
struct trip_shapes {
  trip_shapes(shape_loader_state const&,
              vector_map<gtfs_trip_idx_t, trip> const&);
  void calculate_shape_offsets(timetable const&,
                               shapes_storage*,
                               shape_loader_state const&);
  void store_offsets(vector_map<gtfs_trip_idx_t, trip> const&) const;
  void create_boxes(timetable const&) const;
  shape_idx_t index_offset_;
  std::vector<shape_segments> shape_segments_;
  shapes_storage* shapes_;
};

trip_shapes get_shape_pairs(shape_loader_state const&,
                            vector_map<gtfs_trip_idx_t, trip> const&);

void calculate_shape_offsets(timetable const&,
                             shapes_storage&,
                             vector_map<gtfs_trip_idx_t, trip> const&,
                             shape_loader_state const&);

void calculate_shape_boxes(timetable const&, shapes_storage&);

}  // namespace nigiri::loader::gtfs
