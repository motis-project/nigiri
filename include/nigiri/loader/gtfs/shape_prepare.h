#pragma once

#include "nigiri/loader/gtfs/shape.h"
#include "nigiri/loader/gtfs/trip.h"

namespace nigiri {
struct shapes_storage;
struct timetable;
}  // namespace nigiri

namespace nigiri::loader::gtfs {

// TODO Rename
struct trip_shapes {
  trip_shapes(shape_loader_state const&,
                            vector_map<gtfs_trip_idx_t, trip> const&);
  shape_idx_t index_offset_;
  std::vector<std::vector<stop_seq_t const*>> stop_sequences_;
};

trip_shapes get_shape_pairs(shape_loader_state const&,
                            vector_map<gtfs_trip_idx_t, trip> const&);

void calculate_shape_offsets(timetable const&,
                             shapes_storage&,
                             vector_map<gtfs_trip_idx_t, trip> const&,
                             shape_loader_state const&);

void calculate_shape_boxes(timetable const&, shapes_storage&);

}  // namespace nigiri::loader::gtfs
