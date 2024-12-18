#pragma once

#include "nigiri/loader/gtfs/trip.h"
#include "nigiri/types.h"

namespace nigiri {
struct shape_loader_state;
struct shapes_storage;
struct timetable;
}  // namespace nigiri

namespace nigiri::loader::gtfs {

void calculate_shape_offsets_and_bboxes(
    timetable const&,
    shapes_storage&,
    shape_loader_state const&,
    vector_map<gtfs_trip_idx_t, trip> const&);

}  // namespace nigiri::loader::gtfs
