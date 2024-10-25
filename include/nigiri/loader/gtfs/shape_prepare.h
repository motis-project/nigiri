#pragma once

#include "nigiri/loader/gtfs/shape.h"
#include "nigiri/loader/gtfs/trip.h"

namespace nigiri {
struct shapes_storage;
struct timetable;
}  // namespace nigiri

namespace nigiri::loader::gtfs {

void calculate_shape_offsets(timetable const&,
                             shapes_storage&,
                             vector_map<gtfs_trip_idx_t, trip> const&,
                             shape_loader_state const&);

}  // namespace nigiri::loader::gtfs
