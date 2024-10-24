#pragma once

#include "nigiri/loader/gtfs/shape.h"
#include "nigiri/loader/gtfs/trip.h"
#include "nigiri/shape.h"
#include "nigiri/timetable.h"

namespace nigiri::loader::gtfs {

void calculate_shape_offsets(timetable const&,
                             shapes_storage&,
                             vector_map<gtfs_trip_idx_t, trip> const&,
                             shape_loader_state const&);

void calculate_shape_boxes(timetable const&, shapes_storage&);

}  // namespace nigiri::loader::gtfs
