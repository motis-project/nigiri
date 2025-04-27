#pragma once

#include "geo/latlng.h"

#include "nigiri/types.h"

namespace nigiri {

struct timetable;

bool is_within(timetable const&, flex_area_idx_t, geo::latlng const&);

}  // namespace nigiri