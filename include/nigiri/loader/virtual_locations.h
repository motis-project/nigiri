#pragma once

#include "nigiri/footpath.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
struct shapes_storage;
}  // namespace nigiri

namespace nigiri::loader {

bool add_virtual_locations(timetable&,
                           shapes_storage*,
                           mutable_fws_multimap<location_idx_t, footpath>&);

}  // namespace nigiri::loader
