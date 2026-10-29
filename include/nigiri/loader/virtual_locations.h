#pragma once

#include "nigiri/footpath.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
struct shapes_storage;
}  // namespace nigiri

namespace nigiri::loader {

void apply_transfer_rules(
    timetable&,
    shapes_storage*,
    mutable_fws_multimap<location_idx_t, footpath> const& walks);

}  // namespace nigiri::loader
