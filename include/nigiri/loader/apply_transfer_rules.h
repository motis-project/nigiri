#pragma once

#include "nigiri/footpath.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
struct shapes_storage;
}  // namespace nigiri

namespace nigiri::loader {

struct import_context;

void apply_transfer_rules(
    timetable&,
    import_context&,
    shapes_storage*,
    mutable_fws_multimap<location_idx_t, footpath>& walks);

}  // namespace nigiri::loader
