#pragma once

#include "nigiri/loader/import_context.h"
#include "nigiri/footpath.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri {
struct shapes_storage;
}  // namespace nigiri

namespace nigiri::loader {

void apply_transfer_rules(
    timetable&,
    import_context&,
    shapes_storage*,
    mutable_fws_multimap<location_idx_t, footpath>& walks);

}  // namespace nigiri::loader
