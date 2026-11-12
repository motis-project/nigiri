#pragma once

#include "nigiri/loader/import_context.h"
#include "nigiri/shapes_storage.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::loader {

void rewrite_routes(
    timetable&,
    import_context&,
    shapes_storage*,
    vector_map<route_idx_t, route_idx_t> const& base_route,
    vecvec<route_idx_t, transport_idx_t> const& route_transports);

}  // namespace nigiri::loader
