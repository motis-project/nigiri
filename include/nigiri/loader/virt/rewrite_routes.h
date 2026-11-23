#pragma once

#include <cstdint>

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

// Splits routes into groups of transports with equal key, removes inactive
// transports. Returns the key of each resulting route.
vector_map<route_idx_t, std::uint32_t> regroup_transports(
    timetable&,
    import_context&,
    shapes_storage*,
    vector_map<transport_idx_t, std::uint32_t> const& keys);

}  // namespace nigiri::loader
