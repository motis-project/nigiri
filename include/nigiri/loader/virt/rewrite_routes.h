#pragma once

#include "nigiri/types.h"

namespace nigiri {
struct timetable;
struct shapes_storage;
}  // namespace nigiri

namespace nigiri::loader {

struct import_context;

void rewrite_routes(
    timetable&,
    import_context&,
    shapes_storage*,
    vector_map<route_idx_t, route_idx_t> const& base_route,
    vecvec<route_idx_t, transport_idx_t> const& route_transports);

}  // namespace nigiri::loader
