#pragma once

#include "nigiri/routing/limits.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
struct footpath;
}  // namespace nigiri

namespace nigiri::routing {
struct query;
struct raptor_state;

void lb_raptor(
    timetable const&,
    query const&,
    vecvec<location_idx_t, footpath> const& lb_graph,
    bitvec_map<location_idx_t> const* has_rt,
    vecvec<location_idx_t, footpath> const* rt_lb_graph,
    raptor_state&,
    vector_map<location_idx_t, std::array<std::uint16_t, kMaxTransfers + 2>>&
        location_round_lb);

}  // namespace nigiri::routing