#include "nigiri/routing/lb_raptor.h"

namespace nigiri::routing {

void lb_raptor(
    timetable const& tt,
    query const& q,
    vecvec<location_idx_t, footpath> const& lb_graph,
    bitvec_map<location_idx_t> const* has_rt,
    vecvec<location_idx_t, footpath> const* rt_lb_graph,
    vector_map<location_idx_t, std::array<std::uint16_t, kMaxTransfers + 2>>&
        location_round_lb) {}

}  // namespace nigiri::routing
