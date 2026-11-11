#pragma once

#include <chrono>
#include <optional>

#include "nigiri/routing/query.h"
#include "nigiri/routing/search.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

// Restricted pareto sets (Delling, Dibbelt, Pajor, ALENEX'19,
// doi:10.1137/1.9781611975499.5) as a PROFILE search: a complete
// single-departure BM-RAPTOR per step of a PONG-style scan rather than one
// bound matrix for the whole window; see bmrap_profile.cc.
//
// AlgoState is the scalar (two-criteria) state of ping, pong and the backward
// pruning searches (raptor_state or gpu::gpu_raptor_state, chosen like
// pong_search); Criteria is the multicriteria configuration of the main search.
//
// gpu_mc: with a device scalar state, run the multicriteria phases on the
// device as well, for the combinations the device mcraptor implements
// (kGpuMcSupported in bmrap_common.h). They run on the CPU when false, when
// the device mc state cannot be allocated (gpu_raptor_state::try_mc_state())
// and for every other combination. motis sets it for the pool slots it gave a
// device mc state at startup (server.gpu_mc_states), so no query allocates one.
template <typename Criteria, typename AlgoState>
routing_result bmrap_profile_search(
    timetable const&,
    rt_timetable const*,
    search_state&,
    AlgoState&,
    query,
    direction search_dir,
    std::optional<std::chrono::seconds> timeout = std::nullopt,
    bool gpu_mc = true);

}  // namespace nigiri::routing
