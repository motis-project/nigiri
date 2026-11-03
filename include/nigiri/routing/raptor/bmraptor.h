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
// gpu_mc_mode: how much multicriteria work runs on the device, for the
// combinations the device mcraptor implements (kGpuMcSupported in
// bmrap_common.h); ignored otherwise.
//   0  all on the CPU
//   1  the mc ping only: one big search per step, ~2.5x faster on the device
//   2  the mc pong as well: many tiny per-journey searches the device loses
//      on, so opt-in
// motis maps server.gpu_mc_states to it (0 -> 1, >0 -> 2).
inline constexpr int kBmrapGpuMcModeDefault = 1;

template <typename Criteria, typename AlgoState>
routing_result bmrap_profile_search(
    timetable const&,
    rt_timetable const*,
    search_state&,
    AlgoState&,
    query,
    direction search_dir,
    std::optional<std::chrono::seconds> timeout = std::nullopt,
    int gpu_mc_mode = kBmrapGpuMcModeDefault);

}  // namespace nigiri::routing
