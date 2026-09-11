#pragma once

#include <chrono>
#include <optional>

#include "nigiri/routing/query.h"
#include "nigiri/routing/search.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

// Restricted pareto sets (Delling, Dibbelt, Pajor: "Fast and Exact Public
// Transit Routing with Restricted Pareto Sets", ALENEX'19,
// doi:10.1137/1.9781611975499.5), as a PROFILE search: a complete
// single-departure BM-RAPTOR per step of a PONG-style scan (ping, pong,
// slacked pong, mc ping, mc pong) rather than one bound matrix for the
// whole window. See bmrap_profile.cc for the phases and for why.
//
// AlgoState is the SCALAR (two-criteria) state the ping, the pong and the
// backward pruning searches run on - raptor_state or gpu::gpu_raptor_state,
// selected exactly the way pong_search selects its engine. Criteria picks
// the multicriteria configuration of the main search.
//
// gpu_mc_mode: how much of the multicriteria work runs on the device, for
// the AlgoState + Criteria combinations the device mcraptor implements (see
// kGpuMcSupported in bmrap_common.h); ignored otherwise.
//   0  everything on the CPU
//   1  the mc PING only (phases 4 and 5b) - phase 4 is one big search per
//      step and gains ~2.5x on the device
//   2  the mc PONG (phase 5) as well - a sequence of tiny per-journey
//      searches; the device loses on those, so this is opt-in
// motis drives it via server.gpu_mc_states (0 -> mode 1, >0 -> mode 2).
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
