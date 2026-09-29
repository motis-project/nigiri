#pragma once

#include <chrono>
#include <optional>

#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/mcraptor.h"
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

// The RANGE variant bmrap_profile_search replaced, kept for benchmarking the
// two against each other (bmraptor.cc). Three phases, all range searches over
// the query window:
//
//  1. anchor search: the two-criteria (time, trips) range search (PONG where
//     applicable, rRAPTOR otherwise) yielding the anchor pareto set J_A,
//     closed past the window by close_anchor_profile().
//  2. backward pruning: compute_bounds() over ALL anchors at once, i.e. ONE
//     tau_dep^<- matrix for the whole window. A departure's slack is measured
//     from that departure, so the union is dominated by the window's LAST
//     departure and earlier ones are bounded more loosely, by up to the window
//     width. The trip budget stays per-departure (search::max_transfers_fn_).
//  3. main search: the bounded range McRAPTOR over the anchor window, finally
//     restricted to J_R via outside_restriction().
//
// CPU only; runs every phase on its own raptor_state and the given mc state.
template <typename Criteria>
routing_result bmrap_range_search(
    timetable const&,
    rt_timetable const*,
    search_state&,
    basic_mcraptor_state<Criteria>&,
    query,
    direction search_dir,
    std::optional<std::chrono::seconds> timeout = std::nullopt);

}  // namespace nigiri::routing
