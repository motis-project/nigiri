#pragma once

#include <chrono>
#include <optional>

#include "nigiri/routing/query.h"
#include "nigiri/routing/search.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

template <typename Criteria>
struct basic_mcraptor_state;

// Restricted Pareto sets (Delling, Dibbelt, Pajor, ALENEX'19) as a profile
// search, see bmrap_profile.cc. AlgoState: the 2-criteria state (raptor_state
// or gpu::gpu_raptor_state). mc_state: the state of the mc phases when they run
// on the CPU. gpu_mc: run the mc phases on the device too if AlgoState is
// the device state and its mc state can be allocated.
template <typename Criteria, typename AlgoState>
routing_result bmrap_profile_search(
    timetable const&,
    rt_timetable const*,
    search_state&,
    AlgoState&,
    basic_mcraptor_state<Criteria>& mc_state,
    query,
    direction search_dir,
    std::optional<std::chrono::seconds> timeout = std::nullopt,
    bool gpu_mc = true);

}  // namespace nigiri::routing
