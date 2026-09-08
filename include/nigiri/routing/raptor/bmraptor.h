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
// A range variant - one window-wide tau_dep^<- matrix plus a per-departure
// trip budget, sharing everything in bmrap_common.h with this one - existed
// alongside it and was removed: its backward pruning cost grows linearly
// with the number of window slices (492 -> 916 -> 1431 -> 2143 -> 2762 ms
// for 1/2/3/5/7 slices) while the main search barely improves (825 -> 723
// ms), so it lost to the per-departure bounds this driver builds, and the
// unbounded range McRAPTOR it shared its shape with is available on its own.
//
// AlgoState is the SCALAR (two-criteria) state the ping, the pong and the
// backward pruning searches run on - raptor_state or gpu::gpu_raptor_state,
// selected exactly the way pong_search selects its engine. Criteria picks
// the multicriteria configuration of the main search, which always runs on
// the CPU (see bmrap_algo_for in bmrap_common.h).
template <typename Criteria, typename AlgoState>
routing_result bmrap_profile_search(
    timetable const&,
    rt_timetable const*,
    search_state&,
    AlgoState&,
    query,
    direction search_dir,
    std::optional<std::chrono::seconds> timeout = std::nullopt);

}  // namespace nigiri::routing
