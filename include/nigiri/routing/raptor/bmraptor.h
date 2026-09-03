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
// doi:10.1137/1.9781611975499.5) as a RANGE raptor over the query's
// departure (resp. arrival) window.
//
// Three phases, all of them range searches:
//
//  1. FORWARD PRUNING SEARCH - the two-criteria (time, trips) range search
//     over the whole window. Its result is the anchor pareto set J_A: for
//     every departure the earliest arrival per number of trips. PONG is
//     used when it is applicable, plain rRAPTOR otherwise; both produce the
//     same set.
//
//  2. BACKWARD PRUNING SEARCH - one reverse one-to-all raptor per anchor,
//     started at that anchor's slack-relaxed time and capped at that
//     anchor's slack-relaxed trip budget, all of them accumulating into one
//     round-times matrix (rRAPTOR reuse). Yields tau_dep^<-(v, i) - see
//     bmrap_bounds.
//
//  3. MAIN SEARCH - the range McRAPTOR (arrival time + generalized cost,
//     transfers as an implicit dimension) over the same window, discarding
//     every arrival the bounds of phase 2 rule out. The result is finally
//     restricted to the paper's J_R: a journey survives iff neither its
//     trip count nor its travel time exceeds sigma_tr / sigma_arr times
//     that of its anchor journey.
//
// Slack parameters (defaults sigma_arr = sigma_tr = 1.25) are overridable
// via NIGIRI_BMRAP_ARR_SLACK / NIGIRI_BMRAP_TRIP_SLACK.
template <typename AlgoState>
routing_result bmrap_search(
    timetable const&,
    rt_timetable const*,
    search_state&,
    AlgoState&,
    query,
    direction search_dir,
    std::optional<std::chrono::seconds> timeout = std::nullopt);

}  // namespace nigiri::routing
