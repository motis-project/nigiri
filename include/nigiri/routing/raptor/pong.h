#pragma once

#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/search.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

// Ping-bounds pruning experiment switches. Initialized from the environment
// (NIGIRI_PONG_PRUNE=0 disables pruning, NIGIRI_PONG_NO_LB=1 skips the
// dijkstra lower bounds run for the pong direction when pruning is active),
// can be overridden programmatically (e.g. for A/B benchmark cells).
struct pong_prune_settings {
  bool prune_;
  bool skip_pong_lb_;
};
extern pong_prune_settings pong_prune;

std::optional<std::array<journey::leg, 3U>> get_earliest_alternative(
    timetable const&,
    rt_timetable const*,
    query const&,
    location_idx_t from,
    location_idx_t to,
    unixtime_t from_arr,
    unixtime_t to_dep);

template <typename AlgoState>
routing_result pong_search(
    timetable const&,
    rt_timetable const*,
    search_state&,
    AlgoState&,
    query,
    direction search_dir,
    std::optional<std::chrono::seconds> timeout = std::nullopt);

}  // namespace nigiri::routing