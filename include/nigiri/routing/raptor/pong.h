#pragma once

#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/search.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

struct pong_settings {
  // Extra headroom for the ping's worst_time_at_dest beyond the query's
  // max travel time: the cap anchors at the probe, so a late-departing long
  // journey can stay invisible to every probe that could find it. Search
  // space only -- results are still filtered by max_travel_time_.
  duration_t travel_time_slack_{0};
};
extern pong_settings pong_config;

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