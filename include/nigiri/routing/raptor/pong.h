#pragma once

#include <atomic>
#include <cstdint>

#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/search.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

static constexpr auto kMinLookAhead = 1_days;

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

// how often the multi-world sweep hits its special paths (experiment
// instrumentation for the combined scheduled+rt pong)
struct srt_pong_counters {
  std::atomic<std::uint64_t> sweep_iterations_{0U};
  std::atomic<std::uint64_t> pong_runs_{0U};
  std::atomic<std::uint64_t> dominated_refinds_{0U};
  std::atomic<std::uint64_t> all_filtered_continues_{0U};
  std::atomic<std::uint64_t> flipped_filtered_{0U};
};
srt_pong_counters const& get_srt_pong_counters();

// combined scheduled+rt search (schedrt_criterion, two label slots, one
// ping+pong sweep); copy_on_diverge stores the rt world as a sparse
// overlay over the scheduled base plane
routing_result pong_search_srt(
    timetable const&,
    rt_timetable const*,
    search_state&,
    raptor_state&,
    query,
    direction search_dir,
    std::optional<std::chrono::seconds> timeout = std::nullopt,
    bool copy_on_diverge = false);

}  // namespace nigiri::routing