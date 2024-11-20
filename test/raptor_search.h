#include "nigiri/routing/journey.h"

#include <optional>

#include "nigiri/routing/pareto_set.h"

namespace nigiri {
struct timetable;
struct rt_timetable;
}  // namespace nigiri

namespace nigiri::test {

unixtime_t parse_time(std::string_view s, char const* format);

pareto_set<routing::journey> raptor_search(
    timetable const&,
    rt_timetable const*,
    std::string_view from,
    std::string_view to,
    std::string_view time,
    direction = direction::kForward,
    routing::clasz_mask_t mask = routing::all_clasz_allowed(),
    bool require_bikes_allowed = false,
    std::optional<duration_t> max_travel_time = std::nullopt,
    routing::transfer_time_settings tts = {});

pareto_set<routing::journey> raptor_search(
    timetable const&,
    rt_timetable const*,
    std::string_view from,
    std::string_view to,
    routing::start_time_t,
    direction = direction::kForward,
    routing::clasz_mask_t mask = routing::all_clasz_allowed(),
    bool require_bikes_allowed = false,
    profile_idx_t const profile = 0U,
    std::optional<duration_t> max_travel_time = std::nullopt,
    routing::transfer_time_settings tts = {});

pareto_set<routing::journey> raptor_search(timetable const&,
                                           rt_timetable const*,
                                           routing::query,
                                           direction = direction::kForward);

pareto_set<routing::journey> raptor_intermodal_search(
    timetable const&,
    rt_timetable const*,
    std::vector<routing::offset> start,
    std::vector<routing::offset> destination,
    routing::start_time_t,
    direction = direction::kForward,
    std::uint8_t min_connection_count = 0U,
    bool extend_interval_earlier = false,
    bool extend_interval_later = false);

}  // namespace nigiri::test
