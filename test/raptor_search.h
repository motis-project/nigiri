#include "nigiri/routing/journey.h"

#include "nigiri/routing/pareto_set.h"

namespace nigiri {
struct timetable;
struct rt_timetable;
}  // namespace nigiri

namespace nigiri::test {

pareto_set<routing::journey> raptor_search(
    timetable const&,
    rt_timetable const*,
    std::string_view from,
    std::string_view to,
    std::string_view time,
    direction = direction::kForward,
    routing::clasz_mask_t mask = routing::all_clasz_allowed(),
    bool require_bikes_allowed = false,
    bool require_cars_allowed = false);

pareto_set<routing::journey> raptor_search(
    timetable const&,
    rt_timetable const*,
    std::string_view from,
    std::string_view to,
    routing::start_time_t,
    direction = direction::kForward,
    routing::clasz_mask_t mask = routing::all_clasz_allowed(),
    bool require_bikes_allowed = false,
    bool require_cars_allowed = false,
    profile_idx_t const profile = 0U);

pareto_set<routing::journey> raptor_search(timetable const&,
                                           rt_timetable const*,
                                           routing::query,
                                           direction = direction::kForward);

pareto_set<routing::journey> raptor_search(timetable const& tt,
                                           rt_timetable const* rtt,
                                           routing::query&& q,
                                           std::string_view from,
                                           std::string_view to,
                                           std::string_view time,
                                           direction const search_dir);

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
