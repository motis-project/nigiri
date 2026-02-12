#include "nigiri/routing/journey.h"

#include "nigiri/routing/pareto_set.h"

namespace nigiri {
struct timetable;
struct rt_timetable;
}  // namespace nigiri

namespace nigiri::test {

pareto_set<routing::journey> mcraptor_search(
    timetable const&,
    rt_timetable const*,
    std::string_view from,
    std::string_view to,
    routing::start_time_t,
    direction = direction::kForward,
    routing::clasz_mask_t mask = routing::all_clasz_allowed(),
    bool require_bikes_allowed = false,
    profile_idx_t const profile = 0U);

pareto_set<routing::journey> mcraptor_search(timetable const&,
                                           rt_timetable const*,
                                           routing::query,
                                           direction = direction::kForward);

}  // namespace nigiri::test
