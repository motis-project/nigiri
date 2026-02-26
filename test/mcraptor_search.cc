#include "./raptor_search.h"

#include "nigiri/routing/limits.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/mcraptor/mcraptor_search.h"
#include "nigiri/routing/search.h"
#include "nigiri/timetable.h"

namespace nigiri::test {

pareto_set<routing::journey> mcraptor_search(timetable const& tt,
                                           rt_timetable const* rtt,
                                           routing::query q,
                                           direction const search_dir) {
  using algo_state_t = routing::raptor_state;
  static auto search_state = routing::search_state{};
  static auto algo_state = algo_state_t{};

  return *(routing::mcraptor_search(tt, rtt, search_state, algo_state,
                                  std::move(q), search_dir)
               .journeys_);
}

pareto_set<routing::journey> mcraptor_search(timetable const& tt,
                                           rt_timetable const* rtt,
                                           std::string_view from,
                                           std::string_view to,
                                           routing::start_time_t time,
                                           direction const search_dir,
                                           routing::clasz_mask_t const mask,
                                           bool const require_bikes_allowed,
                                           profile_idx_t const profile) {
  auto const src = source_idx_t{0};
  auto q = routing::query{
      .start_time_ = time,
      .start_ = {{tt.locations_.location_id_to_idx_.at({from, src}), 0_minutes,
                  0U}},
      .destination_ = {{tt.locations_.location_id_to_idx_.at({to, src}),
                        0_minutes, 0U}},
      .prf_idx_ = profile,
      .allowed_claszes_ = mask,
      .require_bike_transport_ = require_bikes_allowed,
      .via_stops_ = {}};
  return mcraptor_search(tt, rtt, std::move(q), search_dir);
}

}  // namespace nigiri::test
