#include "nigiri/routing/mcraptor/mcraptor_search.h"

#include <utility>
#include "date/date.h"
#include "utl/to_vec.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/mcraptor/mcraptor.h"

namespace nigiri::routing {

routing_result mcraptor_search(
    timetable const& tt,
    rt_timetable const* rtt,
    search_state& s_state,
    raptor_state& r_state,
    query q,
    direction const search_dir,
    std::optional<std::chrono::seconds> const timeout) {
  if (rtt != nullptr) {
    return {};
  }

  using algo_t =
      mcraptor<direction::kForward, false, 0, search_mode::kOneToOne>;
  return search<direction::kForward, algo_t>{
      tt, rtt, s_state,r_state, std::move(q), timeout}
        .execute();
}

}  // namespace nigiri::routing