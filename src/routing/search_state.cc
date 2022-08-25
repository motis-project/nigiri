#include "nigiri/routing/search_state.h"

#include "nigiri/routing/limits.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

void search_state::reset(timetable const& tt, routing_time init) {
  station_mark_.resize(tt.n_locations());
  std::fill(begin(station_mark_), end(station_mark_), false);

  route_mark_.resize(tt.n_routes());
  std::fill(begin(route_mark_), end(route_mark_), false);

  best_.resize(tt.n_locations());
  std::fill(begin(best_), end(best_), init);

  round_times_.resize(kMaxTransfers + 1U, tt.n_locations());
  round_times_.reset(init);

  starts_.clear();
  results_.clear();
}

}  // namespace nigiri::routing
