#include "nigiri/routing/raptor_state.h"

#include "nigiri/routing/limits.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

void raptor_state::reset(timetable const& tt, routing_time init) {
  station_mark_.resize(tt.n_locations());
  utl::fill(station_mark_, false);

  prev_station_mark_.resize(tt.n_locations());
  utl::fill(prev_station_mark_, false);

  route_mark_.resize(tt.n_routes());
  utl::fill(route_mark_, false);

  best_.resize(tt.n_locations());
  utl::fill(best_, init);

  round_times_.resize(kMaxTransfers + 1U, tt.n_locations());
  round_times_.reset(init);
}

}  // namespace nigiri::routing
