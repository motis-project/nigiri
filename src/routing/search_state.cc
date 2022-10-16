#include "nigiri/routing/search_state.h"

#include "nigiri/routing/limits.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

destination_comparator::destination_comparator(timetable const& tt) : tt_{tt} {}

bool destination_comparator::operator()(location_idx_t const a,
                                        location_idx_t const b) {
  auto const a_is_child =
      tt_.locations_.parents_[a] != location_idx_t::invalid() ? 1U : 0U;
  auto const b_is_child =
      tt_.locations_.parents_[b] != location_idx_t::invalid() ? 1U : 0U;
  return a_is_child > b_is_child;
}

void search_state::reset(timetable const& tt, routing_time init) {
  is_destination_.resize(tt.n_locations());
  std::fill(begin(is_destination_), end(is_destination_), false);

  station_mark_.resize(tt.n_locations());
  std::fill(begin(station_mark_), end(station_mark_), false);

  prev_station_mark_.resize(tt.n_locations());
  std::fill(begin(prev_station_mark_), end(prev_station_mark_), false);

  route_mark_.resize(tt.n_routes());
  std::fill(begin(route_mark_), end(route_mark_), false);

  best_.resize(tt.n_locations());
  std::fill(begin(best_), end(best_), init);

  round_times_.resize(kMaxTransfers + 1U, tt.n_locations());
  round_times_.reset(init);

  starts_.clear();
  destinations_.clear();
  results_.clear();
}

}  // namespace nigiri::routing
