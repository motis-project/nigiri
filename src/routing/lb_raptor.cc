#include "nigiri/routing/lb_raptor.h"

#include "nigiri/for_each_meta.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

void lb_raptor(
    timetable const& tt,
    query const& q,
    vecvec<location_idx_t, footpath> const& lb_graph,
    bitvec_map<location_idx_t> const* has_rt,
    vecvec<location_idx_t, footpath> const* rt_lb_graph,
    raptor_state& state,
    vector_map<location_idx_t, std::array<std::uint16_t, kMaxTransfers + 2U>>&
        location_round_lb) {
  state.prev_station_mark_.resize(tt.n_locations());
  state.station_mark_.resize(tt.n_locations());
  utl::fill(state.station_mark_.blocks_, 0U);
  location_round_lb.resize(tt.n_locations());
  static constexpr auto kRoundLbInit = []() {
    auto ret = std::array<std::uint16_t, kMaxTransfers + 2>{};
    ret.fill(std::numeric_limits<std::uint16_t>::max());
    return ret;
  }();
  utl::fill(location_round_lb, kRoundLbInit);

  for (auto const& o : q.destination_) {
    for_each_meta(tt, q.dest_match_mode_, o.target_,
                  [&](location_idx_t const x) {
                    location_round_lb[x].fill(o.duration_.count());
                    state.station_mark_.set(to_idx(x), true);
                  });
  }

  for (auto k = 1U; k != kMaxTransfers + 2U; ++k) {
    std::swap(state.prev_station_mark_, state.station_mark_);
    utl::fill(state.station_mark_.blocks_, 0U);

    auto any_marked = false;
    state.prev_station_mark_.for_each_set_bit([&](std::uint64_t const i) {
      any_marked = true;
      // follow lb graph and/or rt lb graph from this station
      // for each edge
      // check if time of this station + edge duration leads to an improvement
      // for the next round if yes write improved time to location_round_lb mark
      // the station with the improved time for the next round
    });
    if (!any_marked) {
      return;
    }
  }
}
}  // namespace nigiri::routing
