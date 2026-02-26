#include "nigiri/routing/lb_raptor.h"

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
    vector_map<location_idx_t, std::array<std::uint16_t, kMaxTransfers + 2>>&
        location_round_lb) {
  state.prev_station_mark_.resize(tt.n_locations());
  utl::fill(state.prev_station_mark_.blocks_, 0U);
  state.station_mark_.resize(tt.n_locations());
  utl::fill(state.station_mark_.blocks_, 0U);
  location_round_lb.resize(tt.n_locations());
  static constexpr auto kRoundLbInit = []() {
    auto ret = std::array<std::uint16_t, kMaxTransfers + 2>{};
    ret.fill(std::numeric_limits<std::uint16_t>::max());
    return ret;
  }();
  utl::fill(location_round_lb, kRoundLbInit);
}

}  // namespace nigiri::routing
