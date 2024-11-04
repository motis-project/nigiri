#include "nigiri/routing/meat/raptor/meat_raptor_state.h"

namespace nigiri::routing::meat::raptor {

meat_raptor_state& meat_raptor_state::prepare_for_tt(timetable const& tt) {
  profile_set_.prepare_for_tt(tt);
  r_state_.resize(tt.n_locations(), tt.n_routes(), 0U);
  station_mark_.resize(tt.n_locations());
  prev_station_mark_.resize(tt.n_locations());
  route_mark_.resize(tt.n_routes());
  fp_dis_to_target_.resize(tt.n_locations());
  reset();

  return *this;
}

}  // namespace nigiri::routing::meat::raptor
