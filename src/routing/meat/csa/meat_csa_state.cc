#include "nigiri/routing/meat/csa/meat_csa_state.h"

namespace nigiri::routing::meat::csa {

template <typename ProfileSet>
meat_csa_state<ProfileSet>& meat_csa_state<ProfileSet>::prepare_for_tt(timetable const& tt) {
  ea_.resize(tt.n_locations());
  trip_first_con_.resize(tt.n_transports());
  profile_set_.prepare_for_tt(tt);

  reset();

  return *this;
}

template struct meat_csa_state<static_profile_set>;
template struct meat_csa_state<dynamic_profile_set>;
template struct meat_csa_state<dynamic_growth_profile_set>;

}  // namespace nigiri::routing::meat::csa
