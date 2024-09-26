#include "nigiri/routing/meat/csa/meat_csa_state.h"

namespace nigiri::routing::meat::csa {

template <typename ProfileSet>
meat_csa_state<ProfileSet>& meat_csa_state<ProfileSet>::prepare_for_tt(
    timetable const& tt) {
  ea_.resize(tt.n_locations());
  profile_set_.prepare_for_tt(tt);

  first_con_reachable_.resize(tt.n_transports());
  for (auto t_idx = transport_idx_t{0}; t_idx < tt.n_transports(); ++t_idx) {
    auto const t_size = tt.travel_duration_days_[t_idx];
    first_con_reachable_[t_idx].resize(t_size);
  }

  reset();

  return *this;
}

template struct meat_csa_state<static_profile_set>;
template struct meat_csa_state<dynamic_profile_set>;
template struct meat_csa_state<dynamic_growth_profile_set>;

}  // namespace nigiri::routing::meat::csa
