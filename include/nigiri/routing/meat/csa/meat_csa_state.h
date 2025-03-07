#pragma once

#include <memory>
#include <queue>

#include "utl/helpers/algorithm.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/connection.h"
#include "nigiri/routing/meat/csa/profile.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing::meat::csa {

struct trip_data {
  meat_t meat_;
  connection_idx_t exit_conn_;
};

template <typename ProfileSet>
struct meat_csa_state {
  using c_idx_t = connection::trip_con_idx_t;

  meat_csa_state() = default;
  meat_csa_state(meat_csa_state const&) = delete;
  meat_csa_state& operator=(meat_csa_state const&) = delete;
  meat_csa_state(meat_csa_state&&) = default;
  meat_csa_state& operator=(meat_csa_state&&) = default;
  ~meat_csa_state() = default;

  meat_csa_state<ProfileSet>& prepare_for_tt(timetable const& tt);
  void reset() {
    utl::fill(ea_, std::numeric_limits<delta_t>::max());
    utl::fill(first_con_reachable_.data_, std::numeric_limits<c_idx_t>::max());
    utl::fill(trip_.data_, trip_data{std::numeric_limits<meat_t>::infinity(),
                                     connection_idx_t::invalid()});
    profile_set_.reset();
  }

  vector_map<location_idx_t, delta_t> ea_;
  vecvec<transport_idx_t, c_idx_t> first_con_reachable_;
  ProfileSet profile_set_;
  vecvec<transport_idx_t, trip_data> trip_;
};

}  // namespace nigiri::routing::meat::csa
