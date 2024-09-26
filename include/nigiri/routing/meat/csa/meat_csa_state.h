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
    for (auto& v : first_con_reachable_) {
      utl::fill(v, std::numeric_limits<c_idx_t>::max());
    }
    profile_set_.reset();
  }

  vector_map<location_idx_t, delta_t> ea_;
  vector_map<transport_idx_t, std::vector<c_idx_t>> first_con_reachable_;
  ProfileSet profile_set_;
  // TODO
  // add trip_ , trip_reset from meat_profile_computer?
  // add to_node_id_ from decision_graph_extractor ?
};

}  // namespace nigiri::routing::meat::csa
