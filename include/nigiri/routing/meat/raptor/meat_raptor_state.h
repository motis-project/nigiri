#pragma once

#include "utl/helpers/algorithm.h"

#include "nigiri/timetable.h"
#include "nigiri/routing/meat/raptor/profile.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/routing/search.h"

namespace nigiri::routing::meat::raptor {

struct meat_raptor_state {
  using c_idx_t = connection::trip_con_idx_t;

  meat_raptor_state() = default;
  meat_raptor_state(meat_raptor_state const&) = delete;
  meat_raptor_state& operator=(meat_raptor_state const&) = delete;
  meat_raptor_state(meat_raptor_state&&) = default;
  meat_raptor_state& operator=(meat_raptor_state&&) = default;
  ~meat_raptor_state() = default;

  meat_raptor_state& prepare_for_tt(timetable const& tt);
  void reset(){
    profile_set_.reset();
    station_mark_.zero_out();
    prev_station_mark_.zero_out();
    route_mark_.zero_out();
    utl::fill(fp_dis_to_target_, std::numeric_limits<meat_t>::infinity());
    utl::fill(latest_dep_added_last_round, std::numeric_limits<delta_t>::min());
    utl::fill(latest_dep_added_current_round, std::numeric_limits<delta_t>::min());
  }

  profile_set profile_set_;
  search_state s_state_;
  raptor_state r_state_;
  bitvec station_mark_;
  bitvec prev_station_mark_;
  bitvec route_mark_;
  vector_map<location_idx_t, meat_t> fp_dis_to_target_;
  vector_map<location_idx_t, delta_t> latest_dep_added_last_round;
  vector_map<location_idx_t, delta_t> latest_dep_added_current_round;
};

}  // namespace nigiri::routing::meat::raptor
