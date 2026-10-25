#pragma once

#include <cassert>
#include <type_traits>

#include "nigiri/routing/journey.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/stop.h"
#include "nigiri/types.h"

namespace nigiri::routing {

template <typename Vec>
void extend_to_rt_virts(rt_timetable const& rtt, Vec& v) {
  if (v.size() == 0U) {
    return;
  }
  assert(v.size() >= rtt.tt_->n_locations());
  v.resize(rtt.n_locations());
  for (auto i = rt_location_idx_t{0U}; i != rtt.n_rt_locations(); ++i) {
    auto const l = to_idx(rtt.to_location(i));
    auto const b = to_idx(rtt.rt_location_parents_[i]);
    if constexpr (std::is_same_v<Vec, bitvec>) {
      v.set(l, v.test(b));
    } else {
      v[l] = v[b];
    }
  }
}

// Real-time virtual locations are for the default profile only: only it has
// the real-time walks that connect them.
inline location_idx_t search_location(rt_timetable const& rtt,
                                      profile_idx_t const prf,
                                      rt_transport_idx_t const rt_t,
                                      stop_idx_t const stop_idx) {
  return prf == kDefaultProfile
             ? rtt.stop_location(rt_t, stop_idx)
             : stop{rtt.rt_transport_location_seq_[rt_t][stop_idx]}
                   .location_idx();
}

inline location_idx_t search_location(profile_idx_t const prf,
                                      rt::run_stop const& stp) {
  return prf == kDefaultProfile && stp.fr_->is_rt() && stp.rtt() != nullptr
             ? stp.rtt()->stop_location(stp.fr_->rt_, stp.stop_idx_)
             : stp.get_stop().location_idx();
}

inline location_idx_t search_location(timetable const& tt,
                                      rt_timetable const* rtt,
                                      profile_idx_t const prf,
                                      journey::leg const& l,
                                      bool const is_exit) {
  auto const& ree = std::get<journey::run_enter_exit>(l.uses_);
  return search_location(
      prf,
      rt::frun{
          tt, rtt,
          ree.r_}[is_exit ? static_cast<stop_idx_t>(ree.stop_range_.to_ - 1U)
                          : ree.stop_range_.from_]);
}

}  // namespace nigiri::routing
