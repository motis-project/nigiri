#pragma once

#include <cassert>
#include <type_traits>

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
    auto const b = to_idx(rtt.rt_locations_.parents_[i]);
    if constexpr (std::is_same_v<Vec, bitvec>) {
      v.set(l, v.test(b));
    } else {
      v[l] = v[b];
    }
  }
}

inline location_idx_t project(timetable const& tt,
                              profile_idx_t const prf,
                              location_idx_t const l) {
  return is_projected(prf) ? tt.base(l) : l;
}

inline location_idx_t search_location(rt_timetable const& rtt,
                                      profile_idx_t const prf,
                                      rt_transport_idx_t const rt_t,
                                      stop_idx_t const stop_idx) {
  return is_projected(prf)
             ? rtt.base(stop{rtt.rt_transport_location_seq_[rt_t][stop_idx]}
                            .location_idx())
             : rtt.stop_location(rt_t, stop_idx);
}

inline location_idx_t search_location(profile_idx_t const prf,
                                      rt::run_stop const& stp) {
  return is_projected(prf) ? stp.get_location_idx()
                           : stp.get_virt().value_or(stp.get_location_idx());
}

}  // namespace nigiri::routing
