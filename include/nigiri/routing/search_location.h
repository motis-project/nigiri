#pragma once

#include "nigiri/rt/frun.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/stop.h"
#include "nigiri/types.h"

namespace nigiri::routing {

inline location_idx_t search_location(rt_timetable const& rtt,
                                      profile_idx_t const prf,
                                      rt_transport_idx_t const rt_t,
                                      stop_idx_t const stop_idx) {
  return is_projected(prf)
             ? stop{rtt.rt_transport_location_seq_[rt_t][stop_idx]}
                   .location_idx()
             : rtt.stop_location(rt_t, stop_idx);
}

inline location_idx_t search_location(profile_idx_t const prf,
                                      rt::run_stop const& stp) {
  auto const& fr = *stp.fr_;
  return fr.is_rt() && fr.rtt_ != nullptr
             ? search_location(*fr.rtt_, prf, fr.rt_, stp.stop_idx_)
             : stp.get_location_idx();
}

}  // namespace nigiri::routing
