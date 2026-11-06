#pragma once

#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
struct rt_timetable;
}  // namespace nigiri

namespace nigiri::routing {

struct n_to_all_result {
  min_label const& at(location_idx_t const l) const {
    return state_.min_[to_idx(l)];
  }

  raptor_state state_;

  interval<unixtime_t> searched_;
};

n_to_all_result raptor_n_to_all(timetable const&,
                                rt_timetable const*,
                                query q);

}  // namespace nigiri::routing