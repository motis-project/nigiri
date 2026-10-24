#pragma once

#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::rt {

void resolve_conflicts(timetable const&, rt_timetable&);

inline location_idx_t base(timetable const& tt,
                           rt_timetable const* rtt,
                           location_idx_t const l) {
  return l == location_idx_t::invalid() ? l
         : rtt == nullptr               ? tt.base(l)
                                        : rtt->base(l);
}

}  // namespace nigiri::rt
