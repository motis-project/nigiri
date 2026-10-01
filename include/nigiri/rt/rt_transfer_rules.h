#pragma once

#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::rt {

location_idx_t get_or_create_location(timetable const&,
                                      rt_timetable&,
                                      rt_transport_idx_t,
                                      stop_idx_t,
                                      location_idx_t base);

inline location_idx_t get_base_idx(timetable const& tt,
                                   rt_timetable const* rtt,
                                   location_idx_t const l) {
  return tt.locations_.get_base_idx(rtt == nullptr ? l : rtt->get_base_idx(l));
}

}  // namespace nigiri::rt
