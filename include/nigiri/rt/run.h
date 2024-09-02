#pragma once

#include <variant>

#include "nigiri/types.h"

namespace nigiri::rt {

// A run represents a single trip on a specific day. It abstracts over
// transport=(transport_idx_t, day_idx_t) and rt_transport_idx_t and provides a
// uniform interface to get information such as scheduled and real-time
// timestamps for stop times, the location sequence, etc. about this specific
// instance.
//
// Variations:
//
//  rt_transport_idx_t  |  transport  |  case
// =====================+=============+=====================================
//  invalid             |  valid      |  SCHEDULED transport, not changed
// ---------------------+-------------+-------------------------------------
//  valid               |  invalid    |  additional real-time transport
//                      |             |  NOT known from static timetable
// ---------------------+-------------+-------------------------------------
//  valid               |  valid      |  SCHEDULED transport with real-time
//                      |             |  update (delays, rerouting, etc.)
// ---------------------+-------------+-------------------------------------
//  invalid             | invalid     |  invalid / not found
struct run {
  bool is_rt() const noexcept { return rt_ != rt_transport_idx_t::invalid(); }
  bool is_scheduled() const noexcept { return t_.is_valid(); }
  bool valid() const noexcept {
    return t_.is_valid() || rt_ != rt_transport_idx_t::invalid();
  }

  // from static timetable, not set for additional services
  transport t_{transport::invalid()};
  interval<stop_idx_t> stop_range_;

  // real-time instance, not set if no real-time info available
  rt_transport_idx_t rt_{rt_transport_idx_t::invalid()};
};

}  // namespace nigiri::rt