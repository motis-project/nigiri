#pragma once

#include "nigiri/rt/rt_timetable.h"
#include "nigiri/stop.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::rt {

// transfers.txt rules under real-time stop changes.
//
// The loader binds route-/trip-qualified rules to trip stops by moving them
// to virtual locations (location_type::kVirt) below their platform. A trip
// that changes platform therefore cannot simply be moved to the new platform:
//   - rules that name the trip (or its route) at station level still apply,
//   - rules that name the old platform do not apply anymore,
//   - rules that name the new platform start to apply.
//
// route_stop_at() recomputes which rule sides match the trip stop at the new
// platform - the same matching the loader does - and routes the stop at
//   1. the platform itself, if no qualified rule matches,
//   2. an existing virtual location of that platform that states exactly the
//      same (same key as in the loader), if there is one,
//   3. a new real-time virtual location otherwise (rt_timetable::rt_virts_),
//      which gets the transfers the loader would have given it: the platform's
//      own transfers as the default, overridden by the most specific rule per
//      partner. Transfers between existing locations never change, so nothing
//      static has to be masked or recomputed.
//
// `l` is the platform the feed names - or, to take a stop change back, the
// scheduled entry of the stop sequence, which may be a virtual location and is
// then used as it is. Returns the location the stop is routed at.
location_idx_t route_stop_at(timetable const&,
                             rt_timetable&,
                             rt_transport_idx_t,
                             stop_idx_t,
                             location_idx_t l);

// The platform of a (possibly virtual) location from the static timetable.
inline location_idx_t platform_of(timetable const& tt, location_idx_t const l) {
  return tt.locations_.get_base_idx(l);
}

// ... of any routing location: a real-time virtual location as well.
inline location_idx_t platform_of(timetable const& tt,
                                  rt_timetable const* rtt,
                                  location_idx_t const l) {
  return platform_of(tt, rtt == nullptr ? l : rtt->physical(l));
}

}  // namespace nigiri::rt
