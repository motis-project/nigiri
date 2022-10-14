#pragma once

#include "nigiri/routing/location_match_mode.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

template <typename Fn>
void for_each_meta(timetable const& tt,
                   location_match_mode const mode,
                   location_idx_t const l,
                   Fn&& fn) {
  if (mode == location_match_mode::kExact) {
    fn(l);
  } else if (mode == location_match_mode::kOnlyChildren) {
    fn(l);
    for (auto const& c : tt.locations_.children_.at(l)) {
      fn(c);
    }
  } else if (mode == location_match_mode::kEquivalent) {
    fn(l);
    for (auto const& c : tt.locations_.children_.at(l)) {
      fn(c);
    }
    for (auto const& eq : tt.locations_.equivalences_.at(l)) {
      fn(eq);
      for (auto const& c : tt.locations_.children_.at(eq)) {
        fn(c);
      }
    }
  }
}

}  // namespace nigiri::routing
