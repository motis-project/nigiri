#pragma once

#include "nigiri/location_match_mode.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

inline rt_timetable const* rt_virts(rt_timetable const* rtt,
                                    profile_idx_t const prf) {
  return prf == kDefaultProfile ? rtt : nullptr;
}

template <typename Fn>
void for_each_meta(timetable const& tt,
                   rt_timetable const* rtt,
                   location_match_mode const mode,
                   location_idx_t const l,
                   Fn&& fn) {
  if (to_idx(l) >= tt.n_locations()) {
    fn(l);
    return;
  }
  auto const each = [&](location_idx_t const x) {
    fn(x);
    if (rtt != nullptr) {
      rtt->for_each_rt_virt(x, fn);
    }
  };
  auto const handle_children = [&](location_idx_t const loc) {
    for (auto const c : tt.locations_.children_.at(loc)) {
      each(c);
      for (auto const cc : tt.locations_.children_.at(c)) {
        each(cc);
      }
    }
  };
  auto const handle_equivalences = [&](location_idx_t const loc) {
    for (auto const eq :
         tt.locations_.equivalences_.at(tt.locations_.get_base_idx(loc))) {
      each(eq);
      handle_children(eq);
    }
  };

  switch (mode) {
    case location_match_mode::kExact:
      each(l);
      tt.locations_.for_each_virt(l, fn);
      break;
    case location_match_mode::kIntermodal:
      each(l);
      for (auto const c : tt.locations_.children_.at(l)) {
        if (tt.locations_.types_.at(c) == location_type::kGeneratedTrack ||
            tt.locations_.types_.at(c) == location_type::kVirt) {
          each(c);
        }
      }
      break;
    case location_match_mode::kOnlyChildren:
      each(l);
      handle_children(l);
      break;
    case location_match_mode::kEquivalent: {
      auto const root = tt.locations_.get_root_idx(l);
      each(root);
      handle_children(root);
      handle_equivalences(l);
      break;
    }
  }
}

inline bool matches(timetable const& tt,
                    rt_timetable const* rtt,
                    location_match_mode const mode,
                    location_idx_t const a,
                    location_idx_t const b) {
  if (a == b) {
    return true;
  }
  auto is_match = false;
  for_each_meta(tt, rtt, mode, a, [&](location_idx_t const candidate) {
    is_match = is_match || candidate == b;
  });
  return is_match;
}

}  // namespace nigiri::routing
