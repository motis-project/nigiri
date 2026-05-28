#pragma once

#include "nigiri/location_match_mode.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

template <typename Fn>
void for_each_meta(timetable const& tt,
                   location_match_mode const mode,
                   location_idx_t const l,
                   Fn&& fn) {
  auto const handle_children = [&](auto const& loc) {
    for (auto const& c : tt.locations_.children_.at(loc)) {
      fn(c);
      for (auto const& cc : tt.locations_.children_.at(c)) {
        fn(cc);
      }
    }
  };
  auto const handle_equivalences = [&](auto const& loc) {
    for (auto const& eq : tt.locations_.equivalences_.at(loc)) {
      fn(eq);
      handle_children(eq);
    }
  };

  if (mode == location_match_mode::kExact) {
    fn(l);
  } else if (mode == location_match_mode::kIntermodal) {
    fn(l);
    for (auto const& c : tt.locations_.children_.at(l)) {
      if (tt.locations_.types_.at(c) == location_type::kGeneratedTrack) {
        fn(c);
      }
    }
  } else if (mode == location_match_mode::kOnlyChildren) {
    fn(l);
    handle_children(l);
  } else if (mode == location_match_mode::kEquivalent) {
    auto const root = tt.locations_.get_root_idx(l);
    fn(root);
    handle_children(root);
    handle_equivalences(l);
  }
}

inline bool matches(timetable const& tt,
                    location_match_mode const mode,
                    location_idx_t const a,
                    location_idx_t const b) {
  switch (mode) {
    case location_match_mode::kExact: return a == b;
    case location_match_mode::kIntermodal:
    case location_match_mode::kOnlyChildren: [[fallthrough]];
    case location_match_mode::kEquivalent:
      if (a == b) {
        return true;
      }

      {
        auto matches = false;
        for_each_meta(tt, mode, a, [&](location_idx_t const candidate) {
          matches = matches || (candidate == b);
        });
        return matches;
      }
  }
  return true;
}

}  // namespace nigiri::routing
