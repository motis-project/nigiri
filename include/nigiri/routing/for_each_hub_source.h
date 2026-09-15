#pragma once

#include "nigiri/footpath.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing {

// The locations a search reaches `l` from through a hub (raptor expand_hubs):
// every source of every hub that has `l` in its target list, at the hub's
// weight. `fn` returns false to stop. Forward: hubs where l is an out-target,
// their in-members are the sources; backward the other way around.
template <direction SearchDir>
void for_each_hub_source(timetable const& tt,
                         profile_idx_t const prf_idx,
                         location_idx_t const l,
                         auto&& fn) {
  constexpr auto const kFwd = SearchDir == direction::kForward;
  auto const& by_loc = kFwd ? tt.locations_.hub_out_by_loc_[prf_idx]
                            : tt.locations_.hub_in_by_loc_[prf_idx];
  if (by_loc.size() == 0U) {
    return;
  }
  for (auto const h : by_loc[l]) {
    auto const d = tt.locations_.hub_time_[prf_idx][h];
    for (auto const source : (kFwd ? tt.locations_.hub_in_[prf_idx]
                                   : tt.locations_.hub_out_[prf_idx])[h]) {
      if (source != l && !fn(footpath{source, d})) {
        return;
      }
    }
  }
}

}  // namespace nigiri::routing
