#pragma once

#include <type_traits>

#include "nigiri/footpath.h"
#include "nigiri/rt/rt_timetable.h"
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
  if (to_idx(l) >= by_loc.size()) {
    return;  // no hubs, or a real-time virtual location (never a hub member)
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

// The footpaths a search in direction `SearchDir` relaxes at `l`: the static
// ones plus, for the default profile, the transfers from / to real-time virtual
// locations (rt_timetable::rt_fps_*). `l` may be such a location itself.
// `fn` may return false to stop.
template <direction SearchDir>
void for_each_footpath_at(timetable const& tt,
                          rt_timetable const* rtt,
                          profile_idx_t const prf_idx,
                          location_idx_t const l,
                          auto&& fn) {
  constexpr auto const kFwd = SearchDir == direction::kForward;
  auto const stop = [&](footpath const& fp) {
    if constexpr (std::is_void_v<decltype(fn(fp))>) {
      fn(fp);
      return false;
    } else {
      return !fn(fp);
    }
  };
  // not a real-time virtual location, and the profile has a footpath layer
  auto const& fps = kFwd ? tt.locations_.footpaths_out_[prf_idx]
                         : tt.locations_.footpaths_in_[prf_idx];
  if (to_idx(l) < fps.size()) {
    for (auto const& fp : fps[l]) {
      if (stop(fp)) {
        return;
      }
    }
  }
  if (rtt == nullptr || projects_virts(prf_idx)) {
    return;
  }
  auto const& rt_fps = kFwd ? rtt->rt_fps_out_ : rtt->rt_fps_in_;
  if (rt_fps.empty()) {
    return;
  }
  if (auto const it = rt_fps.find(l); it != end(rt_fps)) {
    for (auto const& fp : it->second) {
      if (stop(fp)) {
        return;
      }
    }
  }
}

// The whole transfer relation at `l` in the direction of a search: the
// footpaths, the transfers of real-time virtual locations and what the hubs
// hand out - forward the transfers leaving `l`, backward the ones reaching it.
// Everything that reads transfers pair by pair goes through here, so only this
// place knows about hubs. A target can come more than once (a footpath and a
// hub, or two hubs); a caller that looks up one pair takes the minimum. The
// RAPTOR footpath phase does not use it: it relaxes a hub as a whole
// (expand_hubs), not pair by pair. `fn` may return false to stop.
template <direction SearchDir>
void for_each_transfer(timetable const& tt,
                       rt_timetable const* rtt,
                       profile_idx_t const prf_idx,
                       location_idx_t const l,
                       auto&& fn) {
  auto go_on = true;
  auto const call = [&](footpath const fp) {
    if constexpr (std::is_void_v<decltype(fn(fp))>) {
      fn(fp);
    } else {
      go_on = fn(fp);
    }
    return go_on;
  };
  for_each_footpath_at<SearchDir>(tt, rtt, prf_idx, l, call);
  if (go_on) {
    for_each_hub_source<flip(SearchDir)>(tt, prf_idx, l, call);
  }
}

// ... with the direction known at runtime only
void for_each_transfer(direction const dir,
                       timetable const& tt,
                       rt_timetable const* rtt,
                       profile_idx_t const prf_idx,
                       location_idx_t const l,
                       auto&& fn) {
  if (dir == direction::kForward) {
    for_each_transfer<direction::kForward>(tt, rtt, prf_idx, l, fn);
  } else {
    for_each_transfer<direction::kBackward>(tt, rtt, prf_idx, l, fn);
  }
}

}  // namespace nigiri::routing
