#pragma once

#include <type_traits>

#include "nigiri/footpath.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing {

template <typename Fn>
bool call_and_go_on(Fn&& fn, footpath const& fp) {
  if constexpr (std::is_void_v<decltype(fn(fp))>) {
    fn(fp);
    return true;
  } else {
    return fn(fp);
  }
}

inline u8_minutes own_change_time(timetable const& tt,
                                  rt_timetable const* rtt,
                                  profile_idx_t const prf,
                                  location_idx_t const l) {
  return rtt != nullptr && rtt->is_rt_virt(l)
             ? rtt->rt_locations_.transfer_time_[rtt->to_rt_location(l)]
             : tt.locations_.transfer_time_[tt.locations_.project(prf, l)];
}

template <direction SearchDir>
void for_each_hub_source(timetable const& tt,
                         profile_idx_t const prf_idx,
                         location_idx_t const l,
                         auto&& fn) {
  constexpr auto const kFwd = SearchDir == direction::kForward;
  auto const& by_loc = kFwd ? tt.locations_.hub_out_by_loc_[prf_idx]
                            : tt.locations_.hub_in_by_loc_[prf_idx];
  if (l >= by_loc.size()) {
    return;
  }
  for (auto const h : by_loc[l]) {
    auto const d = tt.locations_.hub_time_[prf_idx][h];
    for (auto const source : (kFwd ? tt.locations_.hub_in_[prf_idx]
                                   : tt.locations_.hub_out_[prf_idx])[h]) {
      if (source != l && !call_and_go_on(fn, footpath{source, d})) {
        return;
      }
    }
  }
}

template <direction SearchDir>
bool for_each_footpath_at(timetable const& tt,
                          rt_timetable const* rtt,
                          profile_idx_t const prf_idx,
                          location_idx_t const l,
                          auto&& fn) {
  constexpr auto const kFwd = SearchDir == direction::kForward;
  auto const& fps = kFwd ? tt.locations_.footpaths_out_[prf_idx]
                         : tt.locations_.footpaths_in_[prf_idx];
  if (l < fps.size()) {
    for (auto const& fp : fps[l]) {
      if (!call_and_go_on(fn, fp)) {
        return false;
      }
    }
  }
  if (rtt == nullptr || projects_virts(prf_idx)) {
    return true;
  }
  auto const& rt_footpaths =
      kFwd ? rtt->rt_footpaths_out_ : rtt->rt_footpaths_in_;
  if (l < rt_footpaths.size()) {
    for (auto const& fp : rt_footpaths[l]) {
      if (!call_and_go_on(fn, fp)) {
        return false;
      }
    }
  }
  return true;
}

template <direction SearchDir>
void for_each_transfer(timetable const& tt,
                       rt_timetable const* rtt,
                       profile_idx_t const prf_idx,
                       location_idx_t const l,
                       auto&& fn) {
  if (for_each_footpath_at<SearchDir>(tt, rtt, prf_idx, l, fn)) {
    for_each_hub_source<flip(SearchDir)>(tt, prf_idx, l, fn);
  }
}

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
