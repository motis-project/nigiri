#pragma once

#include <optional>
#include <type_traits>

#include "nigiri/footpath.h"
#include "nigiri/routing/search_location.h"
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

inline u8_minutes get_transfer_time(timetable const& tt,
                                    rt_timetable const* rtt,
                                    location_idx_t const l) {
  return rtt != nullptr && rtt->is_rt_location(l)
             ? rtt->transfer_time(l)
             : tt.locations_.transfer_time_[l];
}

// The footpaths of a location, the real-time ones included (only the default
// profile has real-time virtual locations).
template <direction SearchDir>
void for_each_transfer(timetable const& tt,
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
        return;
      }
    }
  }
  if (rtt == nullptr || prf_idx != kDefaultProfile) {
    return;
  }
  auto const& rt_footpaths =
      kFwd ? rtt->rt_footpaths_out_ : rtt->rt_footpaths_in_;
  if (l < rt_footpaths.size()) {
    for (auto const& fp : rt_footpaths[l]) {
      if (!call_and_go_on(fn, fp)) {
        return;
      }
    }
  }
}

inline std::optional<duration_t> shortest_transfer(timetable const& tt,
                                                   rt_timetable const* rtt,
                                                   profile_idx_t const prf_idx,
                                                   location_idx_t const from,
                                                   location_idx_t const to) {
  auto shortest = std::optional<duration_t>{};
  if (from == to) {
    if (auto const transfer_time = get_transfer_time(tt, rtt, from);
        transfer_time != kNoTransferAllowed) {
      shortest = duration_t{transfer_time.count()};
    }
  }
  for_each_transfer<direction::kForward>(
      tt, rtt, prf_idx, from, [&](footpath const& fp) {
        if (fp.target() == to &&
            (!shortest.has_value() || fp.duration() < *shortest)) {
          shortest = fp.duration();
        }
      });
  return shortest;
}

void for_each_transfer(timetable const& tt,
                       rt_timetable const* rtt,
                       profile_idx_t const prf_idx,
                       direction const dir,
                       location_idx_t const l,
                       auto&& fn) {
  if (dir == direction::kForward) {
    for_each_transfer<direction::kForward>(tt, rtt, prf_idx, l, fn);
  } else {
    for_each_transfer<direction::kBackward>(tt, rtt, prf_idx, l, fn);
  }
}

}  // namespace nigiri::routing
