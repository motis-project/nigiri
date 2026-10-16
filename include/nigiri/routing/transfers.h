#pragma once

#include <array>
#include <optional>
#include <ranges>
#include <span>

#include "nigiri/footpath.h"
#include "nigiri/routing/search_location.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing {

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
auto get_transfers(timetable const& tt,
                   rt_timetable const* rtt,
                   profile_idx_t const prf_idx,
                   location_idx_t const l) {
  constexpr auto const kFwd = SearchDir == direction::kForward;
  auto transfers = std::array<std::span<footpath const>, 2>{};
  auto const& fps = kFwd ? tt.locations_.footpaths_out_[prf_idx]
                         : tt.locations_.footpaths_in_[prf_idx];
  if (l < fps.size()) {
    auto const b = fps[l];
    transfers[0] = {b.data(), b.size()};
  }
  if (rtt != nullptr && prf_idx == kDefaultProfile) {
    auto const& rt_fps = kFwd ? rtt->rt_footpaths_out_ : rtt->rt_footpaths_in_;
    if (l < rt_fps.size()) {
      auto const b = rt_fps[l];
      transfers[1] = {b.begin(), b.end()};
    }
  }
  return std::move(transfers) | std::views::join;
}

inline auto get_transfers(timetable const& tt,
                          rt_timetable const* rtt,
                          profile_idx_t const prf_idx,
                          direction const dir,
                          location_idx_t const l) {
  return dir == direction::kForward
             ? get_transfers<direction::kForward>(tt, rtt, prf_idx, l)
             : get_transfers<direction::kBackward>(tt, rtt, prf_idx, l);
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
  for (auto const fp :
       get_transfers<direction::kForward>(tt, rtt, prf_idx, from)) {
    if (fp.target() == to &&
        (!shortest.has_value() || fp.duration() < *shortest)) {
      shortest = fp.duration();
    }
  }
  return shortest;
}

}  // namespace nigiri::routing
