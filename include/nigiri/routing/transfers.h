#pragma once

#include <array>
#include <optional>
#include <ranges>
#include <span>

#include "nigiri/footpath.h"
#include "nigiri/routing/search_location.h"
#include "nigiri/routing/transfer_time_settings.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/td_footpath.h"
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
  if (rtt != nullptr && profile_uses_rt_virts(prf_idx)) {
    auto const b = (kFwd ? rtt->rt_footpaths_out_ : rtt->rt_footpaths_in_)[l];
    transfers[1] = {b.data(), b.size()};
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

inline bool is_td_source(rt_timetable const* rtt,
                         profile_idx_t const prf_idx,
                         location_idx_t const l) {
  return rtt != nullptr && has_profile_td_footpaths(prf_idx) &&
         rtt->has_td_footpaths_out_[prf_idx].test(l);
}

struct footpath_duration {
  duration_t duration_;
  duration_t travel_time_;
  duration_t raw_;
};

inline std::optional<footpath_duration> get_footpath_duration(
    timetable const& tt,
    rt_timetable const* rtt,
    profile_idx_t const prf_idx,
    transfer_time_settings const& settings,
    direction const dir,
    unixtime_t const t,
    location_idx_t const from,
    location_idx_t const to) {
  if (from != to && is_td_source(rtt, prf_idx, from)) {
    auto best = std::optional<footpath_duration>{};
    for_each_footpath(
        dir, rtt->td_footpaths_out_[prf_idx][from], t,
        [&](location_idx_t const target, duration_t const d,
            duration_t const w) {
          if (target == to && (!best.has_value() || d < best->duration_)) {
            best = footpath_duration{d, w, w};
          }
        });
    return best;
  }
  auto const raw = shortest_transfer(tt, rtt, prf_idx, from, to);
  if (!raw.has_value()) {
    return std::nullopt;
  }
  auto const d = adjusted_transfer_time(settings, *raw);
  return footpath_duration{d, d, *raw};
}

}  // namespace nigiri::routing
