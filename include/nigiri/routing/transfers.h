#pragma once

#include <array>
#include <optional>
#include <ranges>
#include <span>

#include "nigiri/footpath.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/search_location.h"
#include "nigiri/routing/transfer_time_settings.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/special_stations.h"
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

std::optional<duration_t> shortest_transfer(timetable const&,
                                            rt_timetable const*,
                                            profile_idx_t,
                                            location_idx_t from,
                                            location_idx_t to);

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

std::optional<footpath_duration> get_footpath_duration(
    timetable const&,
    rt_timetable const*,
    profile_idx_t,
    transfer_time_settings const&,
    direction,
    unixtime_t,
    location_idx_t from,
    location_idx_t to);

std::optional<journey::leg> get_offset_leg(timetable const&,
                                           event_type,
                                           unixtime_t,
                                           location_idx_t,
                                           special_station,
                                           std::vector<offset> const&,
                                           td_offsets_t const&);

}  // namespace nigiri::routing
