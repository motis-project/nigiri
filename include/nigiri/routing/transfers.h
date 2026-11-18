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

// All footpaths of l ([unrestricted | restricted] + real-time), unfiltered:
// matches queries with default transfer time settings.
template <direction SearchDir>
auto get_transfers(timetable const& tt,
                   rt_timetable const* rtt,
                   profile_idx_t const prf_idx,
                   location_idx_t const l) {
  constexpr auto const kFwd = SearchDir == direction::kForward;
  auto const prf = tt.locations_.footpath_profile(prf_idx);
  auto transfers = std::array<std::span<footpath const>, 2>{};
  auto const& fps = kFwd ? tt.locations_.footpaths_out_[prf]
                         : tt.locations_.footpaths_in_[prf];
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

// Calls fn(footpath raw, duration_t adjusted) for each footpath of l that the
// query's settings allow, adjusted by the settings.
template <direction SearchDir, typename Fn>
void for_each_transfer(timetable const& tt,
                       rt_timetable const* rtt,
                       profile_idx_t const prf_idx,
                       transfer_time_settings const& s,
                       location_idx_t const l,
                       Fn&& fn) {
  constexpr auto const kFwd = SearchDir == direction::kForward;
  auto const& loc = tt.locations_;
  auto const prf = loc.footpath_profile(prf_idx);
  auto const& fps = kFwd ? loc.footpaths_out_[prf] : loc.footpaths_in_[prf];
  if (l < fps.size()) {
    auto const b = fps[l];
    auto const split = loc.restricted_begin(prf, kFwd, l);
    for (auto i = 0U; i != split; ++i) {
      fn(b[i], adjusted_transfer_time(s, b[i].duration()));
    }
    if (split != b.size()) {
      auto const& records = kFwd ? loc.footpath_restrictions_out_[prf]
                                 : loc.footpath_restrictions_in_[prf];
      auto const r = records[l];
      auto const records_start = records.bucket_starts_[to_idx(l)];
      for (auto i = split; i != b.size(); ++i) {
        if (rtt != nullptr &&
            rtt->is_blocked(prf, kFwd, records_start + (i - split))) {
          continue;
        }
        auto const d =
            adjusted_transfer_time(s, b[i].duration().count(), r[i - split]);
        if (d >= 0) {
          fn(b[i], duration_t{d});
        }
      }
    }
  }
  if (rtt != nullptr && profile_uses_rt_virts(prf_idx)) {
    auto const b = (kFwd ? rtt->rt_footpaths_out_ : rtt->rt_footpaths_in_)[l];
    auto const r = (kFwd ? rtt->rt_footpath_restrictions_out_
                         : rtt->rt_footpath_restrictions_in_)[l];
    for (auto i = 0U; i != b.size(); ++i) {
      auto const d = adjusted_transfer_time(s, b[i].duration().count(), r[i]);
      if (d >= 0) {
        fn(b[i], duration_t{d});
      }
    }
  }
}

template <typename Fn>
void for_each_transfer(timetable const& tt,
                       rt_timetable const* rtt,
                       profile_idx_t const prf_idx,
                       transfer_time_settings const& s,
                       direction const dir,
                       location_idx_t const l,
                       Fn&& fn) {
  if (dir == direction::kForward) {
    for_each_transfer<direction::kForward>(tt, rtt, prf_idx, s, l, fn);
  } else {
    for_each_transfer<direction::kBackward>(tt, rtt, prf_idx, s, l, fn);
  }
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

// Fastest static footpath from -> to that the settings allow:
// {adjusted duration, raw duration}.
std::optional<std::pair<duration_t, duration_t>> shortest_transfer(
    timetable const&,
    rt_timetable const*,
    profile_idx_t,
    transfer_time_settings const&,
    location_idx_t from,
    location_idx_t to);

// Restriction record of the fastest station graph footpath from -> to that
// the settings allow (nullopt: no such footpath).
std::optional<footpath_restriction> get_footpath_restriction(
    timetable const&,
    profile_idx_t,
    transfer_time_settings const&,
    location_idx_t from,
    location_idx_t to,
    rt_timetable const* = nullptr);

inline bool is_td_source(rt_timetable const* rtt,
                         profile_idx_t const prf_idx,
                         location_idx_t const l) {
  if (rtt == nullptr) {
    return false;
  }
  auto const prf = rtt->tt_ == nullptr
                       ? prf_idx
                       : rtt->tt_->locations_.footpath_profile(prf_idx);
  return has_profile_td_footpaths(prf) &&
         rtt->has_td_footpaths_out_[prf].test(l);
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
