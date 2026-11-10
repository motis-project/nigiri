#pragma once

#include <cinttypes>

// hide date.h's NOEXCEPT from CCCL's token pasting, see device_times.h
#pragma push_macro("NOEXCEPT")
#undef NOEXCEPT
#include <cuda/std/array>
#include <cuda/std/span>
#pragma pop_macro("NOEXCEPT")

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/limits.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "nigiri/routing/gpu/device_bitvec.cuh"
#include "nigiri/routing/gpu/types.cuh"

namespace nigiri::routing::gpu {

template <typename Key>
struct device_transport_filter {
  __device__ __forceinline__ bool allows(std::uint32_t const i,
                                         unsigned const mode_bit,
                                         unsigned& section_mask) const {
    if (!allowed_[i * 2U] /* not allowed on ALL sections */) {
      if (!allowed_[i * 2U + 1U] /* not allowed on SOME sections */) {
        return false;
      }
      section_mask |= mode_bit;
    }
    return true;
  }

  device_bitvec<std::uint64_t const> allowed_;
  d_vecvec_view<vecvec<Key, bool>> sections_;
};

// query's required-modes bitmask (which modes need the per-section check)
inline constexpr auto kBikeSections = 1U;
inline constexpr auto kCarSections = 2U;
inline constexpr auto kWheelchairSections = 4U;
inline constexpr auto kNoCompulsoryReservationSections = 8U;

template <typename Key>
struct device_transport_filters {
  __device__ __forceinline__ bool section_killed(unsigned const section_mask,
                                                 Key const el,
                                                 unsigned const sec) const {
    if ((section_mask & kBikeSections) != 0U && !bike_.sections_[el][sec]) {
      return true;
    }
    if ((section_mask & kCarSections) != 0U && !car_.sections_[el][sec]) {
      return true;
    }
    if ((section_mask & kWheelchairSections) != 0U &&
        !wheelchair_.sections_[el][sec]) {
      return true;
    }
    if ((section_mask & kNoCompulsoryReservationSections) != 0U &&
        !reservation_not_required_.sections_[el][sec]) {
      return true;
    }
    return false;
  }

  device_transport_filter<Key> bike_;
  device_transport_filter<Key> car_;
  device_transport_filter<Key> wheelchair_;
  device_transport_filter<Key> reservation_not_required_;
};

struct device_rt_timetable {
  using rtt = rt_timetable;

  __device__ delta_t event_time(rt_transport_idx_t const rt_t,
                                stop_idx_t const stop_idx,
                                event_type const ev_type) const {
    auto const ev_idx = stop_idx * 2 - (ev_type == event_type::kArr ? 1 : 0);
    return rt_transport_stop_times_[rt_t][static_cast<unsigned>(ev_idx)];
  }

  std::uint32_t n_rt_transports_{0U};
  day_idx_t base_day_idx_{};

  d_vecvec_view<vecvec<location_idx_t, rt_transport_idx_t>>
      location_rt_transports_;
  d_vecvec_view<decltype(rtt{}.rt_transport_location_seq_)>
      rt_transport_location_seq_;
  d_vecvec_view<decltype(rtt{}.rt_transport_stop_times_)>
      rt_transport_stop_times_;
  d_vecmap_view<rt_transport_idx_t, clasz> rt_transport_clasz_;

  d_vecmap_view<transport_idx_t, bitfield_idx_t> transport_traffic_days_;
  d_vecmap_view<bitfield_idx_t, bitfield> bitfields_;

  // Only set if filters are used.
  // Hidden behind pointer to avoid transferring empty structs (perf impact).
  device_transport_filters<rt_transport_idx_t> const* filters_{nullptr};

  // Only set if:
  // - profile requires it
  // - and rt_timetable contains td_footpaths
  // Hidden behind pointer to avoid transferring empty structs (perf impact).
  struct td_footpaths {
    using fp_view = d_vecvec_view<decltype(rtt{}.td_footpaths_out_[0])>;
    cuda::std::array<fp_view, kNProfiles> out_;
    cuda::std::array<fp_view, kNProfiles> in_;
    cuda::std::array<device_bitvec<std::uint64_t const>, kNProfiles> has_out_;
    cuda::std::array<device_bitvec<std::uint64_t const>, kNProfiles> has_in_;
  };
  td_footpaths const* td_{nullptr};
};

using route_times = timetable::route_times;

// A trip of a route in 32 bits, [rel_day:6 | t_offset:26]: the traffic day
// relative to a search's day_lo, the transport relative to the route's first.
// Ordered as unsigned in scan order: backward stores kEtReverseBase - x, so a
// min() still yields the best trip (the later one).
inline constexpr auto kEtInvalid = ~std::uint32_t{0};
inline constexpr auto kEtRelDayShift = 26U;
inline constexpr auto kEtReverseBase = kEtInvalid - 1U;

struct device_timetable {
  using t = timetable;

  __device__ event_times event_times_at_stop(route_idx_t const r,
                                             stop_idx_t const stop_idx,
                                             event_type const ev_type) const {
    auto const rt = route_times_[r];
    return {
        .first_dep_ = &first_dep_[rt.transports_.from_],
        .pattern_ = &transport_pattern_[rt.transports_.from_],
        .offsets_ = &patterns_[rt.pattern_start_ +
                               event_idx(stop_idx, ev_type) * rt.n_patterns_],
        .first_transport_ = rt.transports_.from_,
        .n_transports_ = static_cast<std::uint32_t>(rt.transports_.size()),
        .n_patterns_ = rt.n_patterns_};
  }

  __device__ delta event_mam(route_idx_t const r,
                             transport_idx_t t,
                             stop_idx_t const stop_idx,
                             event_type const ev_type) const {
    auto const rt = route_times_[r];
    return first_dep_[t] +
           patterns_[rt.pattern_start_ +
                     event_idx(stop_idx, ev_type) * rt.n_patterns_ +
                     (rt.n_patterns_ == 1U ? 0U : transport_pattern_[t])];
  }

  __device__ interval<date::sys_days> internal_interval_days() const {
    return internal_interval_days_;
  }

  __device__ __forceinline__ bool is_route_active(route_idx_t const r,
                                                  day_idx_t const day) const {
    auto const d = static_cast<int>(day.v_);
    return d >= 0 &&
           bitfields_[route_traffic_days_[r]].test(static_cast<std::size_t>(d));
  }

  // kRtBitfieldFlag entries resolve into the rt timetable's bitfields (with
  // an rt timetable, transport_traffic_days_ is its copy)
  __device__ __forceinline__ bool is_transport_active(
      device_rt_timetable const& rtt,
      transport_idx_t const t,
      std::size_t const day) const {
    auto const i = to_idx(transport_traffic_days_[t]);
    return ((i & kRtBitfieldFlag) != 0U
                ? rtt.bitfields_[bitfield_idx_t{i & ~kRtBitfieldFlag}]
                : bitfields_[bitfield_idx_t{i}])
        .test(day);
  }

  // The first trip of r at stop_idx at or after day_at_stop / mam_at_stop
  // (backward: at or before) that runs, as CPU raptor. give_up(day, mam) of
  // a candidate event: no trip from there on can help (returns invalid).
  template <direction SearchDir>
  __device__ transport
  get_earliest_transport(device_rt_timetable const& rtt,
                         route_idx_t const r,
                         stop_idx_t const stop_idx,
                         day_idx_t const day_at_stop,
                         minutes_after_midnight_t const mam_at_stop,
                         auto&& give_up) const {
    constexpr auto kFwd = SearchDir == direction::kForward;
    auto const is_better = [](auto a, auto b) { return kFwd ? a < b : a > b; };
    auto const event_times = event_times_at_stop(
        r, stop_idx, kFwd ? event_type::kDep : event_type::kArr);
    auto const n = static_cast<unsigned>(event_times.size());

    // transport offset of the i-th event in search direction
    auto const t_offset_at = [&](unsigned const i) {
      return kFwd ? i : n - 1U - i;
    };

    // Day 0: the first event at or after the time at the stop (backward: at
    // or before). Routes are FIFO (loader get_index), so the column is sorted
    // by mam and a binary search finds it.
    auto const seek_first_day = [&]() {
      auto lo = 0U;
      auto hi = n;
      while (lo < hi) {
        auto const mid = lo + (hi - lo) / 2U;
        if (is_better(event_times[t_offset_at(mid)].mam(),
                      mam_at_stop.count())) {
          lo = mid + 1U;
        } else {
          hi = mid;
        }
      }
      return lo;
    };

    constexpr auto const kNDaysToIterate = static_cast<day_idx_t::value_t>(
        kMaxTravelTime / std::chrono::days{1} + 1U);
    for (auto i = day_idx_t::value_t{0U}; i != kNDaysToIterate; ++i) {
      auto const day = kFwd ? day_at_stop + i : day_at_stop - i;
      if (!is_route_active(r, day)) {
        continue;
      }
      for (auto pos = i == 0U ? seek_first_day() : 0U; pos < n; ++pos) {
        auto const t_offset = t_offset_at(pos);
        auto const ev = event_times[t_offset];
        auto const ev_mam = ev.mam();
        if (give_up(day, ev_mam)) {
          return {transport_idx_t::invalid(), day_idx_t::invalid()};
        }
        if (i == 0U && is_better(ev_mam, mam_at_stop.count())) {
          continue;
        }
        auto const t = event_times.transport(t_offset);
        auto const start_day = static_cast<int>(day.v_) - ev.days();
        if (!is_transport_active(rtt, t, static_cast<std::size_t>(start_day))) {
          continue;
        }
        return {t, static_cast<day_idx_t>(start_day)};
      }
    }
    return {};
  }

  template <direction SearchDir>
  __device__ __forceinline__ std::uint32_t pack_et(route_idx_t const r,
                                                   transport const t,
                                                   int const day_lo) const {
    if (!t.is_valid()) {
      return kEtInvalid;
    }
    auto const rel_day =
        static_cast<std::uint32_t>(static_cast<int>(t.day_.v_) - day_lo);
    auto const t_offset =
        to_idx(t.t_idx_) - to_idx(route_times_[r].transports_.from_);
    auto const x = (rel_day << kEtRelDayShift) | t_offset;
    return SearchDir == direction::kForward ? x : kEtReverseBase - x;
  }

  template <direction SearchDir>
  __device__ __forceinline__ transport unpack_et(route_idx_t const r,
                                                 std::uint32_t const p,
                                                 int const day_lo) const {
    auto const x = SearchDir == direction::kForward ? p : kEtReverseBase - p;
    auto const rel_day = x >> kEtRelDayShift;
    auto const t_offset = x & ((1U << kEtRelDayShift) - 1U);
    return transport{
        transport_idx_t{to_idx(route_times_[r].transports_.from_) + t_offset},
        day_idx_t{static_cast<day_idx_t::value_t>(static_cast<int>(rel_day) +
                                                  day_lo)}};
  }

  std::uint32_t n_locations_;
  std::uint32_t n_routes_;

  d_vecmap_view<location_idx_t, u8_minutes> transfer_time_;
  cuda::std::array<d_vecvec_view<decltype(t{}.locations_.footpaths_out_[0])>,
                   kNProfiles>
      footpaths_out_;
  cuda::std::array<d_vecvec_view<decltype(t{}.locations_.footpaths_in_[0])>,
                   kNProfiles>
      footpaths_in_;

  d_vecmap_view<transport_idx_t, delta> first_dep_;
  d_vecmap_view<transport_idx_t, std::uint16_t> transport_pattern_;
  d_vecmap_view<route_idx_t, route_times> route_times_;
  cuda::std::span<delta const> patterns_;  // per route: [event][pattern]

  d_vecmap_view<route_idx_t, clasz> route_clasz_;

  d_vecvec_view<decltype(t{}.route_location_seq_)> route_location_seq_;
  d_vecvec_view<decltype(t{}.location_routes_)> location_routes_;

  d_vecmap_view<transport_idx_t, bitfield_idx_t> transport_traffic_days_;
  d_vecmap_view<route_idx_t, bitfield_idx_t> route_traffic_days_;
  d_vecmap_view<transport_idx_t, route_idx_t> transport_route_;
  d_vecmap_view<bitfield_idx_t, bitfield> bitfields_;

  device_transport_filters<route_idx_t> const* filters_{nullptr};

  // Flat (route, stop) index space for the load-balanced compute_et
  // - route_stop_offset_[r] is route r's flat base.
  // - route_of_stop_[flat] maps a flat index back to a route.
  cuda::std::span<std::uint32_t const> route_stop_offset_;
  cuda::std::span<std::uint32_t const> route_of_stop_;

  interval<date::sys_days> internal_interval_days_;
};

}  // namespace nigiri::routing::gpu