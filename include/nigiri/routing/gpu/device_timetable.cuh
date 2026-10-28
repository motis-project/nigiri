#pragma once

#include <cinttypes>

#include <cuda/std/array>
#include <cuda/std/span>

#include "nigiri/common/delta_t.h"
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

// A route's transports and its compressed stop times: the event times of a
// transport are its first departure plus a per-event "time since first
// departure". Transports with identical offsets share one pattern; most routes
// have one.
// A pattern index within its route (see route_times).
using route_pattern_idx_t =
    cista::strong<std::uint16_t, struct _route_pattern_idx>;

struct alignas(16) route_times {
  interval<transport_idx_t> transports_;
  std::uint32_t pattern_start_;  // first entry of the route in patterns_
  std::uint16_t n_patterns_;  // distinct offset sequences in the route
};

// first departure + time since first departure
__device__ __forceinline__ delta add_offset(delta const first_dep,
                                            delta const off) {
  auto mam = static_cast<std::uint16_t>(first_dep.mam() + off.mam());
  auto days = static_cast<std::uint16_t>(first_dep.days() + off.days());
  if (mam >= 1440U) {
    mam -= 1440U;
    ++days;
  }
  return {days, mam};
}

// The event times of all transports of one route at one event.
struct event_column {
  __device__ __forceinline__ delta operator[](unsigned const j) const {
    return add_offset(first_dep_[j], n_patterns_ == 1U
                                         ? off_
                                         : patterns_[to_idx(pattern_idx_[j])]);
  }

  __device__ __forceinline__ unsigned size() const { return n_transports_; }

  __device__ __forceinline__ transport_idx_t transport(unsigned const j) const {
    return transport_idx_t{to_idx(first_transport_) + j};
  }

  transport_idx_t first_transport_;
  delta const* first_dep_;  // of the route's transports
  route_pattern_idx_t const* pattern_idx_;  // of the route's transports
  delta const* patterns_;  // the event's offsets, one per pattern
  delta off_;  // the event's offset of the first pattern
  unsigned n_patterns_;
  unsigned n_transports_;
};

struct device_timetable {
  using t = timetable;

  __device__ __forceinline__ static unsigned event_idx(
      stop_idx_t const stop_idx, event_type const ev_type) {
    return static_cast<unsigned>(stop_idx * 2 -
                                 (ev_type == event_type::kArr ? 1 : 0));
  }

  __device__ event_column event_times_at_stop(route_idx_t const r,
                                              stop_idx_t const stop_idx,
                                              event_type const ev_type) const {
    auto const rt = route_times_[r];
    auto const* patterns =
        &patterns_[rt.pattern_start_ +
                   event_idx(stop_idx, ev_type) * rt.n_patterns_];
    return {.first_transport_ = rt.transports_.from_,
            .first_dep_ = &first_dep_[rt.transports_.from_],
            .pattern_idx_ = &transport_pattern_[rt.transports_.from_],
            .patterns_ = patterns,
            .off_ = *patterns,
            .n_patterns_ = rt.n_patterns_,
            .n_transports_ = static_cast<unsigned>(rt.transports_.size())};
  }

  __device__ delta event_mam(route_idx_t const r,
                             transport_idx_t t,
                             stop_idx_t const stop_idx,
                             event_type const ev_type) const {
    auto const rt = route_times_[r];
    return add_offset(
        first_dep_[t],
        patterns_[rt.pattern_start_ +
                  event_idx(stop_idx, ev_type) * rt.n_patterns_ +
                  (rt.n_patterns_ == 1U ? 0U : to_idx(transport_pattern_[t]))]);
  }

  __device__ interval<date::sys_days> internal_interval_days() const {
    return internal_interval_days_;
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

  // Transports + compressed route stop times, see route_times.
  d_vecmap_view<transport_idx_t, delta> first_dep_;
  d_vecmap_view<transport_idx_t, route_pattern_idx_t> transport_pattern_;
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