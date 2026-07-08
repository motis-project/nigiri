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
    return false;
  }

  device_transport_filter<Key> bike_;
  device_transport_filter<Key> car_;
  device_transport_filter<Key> wheelchair_;
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

struct device_timetable {
  using t = timetable;

  __device__ cuda::std::span<delta const> event_times_at_stop(
      route_idx_t const r,
      stop_idx_t const stop_idx,
      event_type const ev_type) const {
    auto const n_transports =
        static_cast<unsigned>(route_transport_ranges_[r].size());
    auto const idx = static_cast<unsigned>(
        route_stop_time_ranges_[r].from_ +
        n_transports * (stop_idx * 2 - (ev_type == event_type::kArr ? 1 : 0)));
    return {&route_stop_times_[idx], n_transports};
  }

  __device__ delta event_mam(route_idx_t const r,
                             transport_idx_t t,
                             stop_idx_t const stop_idx,
                             event_type const ev_type) const {
    auto const range = route_transport_ranges_[r];
    auto const n_transports = static_cast<unsigned>(range.size());
    auto const route_stop_begin = static_cast<unsigned>(
        route_stop_time_ranges_[r].from_ +
        n_transports * (stop_idx * 2 - (ev_type == event_type::kArr ? 1 : 0)));
    auto const t_idx_in_route = to_idx(t) - to_idx(range.from_);
    return route_stop_times_[route_stop_begin + t_idx_in_route];
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

  cuda::std::span<delta const> route_stop_times_;
  d_vecmap_view<route_idx_t, interval<std::uint32_t>> route_stop_time_ranges_;
  d_vecmap_view<route_idx_t, interval<transport_idx_t>> route_transport_ranges_;
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