#pragma once

#include <cinttypes>

#include <cuda/std/span>

#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "nigiri/routing/gpu/device_bitvec.cuh"
#include "nigiri/routing/gpu/types.cuh"

namespace nigiri::routing::gpu {

struct device_timetable {
  using t = timetable;

  __device__ cuda::std::span<delta const> event_times_at_stop(
      route_idx_t const r,
      stop_idx_t const stop_idx,
      event_type const ev_type) const {
    auto const n_transports = to_idx(route_transport_ranges_[r].size());
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
    auto const n_transports = to_idx(range.size());
    auto const route_stop_begin = static_cast<unsigned>(
        route_stop_time_ranges_[r].from_ +
        n_transports * (stop_idx * 2 - (ev_type == event_type::kArr ? 1 : 0)));
    auto const t_idx_in_route = to_idx(t) - to_idx(range.from_);
    return route_stop_times_[route_stop_begin + t_idx_in_route];
  }

  __device__ interval<date::sys_days> internal_interval_days() const {
    return internal_interval_days_;
  }

  __device__ void print() {
    printf("n_locations=%u\n", n_locations_);
    printf("n_routes=%u\n", n_routes_);
    printf("transfer_time=%" PRIu64 "\n", transfer_time_.data_.size());
    printf("footpaths_out=%u\n", footpaths_out_.size());
    printf("footpaths_out=%u\n", footpaths_in_.size());
    printf("route_stop_times=%" PRIu64 "\n", route_stop_times_.size());
    printf("route_stop_time_ranges=%" PRIu64 "\n",
           route_stop_time_ranges_.data_.size());
    printf("route_transport_ranges=%" PRIu64 "\n",
           route_transport_ranges_.data_.size());
    printf("route_clasz=%" PRIu64 "\n", route_clasz_.data_.size());
    printf("route_bikes_allowed=%" PRIu64 "\n", route_bikes_allowed_.size());
    printf("route_bikes_allowed_per_section=%u\n",
           route_bikes_allowed_per_section_.size());
    printf("transport_traffic_days=%" PRIu64 "\n",
           transport_traffic_days_.data_.size());
    printf("bitfields=%" PRIu64 "\n", bitfields_.data_.size());
  }

  std::uint32_t n_locations_;
  std::uint32_t n_routes_;

  d_vecmap_view<location_idx_t, u8_minutes> transfer_time_;
  d_vecvec_view<decltype(t{}.locations_.footpaths_out_[0])> footpaths_out_;
  d_vecvec_view<decltype(t{}.locations_.footpaths_in_[0])> footpaths_in_;

  cuda::std::span<delta const> route_stop_times_;
  d_vecmap_view<route_idx_t, interval<std::uint32_t>> route_stop_time_ranges_;
  d_vecmap_view<route_idx_t, interval<transport_idx_t>> route_transport_ranges_;
  d_vecmap_view<route_idx_t, clasz> route_clasz_;
  device_bitvec<std::uint64_t const> route_bikes_allowed_;
  d_vecvec_view<decltype(t{}.route_bikes_allowed_per_section_)>
      route_bikes_allowed_per_section_;

  d_vecvec_view<decltype(t{}.route_location_seq_)> route_location_seq_;
  d_vecvec_view<decltype(t{}.location_routes_)> location_routes_;

  d_vecmap_view<transport_idx_t, bitfield_idx_t> transport_traffic_days_;
  d_vecmap_view<bitfield_idx_t, bitfield> bitfields_;

  interval<date::sys_days> internal_interval_days_;
};

}  // namespace nigiri::routing::gpu