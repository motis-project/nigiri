#pragma once

#include <cuda/std/span>
#include <cuda_runtime.h>
#include <cinttypes>
#include "gpu_types.h"
extern "C"{
  struct gpu_timetable;
  void destroy_gpu_timetable(gpu_timetable* gtt);

  struct gpu_timetable {
    gpu_delta* route_stop_times_{nullptr};
    gpu_vecvec<gpu_route_idx_t,gpu_value_type,unsigned int> * route_location_seq_ {nullptr};
    gpu_vecvec<gpu_location_idx_t,gpu_route_idx_t,unsigned int>* location_routes_ {nullptr};
    std::uint32_t n_locations_;
    std::uint32_t n_routes_;
    gpu_vector_map<gpu_route_idx_t,gpu_interval<std::uint32_t>>* route_stop_time_ranges_{nullptr};
    gpu_vector_map<gpu_route_idx_t,gpu_interval<gpu_transport_idx_t >>* route_transport_ranges_{nullptr};
    gpu_vector_map<gpu_bitfield_idx_t, gpu_bitfield>* bitfields_{nullptr};
    gpu_vector_map<gpu_transport_idx_t,gpu_bitfield_idx_t>* transport_traffic_days_{nullptr};
    gpu_interval<gpu_sys_days>* date_range_{nullptr};
    gpu_interval<gpu_sys_days> const* cpu_date_range_{nullptr};
    gpu_locations locations_;
    gpu_vector_map<gpu_route_idx_t, gpu_clasz>* route_clasz_{nullptr};

    gpu_interval<gpu_sys_days> cpu_internal_interval_days() const {
      auto date_range = *cpu_date_range_;
      return {date_range.from_ - (gpu_days{1} + gpu_days{4}),
              date_range.to_ + gpu_days{1}};
    }
    ~gpu_timetable() {
      destroy_gpu_timetable(this);
    }

  };

  struct gpu_timetable* create_gpu_timetable(gpu_delta const* route_stop_times,
                                             std::uint32_t n_route_stop_times,
                                             gpu_vecvec<gpu_route_idx_t,gpu_value_type> const* route_location_seq,
                                             gpu_vecvec<gpu_location_idx_t , gpu_route_idx_t> const* location_routes,
                                             std::uint32_t const n_locations,
                                             std::uint32_t const n_routes,
                                             gpu_vector_map<gpu_route_idx_t,gpu_interval<std::uint32_t>> const* route_stop_time_ranges,
                                             gpu_vector_map<gpu_route_idx_t,gpu_interval<gpu_transport_idx_t >> const* route_transport_ranges,
                                             gpu_vector_map<gpu_bitfield_idx_t, gpu_bitfield> const* bitfields,
                                             gpu_vector_map<gpu_transport_idx_t,gpu_bitfield_idx_t> const* transport_traffic_days,
                                             gpu_interval<gpu_sys_days> const* date_range,
                                             gpu_locations const* locations,
                                             gpu_vector_map<gpu_route_idx_t, gpu_clasz> const* route_clasz);
  } //extern "C"