#pragma once

#include <cuda/std/span>

#include "nigiri/common/delta_t.h"
#include "nigiri/types.h"

namespace nigiri::routing::gpu {

template <direction SearchDir, via_offset_t Vias>
struct device_times {
  __device__ delta_t get(std::uint8_t const k,
                         location_idx_t const l,
                         via_offset_t const via) {
    return data_[internal_idx(k, l, via)];
  }

  __device__ delta_t get(location_idx_t const l, via_offset_t const via) {
    return data_[internal_idx(0U, l, via)];
  }

  __device__ delta_t get(std::uint8_t const i) { return data_[i]; }

  __device__ bool update_min(std::uint8_t const k,
                             location_idx_t const l,
                             via_offset_t const via,
                             delta_t const val) {
    return update_min(internal_idx(k, l, via), val);
  }

  __device__ bool update_min(location_idx_t const l,
                             via_offset_t const via,
                             delta_t const val) {
    return update_min(internal_idx(0U, l, via), val);
  }

  __device__ bool update_min(std::size_t const idx, delta_t const val) {
    delta_t* const arr_address = &data_[idx];
    auto* base_address = (int*)((size_t)arr_address & ~2);
    std::int32_t old_value, new_value;
    do {
      old_value = atomicCAS(base_address, *base_address, *base_address);
      if ((size_t)arr_address & 2) {
        std::int32_t old_upper = (old_value >> 16) & 0xFFFF;
        old_upper = (old_upper << 16) >> 16;
        std::int32_t new_upper = get_best(old_upper, val);
        if (new_upper == old_upper) {
          return false;
        }
        new_value = (old_value & 0x0000FFFF) | (new_upper << 16);
      } else {
        std::int32_t old_lower = old_value & 0xFFFF;
        old_lower = (old_lower << 16) >> 16;
        std::int32_t new_lower = get_best(old_lower, val);
        if (new_lower == old_lower) {
          return false;
        }
        new_value = (old_value & 0xFFFF0000) | (new_lower & 0xFFFF);
      }
    } while (atomicCAS(base_address, old_value, new_value) != old_value);
    return true;
  }

  __device__ __forceinline__ unsigned internal_idx(std::uint8_t const k,
                                                   location_idx_t const l,
                                                   via_offset_t const via) {
    return (k * n_locations_ * Vias) + (l.v_ * Vias) + via;
  }

  __device__ __forceinline__ bool is_better(auto a, auto b) {
    return (SearchDir == direction::kForward) ? a < b : a > b;
  }

  __device__ __forceinline__ auto get_best(auto a, auto b) {
    return is_better(a, b) ? a : b;
  }

  cuda::std::span<delta_t> data_;
  std::uint32_t n_locations_;
};

}  // namespace nigiri::routing::gpu