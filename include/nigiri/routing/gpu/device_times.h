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
    assert(idx < data_.size());
    auto const addr = reinterpret_cast<std::uint32_t*>(&data_[idx]);
    auto const base_address = reinterpret_cast<std::uint32_t*>(
        reinterpret_cast<std::uintptr_t>(addr) &
        ~static_cast<std::uintptr_t>(2U));
    std::uint32_t old_value, new_value;
    do {
      old_value = *base_address;
      auto const is_in_upper_half = reinterpret_cast<std::uintptr_t>(addr) & 2U;
      if (is_in_upper_half) {
        auto const old_upper = static_cast<delta_t>(old_value >> 16U);
        auto const new_upper = get_best(old_upper, val);
        if (new_upper == old_upper) {
          return false;
        }
        new_value = (old_value & 0x0000FFFFU) | (new_upper << 16U);
      } else {
        auto const old_lower = static_cast<delta_t>(old_value & 0xFFFF);
        auto const new_lower = get_best(old_lower, val);
        if (new_lower == old_lower) {
          return false;
        }
        new_value = (old_value & 0xFFFF0000U) | new_lower;
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