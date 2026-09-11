#pragma once

#include <cuda/std/span>

#include "nigiri/common/delta_t.h"
#include "nigiri/types.h"

namespace nigiri::routing::gpu {

// Round-time storage: one 16-bit biased time key per cell, so that a
// smaller key always means a better arrival (in either search direction)
// and 0xFFFF is the invalid marker for both.
//
// There is no 16-bit atomicMin: update_min runs a compare-and-swap loop on
// the 32-bit word holding the cell (data_ is 4-byte aligned). Journeys are
// reconstructed by searching the timetable (see reconstruct_journey), so no
// breadcrumb is stored alongside the time.
template <direction SearchDir, via_offset_t Vias>
struct device_times {
  static constexpr bool kFwd = SearchDir == direction::kForward;

  CISTA_CUDA_COMPAT static std::uint16_t to_key(delta_t const t) {
    return kFwd ? static_cast<std::uint16_t>(static_cast<int>(t) + 32768)
                : static_cast<std::uint16_t>(32767 - static_cast<int>(t));
  }
  CISTA_CUDA_COMPAT static delta_t from_key(std::uint16_t const k) {
    return kFwd ? static_cast<delta_t>(static_cast<int>(k) - 32768)
                : static_cast<delta_t>(32767 - static_cast<int>(k));
  }
  CISTA_CUDA_COMPAT static std::uint16_t invalid_key() { return 0xFFFFU; }

  __device__ delta_t get(std::uint8_t const k,
                         location_idx_t const l,
                         via_offset_t const via) {
    return from_key(data_[internal_idx(k, l, via)]);
  }

  __device__ delta_t get(location_idx_t const l, via_offset_t const via) {
    return from_key(data_[internal_idx(0U, l, via)]);
  }

  __device__ delta_t get(std::uint8_t const i) { return from_key(data_[i]); }

  __device__ bool update_min(std::uint8_t const k,
                             location_idx_t const l,
                             via_offset_t const via,
                             delta_t const val) {
    return update_min(static_cast<std::size_t>(internal_idx(k, l, via)), val);
  }

  __device__ bool update_min(location_idx_t const l,
                             via_offset_t const via,
                             delta_t const val) {
    return update_min(static_cast<std::size_t>(internal_idx(0U, l, via)), val);
  }

  // returns true iff the cell was improved by this call
  __device__ bool update_min(std::size_t const idx, delta_t const val) {
    auto const key = static_cast<std::uint32_t>(to_key(val));
    auto const shift = static_cast<unsigned>(idx & 1U) * 16U;
    auto* const word = reinterpret_cast<unsigned int*>(  // NOLINT
        reinterpret_cast<std::uintptr_t>(data_.data() +
                                         (idx & ~std::size_t{1U})));
    auto old = *word;
    while (true) {
      if (((old >> shift) & 0xFFFFU) <= key) {
        return false;
      }
      auto const desired = (old & ~(0xFFFFU << shift)) | (key << shift);
      auto const seen = atomicCAS(word, old, desired);
      if (seen == old) {
        return true;
      }
      old = seen;
    }
  }

  __device__ __forceinline__ unsigned internal_idx(std::uint8_t const k,
                                                   location_idx_t const l,
                                                   via_offset_t const via) {
    return (k * n_locations_ * Vias) + (l.v_ * Vias) + via;
  }

  cuda::std::span<std::uint16_t> data_;
  std::uint32_t n_locations_;
};

}  // namespace nigiri::routing::gpu
