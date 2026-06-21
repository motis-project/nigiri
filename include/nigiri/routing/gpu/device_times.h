#pragma once

#include <cuda/std/span>

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/gpu/breadcrumb.h"
#include "nigiri/types.h"

namespace nigiri::routing::gpu {

// Packed round-time storage: each entry is a 64-bit word
//   [63:48] biased time key (16 bits, so unsigned atomicMin == "better time")
//   [47:0]  breadcrumb (see breadcrumb.h)
// The time is biased per search direction so that a smaller key always means a
// better arrival; this lets us use a single native 64-bit atomicMin to update
// the time and carry its breadcrumb atomically (no torn writes).
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
  CISTA_CUDA_COMPAT static std::uint64_t pack(delta_t const t,
                                              std::uint64_t const bc) {
    return (static_cast<std::uint64_t>(to_key(t)) << 48U) | (bc & kBcMask);
  }
  // packed value representing kInvalid: all-ones (worst key 0xFFFF in the high
  // bits; breadcrumb bits unused for invalid entries). All-ones so a single
  // cudaMemset(0xFF) produces it.
  CISTA_CUDA_COMPAT static std::uint64_t invalid_packed() {
    return ~std::uint64_t{0};
  }

  __device__ delta_t get(std::uint8_t const k,
                         location_idx_t const l,
                         via_offset_t const via) {
    return from_key(
        static_cast<std::uint16_t>(data_[internal_idx(k, l, via)] >> 48U));
  }

  __device__ delta_t get(location_idx_t const l, via_offset_t const via) {
    return from_key(
        static_cast<std::uint16_t>(data_[internal_idx(0U, l, via)] >> 48U));
  }

  __device__ delta_t get(std::uint8_t const i) {
    return from_key(static_cast<std::uint16_t>(data_[i] >> 48U));
  }

  __device__ std::uint64_t get_bc(std::uint8_t const k,
                                  location_idx_t const l,
                                  via_offset_t const via) {
    return data_[internal_idx(k, l, via)] & kBcMask;
  }

  __device__ bool update_min(std::uint8_t const k,
                             location_idx_t const l,
                             via_offset_t const via,
                             delta_t const val,
                             std::uint64_t const bc = 0U) {
    return update_min(static_cast<std::size_t>(internal_idx(k, l, via)), val,
                      bc);
  }

  __device__ bool update_min(location_idx_t const l,
                             via_offset_t const via,
                             delta_t const val,
                             std::uint64_t const bc = 0U) {
    return update_min(static_cast<std::size_t>(internal_idx(0U, l, via)), val,
                      bc);
  }

  __device__ bool update_min(std::size_t const idx,
                             delta_t const val,
                             std::uint64_t const bc = 0U) {
    auto const new_packed = pack(val, bc);
    auto* const addr =
        reinterpret_cast<unsigned long long*>(&data_[idx]);  // NOLINT
    auto const old =
        atomicMin(addr, static_cast<unsigned long long>(new_packed));
    // mark the first write to a previously-invalid entry in a dirty bitfield,
    // so reset only clears the touched entries instead of memset-ing the whole
    // (huge) buffer. One bit per entry (size/8 bytes).
    if (dirty_bits_ != nullptr && (old >> 48U) == 0xFFFFULL &&
        (new_packed >> 48U) != 0xFFFFULL) {
      atomicOr(&dirty_bits_[idx >> 5U], 1U << (idx & 31U));
    }
    // strictly improved iff the new time key is smaller than the old one
    return (new_packed >> 48U) < (old >> 48U);
  }

  __device__ __forceinline__ unsigned internal_idx(std::uint8_t const k,
                                                   location_idx_t const l,
                                                   via_offset_t const via) {
    return (k * n_locations_ * Vias) + (l.v_ * Vias) + via;
  }

  cuda::std::span<std::uint64_t> data_;
  std::uint32_t n_locations_;
  // optional selective-clear bitfield: one bit per entry, set on first
  // invalid->valid write. reset scans it and clears only the marked entries
  // instead of memset-ing the whole buffer. null = off.
  std::uint32_t* dirty_bits_ = nullptr;
};

}  // namespace nigiri::routing::gpu
