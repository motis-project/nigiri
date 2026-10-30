#pragma once

#include <cstdint>
#include <type_traits>

#include "nigiri/routing/gpu/gpu_runtime.cuh"

// Warp-level collectives, parameterised over the logical warp size.
//
// The RAPTOR kernels use a "logical warp" as their unit of cooperation, and
// that width is baked into the data layout: the station/route mark bitvecs
// store one word per logical warp and lane i owns bit i of that word (see
// mark_block_t below and the `w * kWarpSize + lane` indexing in raptor_impl).
// NVIDIA warps are 32 lanes; AMD GCN/CDNA wavefronts are 64.
//
// kWarpSize may be smaller than the hardware wavefront (two 32-lane logical
// warps per wave64), in which case the collectives below restrict themselves
// to the caller's half of the wave. It may never be larger.
//
// 32 is the default on both back ends. It is the native width on NVIDIA, and
// on a wave64 AMD part two 32-lane logical warps per wave measured ~9% faster
// than one 64-lane warp (RX 580 / gfx803, Swiss timetable, pong: 49.3 vs 45.2
// q/s): loop_routes gives a warp one route each, and most routes have well
// under 64 stops, so a 64-lane warp leaves half its lanes idle in
// update_route_warp. Override with -DNIGIRI_GPU_WARP_SIZE=64 to compare -
// longer routes or a different AMD part may flip the result.

#if !defined(NIGIRI_GPU_WARP_SIZE)
#define NIGIRI_GPU_WARP_SIZE 32
#endif

namespace nigiri::routing::gpu {

inline constexpr auto kWarpSize = unsigned{NIGIRI_GPU_WARP_SIZE};

static_assert(kWarpSize == 32U || kWarpSize == 64U);

#if !defined(NIGIRI_HIP)
static_assert(kWarpSize == 32U, "CUDA warps are 32 lanes wide");
#elif defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN_WAVEFRONT_SIZE__)
static_assert(kWarpSize <= unsigned{__AMDGCN_WAVEFRONT_SIZE__},
              "NIGIRI_GPU_WARP_SIZE exceeds the hardware wavefront size; "
              "pass -DNIGIRI_GPU_WARP_SIZE=32 for a wave32 target");
#endif

// One bit per lane of a logical warp: both the ballot masks and the words of
// the mark bitvecs the kernels iterate lane-per-bit.
using lane_mask_t =
    std::conditional_t<kWarpSize == 64U, std::uint64_t, std::uint32_t>;
using mark_block_t = lane_mask_t;

inline constexpr auto kAllLanes = ~lane_mask_t{0};

// words a device_bitvec<mark_block_t> needs to hold n bits
constexpr std::uint32_t n_bitvec_words(std::uint32_t const n) {
  return n / kWarpSize + 1U;
}

// ---------------------------------------------------------------- bit twiddling

// 1-based index of the lowest set bit, 0 if there is none (like __ffs).
template <typename Word>
__device__ __forceinline__ unsigned find_first_set(Word const w) {
  if constexpr (sizeof(Word) == 8U) {
    return static_cast<unsigned>(__ffsll(static_cast<unsigned long long>(w)));
  } else {
    return static_cast<unsigned>(__ffs(static_cast<int>(w)));
  }
}

template <typename Word>
__device__ __forceinline__ unsigned popcount(Word const w) {
  if constexpr (sizeof(Word) == 8U) {
    return static_cast<unsigned>(__popcll(static_cast<unsigned long long>(w)));
  } else {
    return static_cast<unsigned>(__popc(static_cast<unsigned>(w)));
  }
}

// atomicOr has no overload for the fixed-width aliases (std::uint64_t is
// `unsigned long` here, the intrinsic takes `unsigned long long`).
template <typename Word>
__device__ __forceinline__ void atomic_or(Word* const p, Word const v) {
  if constexpr (sizeof(Word) == 8U) {
    atomicOr(reinterpret_cast<unsigned long long*>(p),
             static_cast<unsigned long long>(v));
  } else {
    atomicOr(reinterpret_cast<unsigned int*>(p),
             static_cast<unsigned int>(v));
  }
}

// mask of the lanes below this one, for ballot-based stream compaction
__device__ __forceinline__ lane_mask_t lanes_below(unsigned const lane) {
  return (lane_mask_t{1U} << lane) - lane_mask_t{1U};
}

// ---------------------------------------------------------------- collectives

#if defined(NIGIRI_HIP)

__device__ __forceinline__ lane_mask_t warp_ballot(bool const pred) {
  // __ballot covers the whole wavefront, so shift the caller's logical warp
  // down to bit 0 and let the narrower lane_mask_t truncate the rest. When the
  // logical warp spans the whole wave the shift amount is always zero.
  return static_cast<lane_mask_t>(__ballot(pred) >>
                                  (__lane_id() & ~(kWarpSize - 1U)));
}

template <typename T>
__device__ __forceinline__ T warp_shfl(T const v, unsigned const src_lane) {
  return __shfl(v, static_cast<int>(src_lane), static_cast<int>(kWarpSize));
}

template <typename T>
__device__ __forceinline__ T warp_shfl_up(T const v, unsigned const delta) {
  return __shfl_up(v, delta, static_cast<int>(kWarpSize));
}

// __any() is a whole-wave reduction, so it is wrong for a narrower logical
// warp; ballot already restricts itself correctly.
__device__ __forceinline__ bool warp_any(bool const pred) {
  return warp_ballot(pred) != lane_mask_t{0U};
}

#else

__device__ __forceinline__ lane_mask_t warp_ballot(bool const pred) {
  return __ballot_sync(kAllLanes, pred);
}

template <typename T>
__device__ __forceinline__ T warp_shfl(T const v, unsigned const src_lane) {
  return __shfl_sync(kAllLanes, v, static_cast<int>(src_lane),
                     static_cast<int>(kWarpSize));
}

template <typename T>
__device__ __forceinline__ T warp_shfl_up(T const v, unsigned const delta) {
  return __shfl_up_sync(kAllLanes, v, delta, static_cast<int>(kWarpSize));
}

__device__ __forceinline__ bool warp_any(bool const pred) {
  return __any_sync(kAllLanes, pred);
}

#endif

}  // namespace nigiri::routing::gpu
