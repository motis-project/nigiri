#pragma once

#include <cstddef>
#include <cstdint>

#include "nigiri/routing/gpu/gpu_std.cuh"

#include "nigiri/routing/gpu/stride.cuh"
#include "nigiri/routing/gpu/warp.cuh"

#define BITS_PER_BLOCK (sizeof(block_t) * 8U)

namespace nigiri::routing::gpu {

// Calls fn(bit_index) for every set bit of the word, lowest first. NOTE: when
// the body contains warp collectives (warp_shfl etc.), the word must be
// warp-uniform so all lanes iterate in lockstep.
template <typename Word, typename Fn>
__device__ __forceinline__ void for_each_set_bit(Word word, Fn&& fn) {
  while (word != Word{0U}) {
    auto const b = find_first_set(word) - 1U;
    word &= word - Word{1U};  // clear LSB
    fn(b);
  }
}

template <typename Block>
__device__ __forceinline__ bool test_bit(Block const* const blocks,
                                         std::size_t const i) {
  constexpr auto const kBits = sizeof(Block) * 8U;
  return (blocks[i / kBits] & (Block{1U} << (i % kBits))) != 0U;
}

template <typename Block = std::uint64_t>
struct device_bitvec {
  using block_t = Block;

  __device__ void mark(block_t const i) {
    atomic_or(&blocks_[i / BITS_PER_BLOCK],
              static_cast<block_t>(block_t{1U} << (i % BITS_PER_BLOCK)));
  }

  __device__ void swap_reset(device_bitvec& o) {
    auto const t_id = get_global_thread_id();
    auto const stride = get_global_stride();
    for (auto idx = t_id; idx < blocks_.size(); idx += stride) {
      blocks_[idx] = o.blocks_[idx];
      o.blocks_[idx] = 0U;
    }
  }

  __device__ void reset() {
    auto const global_t_id = get_global_thread_id();
    auto const global_stride = get_global_stride();
    for (auto i = global_t_id; i < blocks_.size(); i += global_stride) {
      blocks_[i] = 0U;
    }
  }

  __device__ bool test(block_t const i) const { return (*this)[i]; }

  __device__ bool operator[](block_t const i) const {
    return test_bit(blocks_.data(), i);
  }

  __device__ std::size_t size() const {
    return blocks_.size() * BITS_PER_BLOCK;
  }

  d_span<block_t> blocks_;
};

}  // namespace nigiri::routing::gpu

#undef BITS_PER_BLOCK
