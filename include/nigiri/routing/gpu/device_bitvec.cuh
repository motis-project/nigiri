#pragma once

#include <cstdint>

#include "cuda/std/span"
#include "cuda_runtime.h"

#include "nigiri/routing/gpu/stride.cuh"

#define BITS_PER_BLOCK (sizeof(block_t) * 8U)

namespace nigiri::routing::gpu {

template <typename Block = std::uint64_t>
struct device_bitvec {
  using block_t = Block;

  __device__ void mark(block_t const i) {
    atomicOr(&blocks_[i / sizeof(block_t) * 8U],
             block_t{1U} << (i % BITS_PER_BLOCK));
  }

  __device__ void zero_out() {
    auto const global_t_id = get_global_thread_id();
    auto const global_stride = get_global_stride();
    for (auto i = global_t_id; i < blocks_.size(); i += global_stride) {
      blocks_[i] = 0U;
    }
  };

  __device__ bool test(block_t const i) const { return (*this)[i]; }

  __device__ bool operator[](block_t const i) const {
    auto const bit = i % BITS_PER_BLOCK;
    return (blocks_[i / BITS_PER_BLOCK] & (block_t{1U} << bit)) != 0U;
  }

  __device__ std::size_t size() const {
    return blocks_.size() * BITS_PER_BLOCK;
  }

  cuda::std::span<block_t> blocks_;
};

}  // namespace nigiri::routing::gpu

#undef BITS_PER_BLOCK
