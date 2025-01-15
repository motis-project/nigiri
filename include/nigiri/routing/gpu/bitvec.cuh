#pragma once

#include <cstdint>

#include "cuda/std/span"
#include "cuda_runtime.h"

namespace nigiri::routing::gpu {

struct bitvec {
  using block_t = std::uint32_t;
  static constexpr auto const bits_per_block = sizeof(block_t) * 8U;

  __host__ bitvec(thrust::device_vector<block_t>& v)
      : blocks_{thrust::raw_pointer_cast(v.data()), v.size()} {}

  __device__ void mark(block_t const i) {
    atomicOr(&blocks_[i / bits_per_block], block_t{1U} << (i % bits_per_block));
  }

  constexpr bool operator[](block_t const i) {
    auto const bit = i % bits_per_block;
    return (blocks_[i / bits_per_block] & (block_t{1U} << bit)) != 0U;
  }

  constexpr std::size_t size() const { return blocks_.size() * bits_per_block; }

  cuda::std::span<block_t> blocks_;
};

}  // namespace nigiri::routing::gpu