#pragma once

#include "cuda_runtime.h"

namespace nigiri::routing::gpu {

__device__ __forceinline__ unsigned get_block_thread_id() {
  return threadIdx.x + (blockDim.x * threadIdx.y);
}

__device__ __forceinline__ unsigned get_global_thread_id() {
  return get_block_thread_id() + (blockDim.x * blockDim.y * blockIdx.x);
}

__device__ __forceinline__ unsigned get_block_stride() {
  return blockDim.x * blockDim.y;
}

__device__ __forceinline__ unsigned get_global_stride() {
  return get_block_stride() * gridDim.x * gridDim.y;
}

}  // namespace nigiri::routing::gpu