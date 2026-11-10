#pragma once

#include <utility>

#include <cuda_runtime_api.h>

#include "utl/get_or_create.h"

#include "nigiri/types.h"

namespace nigiri::routing::gpu {

// Occupancy-maximizing grid, cached per kernel address: kernels sharing a
// signature need their own block size ("too many resources" otherwise).
template <typename Kernel>
std::pair<int, int> launch_dims(Kernel kernel) {
  thread_local auto cache = hash_map<void const*, std::pair<int, int>>{};
  return utl::get_or_create(
      cache, reinterpret_cast<void const*>(kernel), [&]() {
        auto blocks = 0;
        auto threads = 0;
        cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, kernel, 0, 0);
        return std::pair{blocks, threads};
      });
}

template <typename Kernel, typename... Args>
void launch(Kernel kernel, cudaStream_t stream, Args&&... args) {
  auto const [blocks, threads] = launch_dims(kernel);
  kernel<<<blocks, threads, 0, stream>>>(std::forward<Args>(args)...);
}

}  // namespace nigiri::routing::gpu
