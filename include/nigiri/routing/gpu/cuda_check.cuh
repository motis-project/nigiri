#pragma once

#include <cstdlib>
#include <iostream>

#include "nigiri/routing/gpu/gpu_runtime.cuh"

#define CUDA_CHECK(code)                                              \
  if ((code) != cudaSuccess) {                                        \
    std::cerr << "GPU error: " << cudaGetErrorString(code) << " at " \
              << __FILE__ << ":" << __LINE__;                         \
    std::terminate();                                                 \
  }
