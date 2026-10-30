#pragma once

// Single include point for the GPU runtime API.
//
// nigiri's GPU sources are written against the CUDA runtime API. HIP mirrors
// that API almost name for name, so instead of forking the sources we map the
// (small) set of entry points they actually use onto their HIP equivalents
// here. Everything is an inline wrapper rather than a macro, so overload
// resolution and type checking behave exactly as on the CUDA path.

#if defined(NIGIRI_HIP)

#include <cstddef>

#include <hip/hip_runtime.h>

using cudaError_t = hipError_t;
using cudaStream_t = hipStream_t;
using cudaMemcpyKind = hipMemcpyKind;

inline constexpr auto cudaSuccess = hipSuccess;
inline constexpr auto cudaMemcpyHostToDevice = hipMemcpyHostToDevice;
inline constexpr auto cudaMemcpyDeviceToHost = hipMemcpyDeviceToHost;

inline char const* cudaGetErrorString(cudaError_t const e) {
  return hipGetErrorString(e);
}
inline cudaError_t cudaPeekAtLastError() { return hipPeekAtLastError(); }

inline cudaError_t cudaStreamCreate(cudaStream_t* const s) {
  return hipStreamCreate(s);
}
inline cudaError_t cudaStreamDestroy(cudaStream_t const s) {
  return hipStreamDestroy(s);
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t const s) {
  return hipStreamSynchronize(s);
}

inline cudaError_t cudaMalloc(void** const p, std::size_t const n) {
  return hipMalloc(p, n);
}
inline cudaError_t cudaFree(void* const p) { return hipFree(p); }
inline cudaError_t cudaMallocHost(void** const p, std::size_t const n) {
  return hipHostMalloc(p, n);
}
inline cudaError_t cudaFreeHost(void* const p) { return hipHostFree(p); }

// The stream-ordered allocator needs a memory pool, which not every AMD GPU
// exposes (Polaris/gfx803 does not). Where it is missing we fall back to the
// synchronous allocator: hipMalloc/hipFree synchronise the device, which is
// correct - just slower - and these paths only run when a buffer has to grow.
inline bool has_mem_pools() {
  static auto const supported = [] {
    auto dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) {
      return false;
    }
    auto v = 0;
    return hipDeviceGetAttribute(&v, hipDeviceAttributeMemoryPoolsSupported,
                                 dev) == hipSuccess &&
           v != 0;
  }();
  return supported;
}

inline cudaError_t cudaMallocAsync(void** const p,
                                   std::size_t const n,
                                   cudaStream_t const s) {
  return has_mem_pools() ? hipMallocAsync(p, n, s) : hipMalloc(p, n);
}
inline cudaError_t cudaFreeAsync(void* const p, cudaStream_t const s) {
  return has_mem_pools() ? hipFreeAsync(p, s) : hipFree(p);
}

inline cudaError_t cudaMemcpyAsync(void* const dst,
                                   void const* const src,
                                   std::size_t const n,
                                   cudaMemcpyKind const kind,
                                   cudaStream_t const s) {
  return hipMemcpyAsync(dst, src, n, kind, s);
}
inline cudaError_t cudaMemsetAsync(void* const p,
                                   int const value,
                                   std::size_t const n,
                                   cudaStream_t const s) {
  return hipMemsetAsync(p, value, n, s);
}

template <typename Kernel>
inline cudaError_t cudaOccupancyMaxPotentialBlockSize(
    int* const grid_size,
    int* const block_size,
    Kernel kernel,
    std::size_t const dyn_shared_mem = 0U,
    int const block_size_limit = 0) {
  return hipOccupancyMaxPotentialBlockSize(grid_size, block_size, kernel,
                                           dyn_shared_mem, block_size_limit);
}

#else

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#endif
