#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

#include "nigiri/routing/gpu/cuda_check.cuh"

namespace nigiri::routing::gpu {

// like thrust::device_vector BUT
// - no value-initialization (no init kernel for fully overwritten buffers)
// - no cudaMalloc which synchronizes the whole device (allocation is
//   stream-ordered via cudaMallocAsync)
template <typename T>
struct device_buffer {
  device_buffer() = default;

  device_buffer(device_buffer const&) = delete;
  device_buffer& operator=(device_buffer const&) = delete;

  ~device_buffer() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }

  // sets the used size; contents are undefined until written
  T* ensure(std::size_t const n, cudaStream_t const stream) {
    if (n > cap_) {
      if (ptr_ != nullptr) {
        CUDA_CHECK(cudaFreeAsync(ptr_, stream));
      }
      CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&ptr_),
                                 n * sizeof(T), stream));
      cap_ = n;
    }
    size_ = n;
    return ptr_;
  }

  void clear() { size_ = 0U; }

  T const* data() const { return ptr_; }
  std::size_t size() const { return size_; }

  T* ptr_{nullptr};
  std::size_t size_{0U};
  std::size_t cap_{0U};
};

}  // namespace nigiri::routing::gpu
