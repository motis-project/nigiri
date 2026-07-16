#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

namespace nigiri::routing::gpu {

// grow-only page-locked host staging buffer: cudaMemcpyAsync from pageable
// host memory degrades to a synchronous copy through the driver's own pinned
// staging area, so per-query uploads stage here first.
template <typename T>
struct pinned_host_buffer {
  pinned_host_buffer() = default;

  pinned_host_buffer(pinned_host_buffer const&) = delete;
  pinned_host_buffer& operator=(pinned_host_buffer const&) = delete;

  ~pinned_host_buffer() {
    if (ptr_ != nullptr) {
      cudaFreeHost(ptr_);
    }
  }

  T* ensure(std::size_t const n) {
    if (n > cap_) {
      if (ptr_ != nullptr) {
        cudaFreeHost(ptr_);
      }
      cudaMallocHost(reinterpret_cast<void**>(&ptr_), n * sizeof(T));
      cap_ = n;
    }
    return ptr_;
  }

  T* ptr_{nullptr};
  std::size_t cap_{0U};
};

}  // namespace nigiri::routing::gpu
