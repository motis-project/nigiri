#pragma once

// Device-usable span/array.
//
// CUDA gets them from libcu++, whose containers are annotated __host__
// __device__. HIP has no libcu++, so d_span is a small device-annotated stand
// in (see below); std::array's accessors are usable as they are, because clang
// treats constexpr functions as implicitly __host__ __device__ in HIP mode.

#include <cstddef>

#if defined(NIGIRI_HIP)

#include <array>
#include <iterator>

namespace nigiri::routing::gpu {

// std::span cannot be used in HIP device code: libstdc++'s accessors reference
// __glibcxx_assert_fail on their constant-evaluation path, and clang rejects
// the reference to a __host__ function from __host__ __device__ code even
// though the branch is dead at run time. This is the subset the GPU sources
// need, annotated for the device.
template <typename T>
struct d_span {
  using value_type = std::remove_cv_t<T>;
  using size_type = std::size_t;
  using iterator = T*;
  using reverse_iterator = std::reverse_iterator<T*>;

  d_span() = default;
  __host__ __device__ constexpr d_span(T* const ptr, size_type const size)
      : ptr_{ptr}, size_{size} {}

  template <typename U,
            typename = std::enable_if_t<std::is_same_v<T, U const>>>
  __host__ __device__ constexpr d_span(d_span<U> const& o)
      : ptr_{o.data()}, size_{o.size()} {}

  __host__ __device__ constexpr T* data() const { return ptr_; }
  __host__ __device__ constexpr size_type size() const { return size_; }
  __host__ __device__ constexpr size_type size_bytes() const {
    return size_ * sizeof(T);
  }
  __host__ __device__ constexpr bool empty() const { return size_ == 0U; }

  __host__ __device__ constexpr T& operator[](size_type const i) const {
    return ptr_[i];
  }
  __host__ __device__ constexpr T& front() const { return ptr_[0]; }
  __host__ __device__ constexpr T& back() const { return ptr_[size_ - 1U]; }

  __host__ __device__ constexpr iterator begin() const { return ptr_; }
  __host__ __device__ constexpr iterator end() const { return ptr_ + size_; }
  __host__ __device__ constexpr reverse_iterator rbegin() const {
    return reverse_iterator{end()};
  }
  __host__ __device__ constexpr reverse_iterator rend() const {
    return reverse_iterator{begin()};
  }

  __host__ __device__ constexpr d_span subspan(size_type const offset,
                                               size_type const count) const {
    return {ptr_ + offset, count};
  }
  __host__ __device__ constexpr d_span subspan(size_type const offset) const {
    return {ptr_ + offset, size_ - offset};
  }

  T* ptr_{nullptr};
  size_type size_{0U};
};

template <typename T, std::size_t N>
using d_array = std::array<T, N>;

}  // namespace nigiri::routing::gpu

#else

#include <cuda/std/array>
#include <cuda/std/span>

namespace nigiri::routing::gpu {

template <typename T, std::size_t N = cuda::std::dynamic_extent>
using d_span = cuda::std::span<T, N>;

template <typename T, std::size_t N>
using d_array = cuda::std::array<T, N>;

}  // namespace nigiri::routing::gpu

#endif
