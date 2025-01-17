#pragma once

#include <limits>

#include <cuda/std/span>

#include "thrust/device_vector.h"

#include "cista/containers/bitvec.h"
#include "cista/containers/vecvec.h"

#include "nigiri/common/flat_matrix_view.h"

namespace nigiri::routing::gpu {

static constexpr auto const kUnreachable =
    std::numeric_limits<std::uint16_t>::max();

template <typename T>
using device_flat_matrix_view = base_flat_matrix_view<cuda::std::span<T>>;

template <typename T>
cuda::std::span<T const> to_view(thrust::device_vector<T> const& v) {
  return {thrust::raw_pointer_cast(v.data()), v.size()};
}

template <typename T>
cuda::std::span<T> to_mutable_view(thrust::device_vector<T>& v) {
  return {thrust::raw_pointer_cast(v.data()), v.size()};
}

template <typename T>
thrust::device_vector<typename T::value_type> to_device(T const& t) {
  return thrust::device_vector<typename T::value_type>(begin(t), end(t));
}

thrust::device_vector<std::uint64_t> to_device(cista::raw::bitvec const& t) {
  return to_device(t.blocks_);
}

template <typename Host>
struct device_vecvec {
  using H = std::decay_t<Host>;
  using data_value_type = typename H::data_value_type;
  using index_value_type = typename H::index_value_type;
  explicit device_vecvec(H const& h)
      : data_{to_device(h.data_)}, index_{to_device(h.bucket_starts_)} {}
  thrust::device_vector<data_value_type> data_;
  thrust::device_vector<index_value_type> index_;
};

template <typename Host>
using d_vecvec_view = cista::basic_vecvec<
    typename std::decay_t<Host>::key,
    cuda::std::span<typename std::decay_t<Host>::data_value_type const>,
    cuda::std::span<typename std::decay_t<Host>::index_value_type const>>;

template <typename Host>
d_vecvec_view<Host> to_view(device_vecvec<Host> const& h) {
  return {.data_ = to_view(h.data_), .bucket_starts_ = to_view(h.index_)};
}

template <typename K, typename V>
struct d_vecmap_view {
  d_vecmap_view(cuda::std::span<V const> data) : data_{data} {}
  d_vecmap_view(thrust::device_vector<V> const& data) : data_{to_view(data)} {}

  __forceinline__ __device__ V const& operator[](K const k) const {
    return data_[to_idx(k)];
  }

  cuda::std::span<V const> data_;
};

}  // namespace nigiri::routing::gpu