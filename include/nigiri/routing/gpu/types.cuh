#pragma once

#include <limits>

#include "nigiri/routing/gpu/gpu_runtime.cuh"
#include "nigiri/routing/gpu/gpu_std.cuh"

#include "thrust/device_vector.h"
#include "thrust/execution_policy.h"

#include "cista/containers/bitvec.h"
#include "cista/containers/vecvec.h"

#include "nigiri/common/flat_matrix_view.h"

namespace nigiri::routing::gpu {

static constexpr auto const kUnreachable =
    std::numeric_limits<std::uint16_t>::max();

// thrust's stream-ordered execution policy: the CUDA backend spells it
// thrust::cuda::par, rocThrust's HIP backend thrust::hip::par.
inline auto par_on(cudaStream_t const stream) {
#if defined(NIGIRI_HIP)
  return thrust::hip::par.on(stream);
#else
  return thrust::cuda::par.on(stream);
#endif
}

template <typename T>
using device_flat_matrix_view = base_flat_matrix_view<d_span<T>>;

template <typename T>
d_span<T const> to_view(thrust::device_vector<T> const& v) {
  return d_span<T const>(thrust::raw_pointer_cast(v.data()), v.size());
}

template <typename T>
d_span<T> to_mutable_view(thrust::device_vector<T>& v) {
  return d_span<T>(thrust::raw_pointer_cast(v.data()), v.size());
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
  device_vecvec() = default;
  explicit device_vecvec(H const& h)
      : data_{to_device(h.data_)}, index_{to_device(h.bucket_starts_)} {}
  thrust::device_vector<data_value_type> data_;
  thrust::device_vector<index_value_type> index_;
};

template <typename Host>
using d_vecvec_view = cista::basic_vecvec<
    typename std::decay_t<Host>::key,
    d_span<typename std::decay_t<Host>::data_value_type const>,
    d_span<typename std::decay_t<Host>::index_value_type const>>;

template <typename Host>
d_vecvec_view<Host> to_view(device_vecvec<Host> const& h) {
  return {.data_ = to_view(h.data_), .bucket_starts_ = to_view(h.index_)};
}

template <typename K, typename V>
struct d_vecmap_view {
  d_vecmap_view() = default;
  d_vecmap_view(d_span<V const> data) : data_{data} {}
  d_vecmap_view(thrust::device_vector<V> const& data) : data_{to_view(data)} {}

  __forceinline__ __device__ V const& operator[](K const k) const {
    return data_[to_idx(k)];
  }

  d_span<V const> data_;
};

}  // namespace nigiri::routing::gpu