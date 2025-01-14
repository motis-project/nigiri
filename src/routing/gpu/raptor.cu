#include "nigiri/routing/gpu/raptor.h"

#include <cstdio>
#include <iostream>

#include "cuda/std/array"
#include "cuda/std/span"

#include "thrust/device_vector.h"
#include "thrust/fill.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/common/flat_matrix_view.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

#include "cista/serialization.h"

namespace nigiri::routing::gpu {

template <typename T>
using device_vec = thrust::device_vector<T>;

using device_bitvec = cista::basic_bitvec<thrust::device_vector<std::uint64_t>>;

template <typename T>
using device_flat_matrix_view = base_flat_matrix_view<cuda::std::span<T>>;

inline device_bitvec to_device(bitvec const& v) {
  return v.size() == 0U
             ? device_bitvec{}
             : device_bitvec{{begin(v.blocks_), end(v.blocks_)}, v.size()};
}

template <typename T>
inline thrust::device_vector<typename T::value_type> to_device(T const& t) {
  return {begin(t), end(t)};
}

template <typename T, std::size_t N>
cuda::std::array<T, N> to_device(std::array<T, N> const& a) {
  auto ret = cuda::std::array<T, N>{};
  for (auto i = 0U; i != N; ++i) {
    ret[i] = to_device(a[i]);
  }
  return ret;
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

template <typename T>
cuda::std::span<T const> to_view(thrust::device_vector<T> const& v) {
  return {thrust::raw_pointer_cast(v.data()), v.size()};
}

template <typename Host>
d_vecvec_view<Host> to_view(device_vecvec<Host> const& h) {
  return {.data_ = to_view(h.data_), .bucket_starts_ = to_view(h.index_)};
}

template <typename K, typename V>
struct d_vecmap_view {
  d_vecmap_view(cuda::std::span<V const> data) : data_{data} {}
  d_vecmap_view(thrust::device_vector<V> const& data) : data_{to_view(data)} {}

  __forceinline__ __device__ V const& operator[](K const k) const {
    return data_[k.v_];
  }

  cuda::std::span<V const> data_;
};

struct device_timetable {
  using t = timetable;

  std::uint32_t n_locations_;
  std::uint32_t n_routes_;

  d_vecmap_view<location_idx_t, u8_minutes> transfer_time_;
  d_vecvec_view<decltype(t{}.locations_.footpaths_out_[0])> footpaths_out_;
  d_vecvec_view<decltype(t{}.locations_.footpaths_in_[0])> footpaths_in_;

  cuda::std::span<delta const> route_stop_times_;
  d_vecmap_view<route_idx_t, interval<std::uint32_t>> route_stop_time_ranges_;
  d_vecmap_view<route_idx_t, interval<transport_idx_t>> route_transport_ranges_;
  d_vecmap_view<route_idx_t, clasz> route_clasz_;

  d_vecvec_view<decltype(t{}.route_location_seq_)> route_location_seq_;
  d_vecvec_view<decltype(t{}.location_routes_)> location_routes_;

  d_vecmap_view<transport_idx_t, bitfield_idx_t> transport_traffic_days_;
  d_vecmap_view<bitfield_idx_t, bitfield> bitfields_;

  interval<date::sys_days> date_range_;
};

struct gpu_timetable::impl {
  using t = timetable;

  explicit impl(timetable const& tt)
      : n_locations_{tt.n_locations()},
        n_routes_{tt.n_routes()},
        transfer_time_{to_device(tt.locations_.transfer_time_)},
        footpaths_out_{tt.locations_.footpaths_out_[0]},
        footpaths_in_{tt.locations_.footpaths_in_[0]},
        route_stop_times_{to_device(tt.route_stop_times_)},
        route_stop_time_ranges_{to_device(tt.route_stop_time_ranges_)},
        route_clasz_{to_device(tt.route_clasz_)},
        route_location_seq_{tt.route_location_seq_},
        location_routes_{tt.location_routes_},
        transport_traffic_days_{to_device(tt.transport_traffic_days_)},
        bitfields_{to_device(tt.bitfields_)},
        date_range_{tt.date_range_} {}

  device_timetable to_device_timetable() const {
    return {.n_locations_ = n_locations_,
            .n_routes_ = n_routes_,
            .transfer_time_ = transfer_time_,
            .footpaths_out_ = to_view(footpaths_out_),
            .footpaths_in_ = to_view(footpaths_in_),
            .route_stop_times_ = to_view(route_stop_times_),
            .route_stop_time_ranges_ = to_view(route_stop_time_ranges_),
            .route_transport_ranges_ = to_view(route_transport_ranges_),
            .route_clasz_ = to_view(route_clasz_),
            .route_location_seq_ = to_view(route_location_seq_),
            .location_routes_ = to_view(location_routes_),
            .transport_traffic_days_ = to_view(transport_traffic_days_),
            .bitfields_ = to_view(bitfields_),
            .date_range_ = date_range_};
  }

  std::uint32_t n_locations_;
  std::uint32_t n_routes_;

  thrust::device_vector<u8_minutes> transfer_time_;
  device_vecvec<decltype(t{}.locations_.footpaths_out_[0])> footpaths_out_;
  device_vecvec<decltype(t{}.locations_.footpaths_in_[0])> footpaths_in_;

  thrust::device_vector<delta> route_stop_times_;
  thrust::device_vector<interval<std::uint32_t>> route_stop_time_ranges_;
  thrust::device_vector<interval<transport_idx_t>> route_transport_ranges_;
  thrust::device_vector<clasz> route_clasz_;

  device_vecvec<decltype(t{}.route_location_seq_)> route_location_seq_;
  device_vecvec<decltype(t{}.location_routes_)> location_routes_;

  thrust::device_vector<bitfield_idx_t> transport_traffic_days_;
  thrust::device_vector<bitfield> bitfields_;

  interval<date::sys_days> date_range_;
};

gpu_timetable::gpu_timetable(timetable const& tt)
    : impl_{std::make_unique<impl>(tt)} {}

gpu_timetable::~gpu_timetable() = default;

struct gpu_raptor_state::impl {
  explicit impl(gpu_timetable const& gtt)
      : tt_{gtt.impl_->to_device_timetable()} {}

  void resize(unsigned n_locations,
              unsigned n_routes,
              unsigned n_rt_transports) {
    n_locations_ = n_locations;
    tmp_storage_.resize(n_locations * (kMaxVias + 1));
    best_storage_.resize(n_locations * (kMaxVias + 1));
    round_times_storage_.resize(n_locations * (kMaxVias + 1) *
                                (kMaxTransfers + 1));
    station_mark_.resize(n_locations);
    prev_station_mark_.resize(n_locations);
    route_mark_.resize(n_routes);
    rt_transport_mark_.resize(n_rt_transports);
    end_reachable_.resize(n_locations);
  }

  template <via_offset_t Vias>
  cuda::std::span<std::array<delta_t, Vias + 1>> get_tmp() {
    return {reinterpret_cast<cuda::std::array<delta_t, Vias + 1>*>(
                thrust::raw_pointer_cast(tmp_storage_.data())),
            n_locations_};
  }

  template <via_offset_t Vias>
  cuda::std::span<cuda::std::array<delta_t, Vias + 1>> get_best() {
    return {reinterpret_cast<cuda::std::array<delta_t, Vias + 1>*>(
                thrust::raw_pointer_cast(best_storage_.data())),
            n_locations_};
  }

  template <via_offset_t Vias>
  cuda::std::span<cuda::std::array<delta_t, Vias + 1> const> get_best() const {
    return {reinterpret_cast<cuda::std::array<delta_t, Vias + 1> const*>(
                thrust::raw_pointer_cast(best_storage_.data())),
            n_locations_};
  }

  template <via_offset_t Vias>
  device_flat_matrix_view<cuda::std::array<delta_t, Vias + 1>>
  get_round_times() {
    return {{reinterpret_cast<cuda::std::array<delta_t, Vias + 1>*>(
                 thrust::raw_pointer_cast(round_times_storage_.data())),
             n_locations_ * (kMaxTransfers + 1)},
            kMaxTransfers + 1U,
            n_locations_};
  }

  template <via_offset_t Vias>
  device_flat_matrix_view<cuda::std::array<delta_t, Vias + 1> const>
  get_round_times() const {
    return {{reinterpret_cast<cuda::std::array<delta_t, Vias + 1> const*>(
                 thrust::raw_pointer_cast(round_times_storage_.data())),
             n_locations_ * (kMaxTransfers + 1)},
            kMaxTransfers + 1U,
            n_locations_};
  }

  unsigned n_locations_{};
  thrust::device_vector<delta_t> time_at_dest_{kMaxTransfers};
  thrust::device_vector<delta_t> tmp_storage_{};
  thrust::device_vector<delta_t> best_storage_{};
  thrust::device_vector<delta_t> round_times_storage_{};
  device_bitvec station_mark_{};
  device_bitvec prev_station_mark_{};
  device_bitvec route_mark_{};
  device_bitvec rt_transport_mark_{};
  device_bitvec end_reachable_{};

  device_timetable tt_;
};

gpu_raptor_state::gpu_raptor_state(gpu_timetable const& gtt)
    : impl_{std::make_unique<impl>(gtt)} {}

gpu_raptor_state::~gpu_raptor_state() = default;

gpu_raptor_state& gpu_raptor_state::resize(unsigned const n_locations,
                                           unsigned const n_routes,
                                           unsigned const n_rt_transports) {
  impl_->resize(n_locations, n_routes, n_rt_transports);
  return *this;
}

template <direction SearchDir, bool Rt, via_offset_t Vias>
gpu_raptor<SearchDir, Rt, Vias>::gpu_raptor(
    timetable const& tt,
    rt_timetable const* rtt,
    gpu_raptor_state& state,
    bitvec& is_dest,
    std::array<bitvec, kMaxVias> const& is_via,
    std::vector<std::uint16_t> const& dist_to_dest,
    hash_map<location_idx_t, std::vector<td_offset>> const& td_dist_to_dest,
    std::vector<std::uint16_t> const& lb,
    std::vector<via_stop> const& via_stops,
    day_idx_t const base,
    clasz_mask_t const allowed_claszes,
    bool const require_bike_transport,
    bool const is_wheelchair,
    transfer_time_settings const& tts)
    : tt_{tt},
      rtt_{rtt},
      n_days_{tt_.internal_interval_days().size().count()},
      n_locations_{tt_.n_locations()},
      n_routes_{tt.n_routes()},
      n_rt_transports_{Rt ? rtt->n_rt_transports() : 0U},
      state_{state.resize(n_locations_, n_routes_, n_rt_transports_)},
      is_dest_{is_dest},
      is_via_{is_via},
      dist_to_end_{dist_to_dest},
      td_dist_to_end_{td_dist_to_dest},
      lb_{lb},
      via_stops_{via_stops},
      base_{base},
      allowed_claszes_{allowed_claszes},
      require_bike_transport_{require_bike_transport},
      is_wheelchair_{is_wheelchair},
      transfer_time_settings_{tts} {}

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

template <direction SearchDir>
constexpr static bool is_better(auto a, auto b) {
  return SearchDir == direction::kForward ? a < b : a > b;
}

template <direction SearchDir>
constexpr static bool is_better_or_eq(auto a, auto b) {
  return SearchDir == direction::kForward ? a <= b : a >= b;
}

template <direction SearchDir>
constexpr auto get_best(auto a, auto b) {
  return is_better<SearchDir>(a, b) ? a : b;
}

template <direction SearchDir>
constexpr auto get_best(auto x, auto... y) {
  ((x = get_best<SearchDir>(x, y)), ...);
  return x;
}

constexpr auto min(auto x, auto y) { return x <= y ? x : y; }

constexpr int as_int(location_idx_t const d) { return static_cast<int>(d.v_); }
constexpr int as_int(day_idx_t const d) { return static_cast<int>(d.v_); }

template <direction SearchDir>
constexpr auto dir(auto const a) {
  return (SearchDir == direction::kForward ? 1 : -1) * a;
}

template <direction SearchDir, bool Rt, via_offset_t Vias>
__global__ void exec_raptor(
    device_timetable const tt,
    std::uint8_t const max_transfers,
    date::sys_days const base,
    unixtime_t const worst_time_at_dest,
    delta_t* const time_at_dest,
    device_flat_matrix_view<cuda::std::array<delta_t, Vias + 1>> round_times,
    cuda::std::span<cuda::std::array<delta_t, Vias + 1>> best) {
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();

  auto const d_worst_at_dest = unix_to_delta(base, worst_time_at_dest);
  for (auto i = global_t_id; i < kMaxTransfers + 1U; i += global_stride) {
    time_at_dest[i] = get_best<SearchDir>(d_worst_at_dest, time_at_dest[i]);
  }

  auto const end_k = min(max_transfers, kMaxTransfers) + 1U;
  for (auto k = 1U; k != end_k; ++k) {
    // ==================
    // RAPTOR ROUND START
    // ------------------

    // Reuse best time from previous time at start (for range queries).
    for (auto i = global_t_id; i < tt.n_locations_; i += global_stride) {
      for (auto v = 0U; v != Vias + 1; ++v) {
        best[i][v] = get_best<SearchDir>(round_times[k][i][v], best[i][v]);
      }
    }

    // ----------------
    // RAPTOR ROUND END
    // ================
  }
}

template <direction SearchDir, bool Rt, via_offset_t Vias>
void gpu_raptor<SearchDir, Rt, Vias>::execute(
    unixtime_t const start_time,
    std::uint8_t const max_transfers,
    unixtime_t const worst_time_at_dest,
    profile_idx_t const prf_idx,
    pareto_set<journey>& results) {
  exec_raptor<SearchDir, Rt, Vias><<<1, 1>>>(
      state_.impl_->tt_, max_transfers, base(), worst_time_at_dest,
      thrust::raw_pointer_cast(state_.impl_->time_at_dest_.data()),
      state_.impl_->get_round_times<Vias>(), state_.impl_->get_best<Vias>());
}

template <direction SearchDir, bool Rt, via_offset_t Vias>
void gpu_raptor<SearchDir, Rt, Vias>::reset_arrivals() {
  thrust::fill(thrust::cuda::par_nosync, begin(state_.impl_->time_at_dest_),
               end(state_.impl_->time_at_dest_), kInvalid);
  thrust::fill(thrust::cuda::par_nosync,
               begin(state_.impl_->round_times_storage_),
               end(state_.impl_->round_times_storage_), kInvalid);
  cudaDeviceSynchronize();
}

template <direction SearchDir, bool Rt, via_offset_t Vias>
void gpu_raptor<SearchDir, Rt, Vias>::next_start_time() {
  starts_.clear();
  thrust::fill(thrust::cuda::par_nosync, begin(state_.impl_->best_storage_),
               end(state_.impl_->best_storage_), kInvalid);
  thrust::fill(thrust::cuda::par_nosync, begin(state_.impl_->tmp_storage_),
               end(state_.impl_->tmp_storage_), kInvalid);
  thrust::fill(thrust::cuda::par_nosync,
               begin(state_.impl_->prev_station_mark_.blocks_),
               end(state_.impl_->prev_station_mark_.blocks_), 0U);
  thrust::fill(thrust::cuda::par_nosync,
               begin(state_.impl_->station_mark_.blocks_),
               end(state_.impl_->station_mark_.blocks_), 0U);
  thrust::fill(thrust::cuda::par_nosync,
               begin(state_.impl_->route_mark_.blocks_),
               end(state_.impl_->route_mark_.blocks_), 0U);
  if constexpr (Rt) {
    thrust::fill(thrust::cuda::par_nosync,
                 begin(state_.impl_->rt_transport_mark_.blocks_),
                 end(state_.impl_->rt_transport_mark_.blocks_), 0U);
  }
  cudaDeviceSynchronize();
}

template <direction SearchDir, bool Rt, via_offset_t Vias>
void gpu_raptor<SearchDir, Rt, Vias>::reconstruct(query const&, journey&) {}

template <direction SearchDir, bool Rt, via_offset_t Vias>
void gpu_raptor<SearchDir, Rt, Vias>::add_start(location_idx_t const l,
                                                unixtime_t const t) {
  starts_.emplace_back(l, t);
}

template class gpu_raptor<direction::kForward, true, 0U>;
template class gpu_raptor<direction::kForward, true, 1U>;
template class gpu_raptor<direction::kForward, true, 2U>;
template class gpu_raptor<direction::kForward, false, 0U>;
template class gpu_raptor<direction::kForward, false, 1U>;
template class gpu_raptor<direction::kForward, false, 2U>;
template class gpu_raptor<direction::kBackward, true, 0U>;
template class gpu_raptor<direction::kBackward, true, 1U>;
template class gpu_raptor<direction::kBackward, true, 2U>;
template class gpu_raptor<direction::kBackward, false, 0U>;
template class gpu_raptor<direction::kBackward, false, 1U>;
template class gpu_raptor<direction::kBackward, false, 2U>;

}  // namespace nigiri::routing::gpu