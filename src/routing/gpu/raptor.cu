#include "nigiri/routing/gpu/raptor.h"

#include <cstdio>
#include <iostream>

#include "thrust/device_vector.h"

#include "nigiri/timetable.h"

#include "cista/serialization.h"

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

constexpr int as_int(location_idx_t const d) { return static_cast<int>(d.v_); }
constexpr int as_int(day_idx_t const d) { return static_cast<int>(d.v_); }

template <direction SearchDir>
__device__ auto dir(auto a) {
  return (SearchDir == direction::kForward ? 1 : -1) * a;
}

raptor_state& raptor_state::resize(unsigned n_locations,
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
  return *this;
}

template <direction SearchDir, bool Rt, via_offset_t Vias>
__global__ void exec_raptor(
    timetable const* tt,
    std::uint32_t const n_locations,
    std::uint8_t const max_transfers,
    date::sys_days const base,
    unixtime_t const worst_time_at_dest,
    delta_t* const time_at_dest,
    device_flat_matrix_view<cuda::std::array<delta_t, Vias + 1>> round_times,
    cuda::std::span<cuda::std::array<delta_t, Vias + 1>>* best) {
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();

  auto const d_worst_at_dest = unix_to_delta(base, worst_time_at_dest);
  for (auto i = global_t_id; i < kMaxTransfers + 1U; i += global_stride) {
    time_at_dest[i] = get_best<SearchDir>(d_worst_at_dest, time_at_dest[i]);
  }

  auto const end_k = std::min(max_transfers, kMaxTransfers) + 1U;
  for (auto k = 1U; k != end_k; ++k) {
    // ==================
    // RAPTOR ROUND START
    // ------------------

    // Reuse best time from previous time at start (for range queries).
    for (auto i = global_t_id; i != n_locations; i += global_stride) {
      for (auto v = 0U; v != Vias + 1; ++v) {
        best[i][v] = get_best(round_times[k][i][v], best[i][v]);
      }
    }

    // ----------------
    // RAPTOR ROUND END
    // ================
  }
}

template <direction SearchDir, bool Rt, via_offset_t Vias>
void raptor<SearchDir, Rt, Vias>::execute(unixtime_t const start_time,
                                          std::uint8_t const max_transfers,
                                          unixtime_t const worst_time_at_dest,
                                          profile_idx_t const prf_idx,
                                          pareto_set<journey>& results){
    cudaLaunchCooperativeKernel()}

thrust::device_vector<std::uint8_t> copy_timetable(timetable const& tt) {
  auto const buf = cista::serialize<cista::mode::CAST>(tt);
  return {begin(buf), end(buf)};
}

}  // namespace nigiri::routing::gpu