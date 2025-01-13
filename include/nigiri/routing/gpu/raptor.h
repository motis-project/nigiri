#pragma once

#include <cuda/std/array>
#include <cuda/std/span>

#include "thrust/device_vector.h"
#include "thrust/fill.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/common/flat_matrix_view.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

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

template <typename T, std::size_t N>
cuda::std::array<T, N> to_device(std::array<T, N> const& a) {
  auto ret = cuda::std::array<T, N>{};
  for (auto i = 0U; i != N; ++i) {
    ret[i] = to_device(a[i]);
  }
  return ret;
}

struct raptor_state {
  raptor_state& resize(unsigned n_locations,
                       unsigned n_routes,
                       unsigned n_rt_transports);

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
  device_vec<delta_t> tmp_storage_{};
  device_vec<delta_t> best_storage_{};
  device_vec<delta_t> round_times_storage_{};
  device_bitvec station_mark_{};
  device_bitvec prev_station_mark_{};
  device_bitvec route_mark_{};
  device_bitvec rt_transport_mark_{};
  device_bitvec end_reachable_{};

  thrust::device_vector<std::uint8_t> const& gpu_tt_;
};

template <direction SearchDir, bool Rt, via_offset_t Vias>
struct raptor {
  static constexpr auto const kInvalid = kInvalidDelta<SearchDir>;
  static constexpr auto const kInvalidArray = []() {
    auto a = std::array<delta_t, Vias + 1>{};
    a.fill(kInvalid);
    return a;
  }();

  raptor(timetable const& tt,
         rt_timetable const* rtt,
         raptor_state& state,
         bitvec& is_dest,
         std::array<bitvec, kMaxVias>& is_via,
         std::vector<std::uint16_t>& dist_to_dest,
         hash_map<location_idx_t, std::vector<td_offset>> const&,
         std::vector<std::uint16_t>& lb,
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
        tmp_{state_.get_tmp<Vias>()},
        best_{state_.get_best<Vias>()},
        round_times_{state.get_round_times<Vias>()},
        is_dest_{to_device(is_dest)},
        is_via_{to_device(is_via)},
        dist_to_end_{dist_to_dest},
        lb_{lb},
        via_stops_{via_stops},
        base_{base},
        allowed_claszes_{allowed_claszes},
        require_bike_transport_{require_bike_transport},
        is_wheelchair_{is_wheelchair},
        transfer_time_settings_{tts} {}

  raptor_stats get_stats() const { return stats_; }

  void reset_arrivals() {
    thrust::fill(thrust::cuda::par_nosync, time_at_dest_, kInvalid);
    thrust::fill(thrust::cuda::par_nosync, round_times_.entries_,
                 kInvalidArray);
    cudaDeviceSynchronize();
  }

  void next_start_time() {
    starts_.clear();

    thrust::fill(thrust::cuda::par_nosync, begin(best_), end(best_),
                 kInvalidArray);
    thrust::fill(thrust::cuda::par_nosync, begin(tmp_), end(tmp_),
                 kInvalidArray);
    thrust::fill(thrust::cuda::par_nosync,
                 begin(state_.prev_station_mark_.blocks_),
                 end(state_.prev_station_mark_.blocks_), 0U);
    thrust::fill(thrust::cuda::par_nosync, begin(state_.station_mark_.blocks_),
                 end(state_.station_mark_.blocks_), 0U);
    thrust::fill(thrust::cuda::par_nosync, begin(state_.route_mark_.blocks_),
                 end(state_.route_mark_.blocks_), 0U);
    if constexpr (Rt) {
      thrust::fill(thrust::cuda::par_nosync,
                   begin(state_.rt_transport_mark_.blocks_),
                   end(state_.rt_transport_mark_.blocks_), 0U);
    }
    cudaDeviceSynchronize();
  }

  void add_start(location_idx_t const l, unixtime_t const t) {
    starts_.emplace_back(l, t);
  }

  void execute(unixtime_t const start_time,
               std::uint8_t const max_transfers,
               unixtime_t const worst_time_at_dest,
               profile_idx_t const prf_idx,
               pareto_set<journey>& results);

private:
  date::sys_days base() const {
    return tt_.internal_interval_days().from_ + as_int(base_) * date::days{1};
  }

  int as_int(day_idx_t const d) const { return static_cast<int>(d.v_); }

  timetable const& tt_;
  rt_timetable const* rtt_{nullptr};
  int n_days_;
  std::uint32_t n_locations_, n_routes_, n_rt_transports_;
  raptor_state& state_;
  cuda::std::span<cuda::std::array<delta_t, Vias + 1>> tmp_;
  cuda::std::span<cuda::std::array<delta_t, Vias + 1>> best_;
  device_flat_matrix_view<cuda::std::array<delta_t, Vias + 1>> round_times_;
  device_bitvec is_dest_;
  cuda::std::array<bitvec, kMaxVias> is_via_;
  device_vec<std::uint16_t> dist_to_end_;
  device_vec<std::uint16_t> lb_;
  device_vec<via_stop> via_stops_;
  cuda::std::array<delta_t, kMaxTransfers + 1> time_at_dest_;
  day_idx_t base_;
  raptor_stats stats_;
  clasz_mask_t allowed_claszes_;
  bool require_bike_transport_;
  bool is_wheelchair_;
  transfer_time_settings transfer_time_settings_;

  std::vector<std::pair<location_idx_t, unixtime_t>> starts_;
};

thrust::device_vector<std::uint8_t> copy_timetable(timetable const&);

}  // namespace nigiri::routing::gpu