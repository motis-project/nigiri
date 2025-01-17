#include "nigiri/routing/gpu/raptor.h"

#include <cstdio>
#include <iostream>

#include "cuda/std/array"
#include "cuda/std/span"

#include "cooperative_groups.h"

#include "thrust/device_vector.h"
#include "thrust/fill.h"
#include "thrust/host_vector.h"

#include "nigiri/routing/gpu/device_timetable.cuh"
#include "nigiri/routing/gpu/raptor_impl.cuh"
#include "nigiri/routing/gpu/types.cuh"

namespace cg = cooperative_groups;

namespace nigiri::routing::gpu {

#define CUDA_CHECK(code)                                       \
  utl_verify((code) == cudaSuccess, "CUDA error: {} at {}:{}", \
             cudaGetErrorString(code), __FILE__, __LINE__);

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
        route_transport_ranges_{to_device(tt.route_transport_ranges_)},
        route_clasz_{to_device(tt.route_clasz_)},
        route_bikes_allowed_{to_device(tt.route_bikes_allowed_)},
        route_bikes_allowed_per_section_{tt.route_bikes_allowed_per_section_},
        route_location_seq_{tt.route_location_seq_},
        location_routes_{tt.location_routes_},
        transport_traffic_days_{to_device(tt.transport_traffic_days_)},
        bitfields_{to_device(tt.bitfields_)},
        internal_interval_days_{tt.internal_interval_days()} {}

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
            .route_bikes_allowed_ = {to_view(route_bikes_allowed_)},
            .route_bikes_allowed_per_section_ =
                to_view(route_bikes_allowed_per_section_),
            .route_location_seq_ = to_view(route_location_seq_),
            .location_routes_ = to_view(location_routes_),
            .transport_traffic_days_ = to_view(transport_traffic_days_),
            .bitfields_ = to_view(bitfields_),
            .internal_interval_days_ = internal_interval_days_};
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
  thrust::device_vector<std::uint64_t> route_bikes_allowed_;
  device_vecvec<decltype(t{}.route_bikes_allowed_per_section_)>
      route_bikes_allowed_per_section_;

  device_vecvec<decltype(t{}.route_location_seq_)> route_location_seq_;
  device_vecvec<decltype(t{}.location_routes_)> location_routes_;

  thrust::device_vector<bitfield_idx_t> transport_traffic_days_;
  thrust::device_vector<bitfield> bitfields_;

  interval<date::sys_days> internal_interval_days_;
};

gpu_timetable::gpu_timetable(timetable const& tt)
    : impl_{std::make_unique<impl>(tt)} {}

gpu_timetable::~gpu_timetable() = default;

struct gpu_raptor_state::impl {
  explicit impl(gpu_timetable const& gtt)
      : tt_{gtt.impl_->to_device_timetable()} {}

  void resize(unsigned n_locations,
              unsigned n_routes,
              unsigned n_rt_transports,
              std::array<nigiri::bitvec, kMaxVias> const& is_via,
              std::vector<via_stop> const& via_stops,
              nigiri::bitvec const& is_dest,
              std::vector<std::uint16_t> const& dist_to_dest,
              std::vector<std::uint16_t> const& lb) {
    is_intermodal_dest_ = !dist_to_dest.empty();
    n_locations_ = host_state_.n_locations_ = n_locations;

    time_at_dest_.resize(kMaxTransfers + 1);

    tmp_storage_.resize(n_locations * (kMaxVias + 1));
    best_storage_.resize(n_locations * (kMaxVias + 1));
    round_times_storage_.resize(n_locations * (kMaxVias + 1) *
                                (kMaxTransfers + 1));
    host_round_times_.resize(round_times_storage_.size());
    station_mark_.resize(n_locations);
    prev_station_mark_.resize(n_locations);
    route_mark_.resize(n_routes);
    rt_transport_mark_.resize(n_rt_transports);
    end_reachable_.resize(n_locations);

    for (auto i = 0U; i != is_via.size(); ++i) {
      is_via_[i].resize(is_via[i].blocks_.size());
      thrust::copy(thrust::cuda::par_nosync, begin(is_via[i].blocks_),
                   end(is_via[i].blocks_), begin(is_via_[i]));
    }

    via_stops_.resize(via_stops.size());
    thrust::copy(thrust::cuda::par_nosync, begin(via_stops), end(via_stops),
                 begin(via_stops_));

    is_dest_.resize(is_dest.blocks_.size());
    utl::verify(
        cudaSuccess ==
            cudaMemcpy(thrust::raw_pointer_cast(is_dest_.data()),
                       is_dest.blocks_.data(),
                       is_dest.blocks_.size() *
                           sizeof(std::decay_t<decltype(is_dest)>::block_t),
                       cudaMemcpyHostToDevice),
        "could not copy is_dest bitvector");

    dist_to_dest_.resize(dist_to_dest.size());
    thrust::copy(thrust::cuda::par_nosync, begin(dist_to_dest),
                 end(dist_to_dest), begin(dist_to_dest_));

    lb_.resize(lb.size());
    utl::verify(
        cudaSuccess == cudaMemcpy(thrust::raw_pointer_cast(lb_.data()),
                                  lb.data(), lb.size() * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice),
        "could not copy lb bitvector");
  }

  template <via_offset_t Vias>
  cuda::std::span<cuda::std::array<delta_t, Vias + 1>> get_tmp() {
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

  unsigned n_locations_;
  bool is_intermodal_dest_;
  thrust::device_vector<delta_t> time_at_dest_;
  thrust::device_vector<delta_t> tmp_storage_;
  thrust::device_vector<delta_t> best_storage_;
  thrust::device_vector<delta_t> round_times_storage_;
  thrust::device_vector<std::uint32_t> station_mark_;
  thrust::device_vector<std::uint32_t> prev_station_mark_;
  thrust::device_vector<std::uint32_t> route_mark_;
  thrust::device_vector<std::uint32_t> rt_transport_mark_;

  thrust::device_vector<std::uint32_t> end_reachable_;
  thrust::device_vector<std::uint64_t> is_dest_;
  std::array<thrust::device_vector<std::uint64_t>, kMaxVias> is_via_;
  thrust::device_vector<via_stop> via_stops_;
  thrust::device_vector<std::uint16_t> dist_to_dest_;
  thrust::device_vector<std::uint16_t> lb_;

  device_timetable tt_;

  thrust::host_vector<delta_t> host_round_times_;
  raptor_state host_state_;
};

gpu_raptor_state::gpu_raptor_state(gpu_timetable const& gtt)
    : impl_{std::make_unique<impl>(gtt)} {}

gpu_raptor_state::~gpu_raptor_state() = default;

template <direction SearchDir, bool Rt, via_offset_t Vias>
gpu_raptor<SearchDir, Rt, Vias>::gpu_raptor(
    timetable const& tt,
    rt_timetable const* rtt,
    gpu_raptor_state& state,
    nigiri::bitvec& is_dest,
    std::array<nigiri::bitvec, kMaxVias> const& is_via,
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
      state_{state},
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
      transfer_time_settings_{tts} {
  state_.impl_->resize(tt.n_locations(), tt.n_routes(),
                       rtt ? rtt->n_rt_transports() : 0U, is_via, via_stops,
                       is_dest, dist_to_dest, lb);
  reset_arrivals();
}

template <direction SearchDir, bool Rt, via_offset_t Vias>
__global__ void exec_raptor(unixtime_t const start_time,
                            std::uint8_t const max_transfers,
                            unixtime_t const worst_time_at_dest,
                            raptor_impl<SearchDir, Rt, Vias> r) {
  r.execute(start_time, max_transfers, worst_time_at_dest);
}

template <direction SearchDir, bool Rt, via_offset_t Vias>
void gpu_raptor<SearchDir, Rt, Vias>::execute(
    unixtime_t const start_time,
    std::uint8_t const max_transfers,
    unixtime_t const worst_time_at_dest,
    profile_idx_t,
    pareto_set<journey>& results) {
  auto const starts =
      thrust::device_vector<std::pair<location_idx_t, unixtime_t>>{starts_};
  cudaDeviceSynchronize();
  CUDA_CHECK(cudaPeekAtLastError());

  auto& s = *state_.impl_;
  exec_raptor<SearchDir, Rt, Vias><<<1, 1>>>(
      start_time, max_transfers, worst_time_at_dest,
      raptor_impl<SearchDir, Rt, Vias>{
          .tt_ = s.tt_,
          .n_locations_ = s.tt_.n_locations_,
          .n_routes_ = s.tt_.n_routes_,
          .n_rt_transports_ = 0U,  // TODO
          .transfer_time_settings_ = transfer_time_settings_,
          .max_transfers_ = max_transfers,
          .allowed_claszes_ = allowed_claszes_,
          .require_bike_transport_ = require_bike_transport_,
          .base_ = base_,
          .worst_time_at_dest_ = worst_time_at_dest,
          .is_intermodal_dest_ = s.is_intermodal_dest_,
          .starts_ = to_view(starts),
          .is_via_ =
              [&]() {
                auto ret = cuda::std::array<device_bitvec<std::uint64_t const>,
                                            kMaxVias>{};
                for (auto i = 0U; i != kMaxVias; ++i) {
                  ret[i] = {.blocks_ = to_view(s.is_via_[i])};
                }
                return ret;
              }(),
          .via_stops_ = to_view(s.via_stops_),
          .is_dest_ = {to_view(s.is_dest_)},
          .end_reachable_ = {to_mutable_view(s.end_reachable_)},
          .dist_to_end_ = to_view(s.dist_to_dest_),
          .lb_ = to_view(s.lb_),
          .round_times_ = s.get_round_times<Vias>(),
          .best_ = s.get_best<Vias>(),
          .tmp_ = s.get_tmp<Vias>(),
          .time_at_dest_ = to_mutable_view(s.time_at_dest_),
          .station_mark_ = {to_mutable_view(s.station_mark_)},
          .prev_station_mark_ = {to_mutable_view(s.prev_station_mark_)},
          .route_mark_ = {to_mutable_view(s.route_mark_)}});

  cudaDeviceSynchronize();
  CUDA_CHECK(cudaPeekAtLastError());

  utl::verify(
      cudaSuccess ==
          cudaMemcpy(thrust::raw_pointer_cast(s.host_round_times_.data()),
                     thrust::raw_pointer_cast(s.round_times_storage_.data()),
                     s.round_times_storage_.size() * sizeof(delta_t),
                     cudaMemcpyDeviceToHost),
      "could not copy is_dest bitvector");

  sync_round_times();

  auto const round_times = s.host_state_.get_round_times<Vias>();
  auto const end_k = std::min(max_transfers, kMaxTransfers) + 1U;
  is_dest_.for_each_set_bit([&](auto const i) {
    for (auto k = 1U; k != end_k; ++k) {
      auto const dest_time = round_times[k][i][Vias];
      if (dest_time != kInvalid) {
        auto const [optimal, it, dominated_by] = results.add(
            journey{.legs_ = {},
                    .start_time_ = start_time,
                    .dest_time_ = delta_to_unix(base(), dest_time),
                    .dest_ = location_idx_t{i},
                    .transfers_ = static_cast<std::uint8_t>(k - 1)});
      }
    }
  });
}

template <direction SearchDir, bool Rt, via_offset_t Vias>
void gpu_raptor<SearchDir, Rt, Vias>::sync_round_times() {
  auto& s = *state_.impl_;
  s.host_state_.round_times_storage_.resize(s.host_round_times_.size());
  std::copy(begin(s.host_round_times_), end(s.host_round_times_),
            begin(s.host_state_.round_times_storage_));
}

template <direction SearchDir, bool Rt, via_offset_t Vias>
void gpu_raptor<SearchDir, Rt, Vias>::reset_arrivals() {
  thrust::fill(thrust::cuda::par_nosync, begin(state_.impl_->time_at_dest_),
               end(state_.impl_->time_at_dest_), kInvalid);
  thrust::fill(thrust::cuda::par_nosync,
               begin(state_.impl_->round_times_storage_),
               end(state_.impl_->round_times_storage_), kInvalid);
}

template <direction SearchDir, bool Rt, via_offset_t Vias>
void gpu_raptor<SearchDir, Rt, Vias>::next_start_time() {
  starts_.clear();
  thrust::fill(thrust::cuda::par_nosync, begin(state_.impl_->best_storage_),
               end(state_.impl_->best_storage_), kInvalid);
  thrust::fill(thrust::cuda::par_nosync, begin(state_.impl_->tmp_storage_),
               end(state_.impl_->tmp_storage_), kInvalid);
  thrust::fill(thrust::cuda::par_nosync,
               begin(state_.impl_->prev_station_mark_),
               end(state_.impl_->prev_station_mark_), 0U);
  thrust::fill(thrust::cuda::par_nosync, begin(state_.impl_->station_mark_),
               end(state_.impl_->station_mark_), 0U);
  thrust::fill(thrust::cuda::par_nosync, begin(state_.impl_->route_mark_),
               end(state_.impl_->route_mark_), 0U);
  if constexpr (Rt) {
    thrust::fill(thrust::cuda::par_nosync,
                 begin(state_.impl_->rt_transport_mark_),
                 end(state_.impl_->rt_transport_mark_), 0U);
  }
}

template <direction SearchDir, bool Rt, via_offset_t Vias>
void gpu_raptor<SearchDir, Rt, Vias>::reconstruct(query const& q, journey& j) {
  reconstruct_journey<SearchDir>(tt_, rtt_, q, state_.impl_->host_state_, j,
                                 base(), base_);
}

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