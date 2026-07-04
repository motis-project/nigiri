#include "nigiri/routing/gpu/raptor.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iostream>

#include "cuda/std/array"
#include "cuda/std/span"

#include "thrust/copy.h"
#include "thrust/device_vector.h"
#include "thrust/fill.h"
#include "thrust/host_vector.h"

#include "nigiri/for_each_meta.h"
#include "nigiri/routing/gpu/device_timetable.cuh"
#include "nigiri/routing/gpu/raptor_impl.cuh"
#include "nigiri/routing/gpu/types.cuh"
#include "utl/timer.h"

namespace nigiri::routing::gpu {

template <typename T>
struct pinned_buffer {
  pinned_buffer() = default;

  pinned_buffer(pinned_buffer const&) = delete;
  pinned_buffer& operator=(pinned_buffer const&) = delete;

  ~pinned_buffer() {
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

  T* ptr_ = nullptr;
  std::size_t cap_ = 0U;
};

#define CUDA_CHECK(code)                                              \
  if ((code) != cudaSuccess) {                                        \
    std::cerr << "CUDA error: " << cudaGetErrorString(code) << " at " \
              << __FILE__ << ":" << __LINE__;                         \
    std::terminate();                                                 \
  }

struct gpu_timetable::impl {
  using t = timetable;

  static std::vector<std::uint32_t> build_route_stop_offset(
      timetable const& tt) {
    auto v = std::vector<std::uint32_t>(tt.n_routes() + 1U);
    for (auto r = 0U; r < tt.n_routes(); ++r) {
      v[r + 1U] = v[r] + static_cast<std::uint32_t>(
                             tt.route_location_seq_[route_idx_t{r}].size());
    }
    return v;
  }

  static std::vector<std::uint32_t> build_route_of_stop(
      timetable const& tt, std::vector<std::uint32_t> const& off) {
    auto v = std::vector<std::uint32_t>(off.back());
    for (auto r = 0U; r < tt.n_routes(); ++r) {
      for (auto s = off[r]; s < off[r + 1U]; ++s) {
        v[s] = r;
      }
    }
    return v;
  }

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
        route_location_seq_{tt.route_location_seq_},
        location_routes_{tt.location_routes_},
        transport_traffic_days_{to_device(tt.transport_traffic_days_)},
        route_traffic_days_{to_device(tt.route_traffic_days_)},
        transport_route_{to_device(tt.transport_route_)},
        bitfields_{to_device(tt.bitfields_)},
        internal_interval_days_{tt.internal_interval_days()} {
    auto const off = build_route_stop_offset(tt);
    route_stop_offset_.assign(off.begin(), off.end());
    auto const ros = build_route_of_stop(tt, off);
    route_of_stop_.assign(ros.begin(), ros.end());
  }

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
            .route_traffic_days_ = to_view(route_traffic_days_),
            .transport_route_ = to_view(transport_route_),
            .bitfields_ = to_view(bitfields_),
            .route_stop_offset_ = to_view(route_stop_offset_),
            .route_of_stop_ = to_view(route_of_stop_),
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

  device_vecvec<decltype(t{}.route_location_seq_)> route_location_seq_;
  device_vecvec<decltype(t{}.location_routes_)> location_routes_;

  thrust::device_vector<bitfield_idx_t> transport_traffic_days_;
  thrust::device_vector<bitfield_idx_t> route_traffic_days_;
  thrust::device_vector<route_idx_t> transport_route_;
  thrust::device_vector<bitfield> bitfields_;
  thrust::device_vector<std::uint32_t> route_stop_offset_;
  thrust::device_vector<std::uint32_t> route_of_stop_;

  interval<date::sys_days> internal_interval_days_;
};

gpu_timetable::gpu_timetable(timetable const& tt)
    : impl_{std::make_unique<impl>(tt)} {}

gpu_timetable::~gpu_timetable() = default;

struct gpu_raptor_state::impl {
  explicit impl(gpu_timetable const& gtt)
      : tt_{gtt.impl_->to_device_timetable()} {
    cudaStreamCreate(&stream_);
    // compute_et boarding pre-pass buffers, sized to the flat route-stop space.
    auto const n_route_stops = tt_.route_of_stop_.size();
    et_result_.resize(n_route_stops);
    et_task_list_.resize(n_route_stops);
    et_task_count_.resize(1U);
    route_list_.resize(tt_.n_routes_);
    route_list_count_.resize(1U);
  }

  ~impl() { cudaStreamDestroy(stream_); }

  void resize(unsigned n_locations, unsigned n_routes) {
    time_at_dest_.resize(kMaxTransfers + 2);
    tmp_.resize(n_locations);
    best_.resize(n_locations);
    auto const first_time = round_times_.empty();
    round_times_.resize(n_locations * (kMaxTransfers + 2));
    if (first_time) {
      cudaMemsetAsync(thrust::raw_pointer_cast(round_times_.data()), 0xFF,
                      round_times_.size() * sizeof(std::uint64_t), stream_);
      cudaMemsetAsync(thrust::raw_pointer_cast(best_.data()), 0xFF,
                      best_.size() * sizeof(std::uint64_t), stream_);
      cudaMemsetAsync(thrust::raw_pointer_cast(tmp_.data()), 0xFF,
                      tmp_.size() * sizeof(std::uint64_t), stream_);
    }
    station_mark_.resize(n_locations / 32U + 1U);
    prev_station_mark_.resize(n_locations / 32U + 1U);
    route_mark_.resize(n_routes / 32U + 1U);
    any_marked_.resize(1U);
    done_.resize(1U);

    // is_dest + dist_to_dest they change between ping vs pong
    // -> handled in upload_query
  }

  // Uploads one direction's query data into its own slot. Ping and pong share
  // this state and their execute() calls interleave (pong_search runs
  // ping, pong*N, ping, pong*M, ...), so each direction keeps separate device
  // buffers and uploads exactly once (from the gpu_raptor ctor) instead of
  // re-uploading identical data on every execute().
  void upload_query(unsigned const dir,
                    nigiri::bitvec const& is_dest,
                    std::vector<std::uint16_t> const& dist_to_dest) {
    is_intermodal_dest_[dir] = !dist_to_dest.empty();

    is_dest_[dir].resize(is_dest.blocks_.size());
    utl::verify(
        cudaSuccess == cudaMemcpyAsync(
                           thrust::raw_pointer_cast(is_dest_[dir].data()),
                           is_dest.blocks_.data(),
                           is_dest.blocks_.size() *
                               sizeof(std::decay_t<decltype(is_dest)>::block_t),
                           cudaMemcpyHostToDevice, stream_),
        "could not copy is_dest");

    dist_to_dest_dev_[dir].resize(dist_to_dest.size());
    auto* const dd_pinned = dist_to_dest_[dir].ensure(dist_to_dest.size());
    std::copy(dist_to_dest.begin(), dist_to_dest.end(), dd_pinned);
    utl::verify(cudaSuccess ==
                    cudaMemcpyAsync(
                        thrust::raw_pointer_cast(dist_to_dest_dev_[dir].data()),
                        dd_pinned, dist_to_dest.size() * sizeof(std::uint16_t),
                        cudaMemcpyHostToDevice, stream_),
                "could not copy dist to dest");
  }

  bool is_intermodal_dest_[2];  // per direction: [0]=fwd, [1]=bwd
  thrust::device_vector<std::uint32_t> any_marked_;
  thrust::device_vector<std::uint32_t> done_;

  // 16bit time | 48bit breadcrumb
  thrust::device_vector<std::uint64_t> time_at_dest_;
  thrust::device_vector<std::uint64_t> tmp_;
  thrust::device_vector<std::uint64_t> best_;
  thrust::device_vector<std::uint64_t> round_times_;

  thrust::device_vector<std::uint32_t> station_mark_;
  thrust::device_vector<std::uint32_t> prev_station_mark_;
  thrust::device_vector<std::uint32_t> route_mark_;

  // per-direction query data: [0]=fwd, [1]=bwd (ping/pong interleave on the
  // shared state, each direction uploads its slot once in the ctor)
  thrust::device_vector<std::uint64_t> is_dest_[2];

  thrust::device_vector<std::uint16_t> dist_to_dest_dev_[2];

  pinned_buffer<std::uint16_t> dist_to_dest_[2];
  pinned_buffer<std::pair<location_idx_t, unixtime_t>> starts_;
  thrust::device_vector<std::pair<location_idx_t, unixtime_t>> starts_dev_;

  // reusable reconstruction buffers
  thrust::device_vector<location_idx_t> rec_dest_;
  thrust::device_vector<gpu_journey> rec_out_;
  thrust::host_vector<gpu_journey> rec_host_out_;

  device_timetable tt_;

  thrust::device_vector<std::uint32_t> et_result_;
  thrust::device_vector<std::uint32_t> et_task_list_;
  thrust::device_vector<std::uint32_t> et_task_count_;
  thrust::device_vector<std::uint32_t> route_list_;
  thrust::device_vector<std::uint32_t> route_list_count_;

  cudaStream_t stream_;
};

gpu_raptor_state::gpu_raptor_state(gpu_timetable const& gtt)
    : impl_{std::make_unique<impl>(gtt)} {}

gpu_raptor_state::~gpu_raptor_state() = default;

template <direction SearchDir>
gpu_raptor<SearchDir>::gpu_raptor(
    timetable const& tt,
    rt_timetable const* /* rtt (GPU: no realtime) */,
    gpu_raptor_state& state,
    nigiri::bitvec& is_dest,
    std::array<nigiri::bitvec, kMaxVias> const& /* is_via (GPU: no vias) */,
    std::vector<std::uint16_t> const& dist_to_dest,
    hash_map<location_idx_t,
             std::vector<td_offset>> const& /* td_dist_to_dest (GPU: no td) */,
    std::vector<std::uint16_t> const& /* lb (GPU: no lower bounds) */,
    std::vector<via_stop> const& via_stops,
    day_idx_t const base,
    clasz_mask_t const allowed_claszes,
    bool const require_bike_transport,
    bool const require_car_transport,
    bool const is_wheelchair,
    transfer_time_settings const& tts)
    : tt_{tt},
      n_locations_{tt_.n_locations()},
      state_{state},
      is_dest_{is_dest},
      base_{base},
      allowed_claszes_{allowed_claszes},
      transfer_time_settings_{tts} {
  state_.impl_->resize(tt.n_locations(), tt.n_routes());
  reset_arrivals();
  state_.impl_->upload_query(kDirIdx, is_dest, dist_to_dest);
}

template <direction SearchDir>
__global__ void init_arrivals_kernel(raptor_impl<SearchDir> r,
                                     unixtime_t const worst_time_at_dest) {
  r.init_arrivals(worst_time_at_dest);
}

template <direction SearchDir>
__global__ void reuse_previous_arrivals_kernel(raptor_impl<SearchDir> r,
                                               unsigned const k) {
  if (*r.done_) {
    return;
  }
  r.reuse_previous_arrivals(k);
}

template <direction SearchDir>
__global__ void mark_routes_kernel(raptor_impl<SearchDir> r, unsigned const k) {
  if (*r.done_) {
    return;
  }
  r.mark_routes(k);
}

template <direction SearchDir>
__global__ void begin_transit_phase_kernel(raptor_impl<SearchDir> r) {
  if (*r.done_) {
    return;
  }
  if (*r.any_marked_ == 0U) {  // no route marked -> search converged
    if (get_global_thread_id() == 0U) {
      *r.done_ = 1U;
    }
    return;
  }
  r.begin_transit_phase();
}

template <direction SearchDir>
__global__ void et_build_route_list_kernel(raptor_impl<SearchDir> r) {
  if (*r.done_) {
    return;
  }
  r.et_build_route_list();
}

template <direction SearchDir>
__global__ void et_collect_tasks_kernel(raptor_impl<SearchDir> r,
                                        unsigned const k) {
  if (*r.done_) {
    return;
  }
  r.et_collect_tasks(k);
}

template <direction SearchDir>
__global__ void et_run_lookups_kernel(raptor_impl<SearchDir> r,
                                      unsigned const k) {
  if (*r.done_) {
    return;
  }
  r.et_run_lookups(k);
}

template <direction SearchDir, bool WithClaszFilter>
__global__ void loop_routes_kernel(raptor_impl<SearchDir> r, unsigned const k) {
  if (*r.done_) {
    return;
  }
  r.template loop_routes<WithClaszFilter>(k);
}

template <direction SearchDir>
__global__ void begin_footpath_phase_kernel(raptor_impl<SearchDir> r) {
  if (*r.done_) {
    return;
  }
  if (*r.any_marked_ == 0U) {  // no location improved -> search converged
    if (get_global_thread_id() == 0U) {
      *r.done_ = 1U;
    }
    return;
  }
  r.begin_footpath_phase();
}

template <direction SearchDir>
__global__ void transfers_footpaths_kernel(raptor_impl<SearchDir> r,
                                           unsigned const k) {
  if (*r.done_) {
    return;
  }
  r.update_transfers_and_footpaths(k);
  r.route_mark_.reset();
}

template <typename Kernel>
std::pair<int, int> launch_dims(Kernel kernel) {
  static auto const dims = [&]() {
    auto blocks = 0;
    auto threads = 0;
    // half + quarter benchmarked with less throughput
    cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, kernel, 0, 0);
    return std::pair{blocks, threads};
  }();
  return dims;
}

template <typename Kernel, typename... Args>
void launch(Kernel kernel, cudaStream_t stream, Args&&... args) {
  auto const [blocks, threads] = launch_dims(kernel);
  kernel<<<blocks, threads, 0, stream>>>(std::forward<Args>(args)...);
}

template <direction SearchDir>
__global__ void reconstruct_kernel(location_idx_t const* const dest_list,
                                   std::uint32_t const n_dest,
                                   std::uint32_t const end_k,
                                   raptor_impl<SearchDir> r,
                                   gpu_journey* const out) {
  auto const tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_dest * end_k) {
    return;
  }
  out[tid].valid_ = 0U;
  auto const k = tid % end_k;
  if (k == 0U) {
    return;
  }
  r.reconstruct_journey(dest_list[tid / end_k], k, &out[tid]);
}

template <direction SearchDir>
void gpu_raptor<SearchDir>::execute(unixtime_t start_time,
                                    std::uint8_t max_transfers,
                                    unixtime_t worst_time_at_dest,
                                    profile_idx_t,
                                    pareto_set<journey>& results) {
  auto& s = *state_.impl_;

  // Copy starts.
  auto* const starts_pinned = s.starts_.ensure(starts_.size());
  std::memset(starts_pinned, 0,  // otherwise initcheck flags
              starts_.size() * sizeof(std::pair<location_idx_t, unixtime_t>));
  std::copy(starts_.begin(), starts_.end(), starts_pinned);
  if (s.starts_dev_.size() < starts_.size()) {
    s.starts_dev_.resize(starts_.size());
  }
  cudaMemcpyAsync(
      thrust::raw_pointer_cast(s.starts_dev_.data()), starts_pinned,
      starts_.size() * sizeof(std::pair<location_idx_t, unixtime_t>),
      cudaMemcpyHostToDevice, s.stream_);
  auto const starts =
      cuda::std::span<std::pair<location_idx_t, unixtime_t> const>{
          thrust::raw_pointer_cast(s.starts_dev_.data()), starts_.size()};

  cudaStreamSynchronize(s.stream_);
  CUDA_CHECK(cudaPeekAtLastError());

  auto r = raptor_impl<SearchDir>{
      .any_marked_ = thrust::raw_pointer_cast(s.any_marked_.data()),
      .done_ = thrust::raw_pointer_cast(s.done_.data()),
      .tt_ = s.tt_,
      .transfer_time_settings_ = transfer_time_settings_,
      .max_transfers_ = max_transfers,
      .allowed_claszes_ = allowed_claszes_,
      .base_ = base_,
      .starts_ = starts,
      .is_dest_ = {to_view(s.is_dest_[kDirIdx])},
      .dist_to_end_ = to_view(s.dist_to_dest_dev_[kDirIdx]),
      .round_times_ = {to_mutable_view(s.round_times_), n_locations_},
      .best_ = {to_mutable_view(s.best_), n_locations_},
      .tmp_ = {to_mutable_view(s.tmp_), n_locations_},
      .time_at_dest_ = {to_mutable_view(s.time_at_dest_), n_locations_},
      .station_mark_ = {to_mutable_view(s.station_mark_)},
      .prev_station_mark_ = {to_mutable_view(s.prev_station_mark_)},
      .route_mark_ = {to_mutable_view(s.route_mark_)},
      .et_result_ = to_mutable_view(s.et_result_),
      .et_task_list_ = to_mutable_view(s.et_task_list_),
      .et_task_count_ = thrust::raw_pointer_cast(s.et_task_count_.data()),
      .route_list_ = to_mutable_view(s.route_list_),
      .route_list_count_ =
          thrust::raw_pointer_cast(s.route_list_count_.data())};

  auto const end_k =
      static_cast<std::uint32_t>(std::min(max_transfers, kMaxTransfers) + 2U);

  launch(init_arrivals_kernel<SearchDir>, s.stream_, r, worst_time_at_dest);
  for (auto k = 1U; k != end_k; ++k) {
    launch(reuse_previous_arrivals_kernel<SearchDir>, s.stream_, r, k);
    launch(mark_routes_kernel<SearchDir>, s.stream_, r, k);
    launch(begin_transit_phase_kernel<SearchDir>, s.stream_, r);
    launch(et_build_route_list_kernel<SearchDir>, s.stream_, r);
    launch(et_collect_tasks_kernel<SearchDir>, s.stream_, r, k);
    launch(et_run_lookups_kernel<SearchDir>, s.stream_, r, k);
    if (allowed_claszes_ == all_clasz_allowed()) {
      launch(loop_routes_kernel<SearchDir, false>, s.stream_, r, k);
    } else {
      launch(loop_routes_kernel<SearchDir, true>, s.stream_, r, k);
    }
    launch(begin_footpath_phase_kernel<SearchDir>, s.stream_, r);
    launch(transfers_footpaths_kernel<SearchDir>, s.stream_, r, k);
  }
  cudaStreamSynchronize(s.stream_);
  CUDA_CHECK(cudaPeekAtLastError());

  auto dest_list = std::vector<location_idx_t>{};
  if (s.is_intermodal_dest_[kDirIdx]) {
    dest_list.push_back(get_special_station(special_station::kEnd));
  } else {
    is_dest_.for_each_set_bit(
        [&](auto const i) { dest_list.push_back(location_idx_t{i}); });
  }
  if (dest_list.empty()) {
    return;
  }

  auto const n_dest = static_cast<std::uint32_t>(dest_list.size());
  auto const total = n_dest * end_k;

  s.rec_dest_ = dest_list;
  if (s.rec_out_.size() < total) {
    s.rec_out_.resize(total);
  }

  {
    auto const threads = 128U;
    auto const blocks = (total + threads - 1U) / threads;
    reconstruct_kernel<SearchDir><<<blocks, threads, 0, s.stream_>>>(
        thrust::raw_pointer_cast(s.rec_dest_.data()), n_dest, end_k, r,
        thrust::raw_pointer_cast(s.rec_out_.data()));
    cudaStreamSynchronize(s.stream_);
  }
  CUDA_CHECK(cudaPeekAtLastError());

  if (s.rec_host_out_.size() < total) {
    s.rec_host_out_.resize(total);
  }

  thrust::copy(s.rec_out_.begin(), s.rec_out_.begin() + total,
               s.rec_host_out_.begin());
  for (auto idx = std::uint32_t{0U}; idx != total; ++idx) {
    auto const& gj = s.rec_host_out_[idx];
    if (gj.valid_ == 0U || gj.n_legs_ == 0U) {
      continue;
    }
    auto j = journey{};
    j.start_time_ = start_time;
    j.dest_time_ = delta_to_unix(base(), gj.dest_time_);
    j.dest_ = location_idx_t{gj.dest_l_};
    j.transfers_ = gj.transfers_;

    // POD legs are emitted in:
    // - reverse-chronological for forward
    // - chronological for backward
    //
    // Emit the journey in chronological order accordingly.
    for (auto li2 = 0U; li2 != gj.n_legs_; ++li2) {
      auto const li = (SearchDir == direction::kForward)
                          ? (static_cast<unsigned>(gj.n_legs_) - 1U - li2)
                          : li2;
      auto const& gl = gj.legs_[li];
      auto const from = location_idx_t{gl.from_l_};
      auto const to = location_idx_t{gl.to_l_};
      auto const dep = delta_to_unix(base(), gl.dep_);
      auto const arr = delta_to_unix(base(), gl.arr_);

      if (gl.is_footpath_ != 0U) {
        j.legs_.emplace_back(journey::leg{
            SearchDir, from, to, dep, arr,
            footpath{to, duration_t{
                             static_cast<duration_t::rep>(gl.fp_duration_)}}});
      } else {
        auto const t_idx = transport_idx_t{gl.transport_};
        auto const route = tt_.transport_route_[t_idx];
        auto const route_len =
            static_cast<stop_idx_t>(tt_.route_location_seq_[route].size());
        auto const run = rt::run{
            .t_ = transport{t_idx, day_idx_t{gl.day_}},
            .stop_range_ = interval<stop_idx_t>{stop_idx_t{0}, route_len}};
        j.legs_.emplace_back(journey::leg{
            SearchDir, from, to, dep, arr,
            journey::run_enter_exit{run, gl.enter_stop_, gl.exit_stop_}});
      }
    }

    // Re-anchor transfer/footpath legs to the preceding leg's arrival.
    for (auto i = std::size_t{1U}; i < j.legs_.size(); ++i) {
      if (std::holds_alternative<footpath>(j.legs_[i].uses_)) {
        auto const dur = std::get<footpath>(j.legs_[i].uses_).duration();
        j.legs_[i].dep_time_ = j.legs_[i - 1U].arr_time_;
        j.legs_[i].arr_time_ = j.legs_[i].dep_time_ + dur;
      }
    }

    // is_reconstructed_ stays false
    // -> only reconstruct() might set it to true
    results.add(std::move(j));
  }
}

template <direction SearchDir>
void gpu_raptor<SearchDir>::reset_arrivals() {
  auto& s = *state_.impl_;
  cudaMemsetAsync(thrust::raw_pointer_cast(s.time_at_dest_.data()), 0xFF,
                  s.time_at_dest_.size() * sizeof(std::uint64_t), s.stream_);
  cudaMemsetAsync(thrust::raw_pointer_cast(s.round_times_.data()), 0xFF,
                  s.round_times_.size() * sizeof(std::uint64_t), s.stream_);
}

template <direction SearchDir>
void gpu_raptor<SearchDir>::next_start_time() {
  starts_.clear();
  auto& s = *state_.impl_;
  cudaMemsetAsync(thrust::raw_pointer_cast(s.best_.data()), 0xFF,
                  s.best_.size() * sizeof(std::uint64_t), s.stream_);
  cudaMemsetAsync(thrust::raw_pointer_cast(s.tmp_.data()), 0xFF,
                  s.tmp_.size() * sizeof(std::uint64_t), s.stream_);
  thrust::fill(thrust::cuda::par.on(state_.impl_->stream_),
               begin(state_.impl_->prev_station_mark_),
               end(state_.impl_->prev_station_mark_), 0U);
  thrust::fill(thrust::cuda::par.on(state_.impl_->stream_),
               begin(state_.impl_->station_mark_),
               end(state_.impl_->station_mark_), 0U);
  thrust::fill(thrust::cuda::par.on(state_.impl_->stream_),
               begin(state_.impl_->route_mark_), end(state_.impl_->route_mark_),
               0U);
}

template <direction SearchDir>
void gpu_raptor<SearchDir>::reconstruct(query const& q, journey& j) {
  // The core legs (boarding station -> alighting station) were filled on the
  // GPU inside execute(). For intermodal queries the first/last mile mumo legs
  // to the special kStart/kEnd stations are added here on the host, where the
  // query offsets (and their transport_mode_id_) live. Mirrors the intermodal
  // branches of the CPU reconstruct_journey<SearchDir>.
  //
  // The journey's legs are chronological. Depending on the search direction the
  // start-seed end and the intermodal-egress end map to front/back:
  //   forward:  front = start side (q.start_/kStart),
  //             back  = dest side  (q.destination_/kEnd)
  //   backward: front = dest side  (q.destination_/kEnd),
  //             back  = start side (q.start_/kStart)
  constexpr auto const is_fwd = SearchDir == direction::kForward;
  if (!j.legs_.empty()) {
    auto const find_offset = [&](std::vector<offset> const& offsets,
                                 location_match_mode const mode,
                                 location_idx_t const l) -> offset const* {
      for (auto const& o : offsets) {
        if (matches(tt_, mode, o.target(), l)) {
          return &o;
        }
      }
      return nullptr;
    };

    // front-side mumo leg: special_station -> first real station
    auto const front_l = j.legs_.front().from_;
    auto const front_t = j.legs_.front().dep_time_;
    auto const front_mode = is_fwd ? q.start_match_mode_ : q.dest_match_mode_;
    if (front_mode == location_match_mode::kIntermodal) {
      auto const& offsets = is_fwd ? q.start_ : q.destination_;
      if (auto const* o = find_offset(offsets, front_mode, front_l)) {
        auto const special = get_special_station(
            is_fwd ? special_station::kStart : special_station::kEnd);
        auto const dep = front_t - o->duration();
        // direction::kForward used as a raw setter (from=a, to=b, dep, arr).
        j.legs_.insert(begin(j.legs_),
                       journey::leg{direction::kForward, special, front_l, dep,
                                    front_t, *o});
      }
    }

    // back-side mumo leg: last real station -> special_station
    auto const back_l = j.legs_.back().to_;
    auto const back_t = j.legs_.back().arr_time_;
    auto const back_mode = is_fwd ? q.dest_match_mode_ : q.start_match_mode_;
    if (back_mode == location_match_mode::kIntermodal) {
      auto const& offsets = is_fwd ? q.destination_ : q.start_;
      if (auto const* o = find_offset(offsets, back_mode, back_l)) {
        auto const special = get_special_station(
            is_fwd ? special_station::kEnd : special_station::kStart);
        auto const arr = back_t + o->duration();
        j.legs_.push_back(journey::leg{direction::kForward, back_l, special,
                                       back_t, arr, *o});
        j.dest_ = special;
      }
    }
  }

  optimize_footpaths(tt_, nullptr, q, j);

  j.is_reconstructed_ = true;
}

template <direction SearchDir>
void gpu_raptor<SearchDir>::add_start(location_idx_t const l,
                                      unixtime_t const t) {
  starts_.emplace_back(l, t);
}

template class gpu_raptor<direction::kForward>;
template class gpu_raptor<direction::kBackward>;

}  // namespace nigiri::routing::gpu