#include "nigiri/routing/gpu/mcraptor.h"

#include <cstring>
#include <algorithm>
#include <optional>
#include <unordered_map>

// hide date.h's NOEXCEPT from CCCL's token pasting, see device_times.h
#pragma push_macro("NOEXCEPT")
#undef NOEXCEPT
#include "cuda/std/span"
#pragma pop_macro("NOEXCEPT")

#include "thrust/device_vector.h"
#include "thrust/functional.h"
#include "thrust/reduce.h"
#include "thrust/transform.h"
#include "thrust/transform_reduce.h"

#include "utl/helpers/algorithm.h"
#include "utl/verify.h"

#include "nigiri/for_each_meta.h"
#include "nigiri/logging.h"
#include "nigiri/routing/gpu/cuda_check.cuh"
#include "nigiri/routing/gpu/device_buffer.cuh"
#include "nigiri/routing/gpu/mcraptor_impl.cuh"
#include "nigiri/routing/gpu/pinned_host_buffer.cuh"
#include "nigiri/routing/gpu/timetable_impl.cuh"
#include "nigiri/routing/raptor/mcraptor.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/special_stations.h"
#include "nigiri/td_footpath.h"

namespace nigiri::routing::gpu {

struct gpu_mcraptor_state::impl {
  explicit impl(gpu_timetable const& gtt)
      : tt_{gtt.impl_->to_device_timetable()} {
    cudaStreamCreate(&stream_);

    auto const n_locations = tt_.n_locations_;
    auto const checked_cap = [](char const* what, std::size_t const x,
                                std::uint32_t const max) {
      utl::verify(x >= 4U && x <= max,
                  "gpu mcraptor: {} cap {} out of range "
                  "[4, {}]",
                  what, x, max);
      return static_cast<std::uint32_t>(x);
    };
    // inline slots per stop; the hwm byte bounds inline + block chain
    bag_cap_ = checked_cap("bag", kMcBagCapDefault, 255U - kMcBagBlock);
    bags_.resize(static_cast<std::size_t>(n_locations) * bag_cap_);
    cudaMemsetAsync(thrust::raw_pointer_cast(bags_.data()), 0xFF,
                    bags_.size() * sizeof(mc_label_t), stream_);
    // chained blocks: the pool scales with the labels, not a worst-case row
    bag_pool_cap_ = checked_cap(
        "bag pool",
        std::min<std::size_t>(std::max<std::size_t>(n_locations / 2U, 524'288U),
                              kMcBagPoolDefault * 4U),
        1U << 25U);
    bag_pool_.resize(static_cast<std::size_t>(bag_pool_cap_) * kMcBagBlock);
    cudaMemsetAsync(thrust::raw_pointer_cast(bag_pool_.data()), 0xFF,
                    bag_pool_.size() * sizeof(mc_label_t), stream_);
    bag_next_.resize(bag_pool_cap_);
    bag_ovf_.resize(n_locations);
    cudaMemsetAsync(thrust::raw_pointer_cast(bag_ovf_.data()), 0xFF,
                    bag_ovf_.size() * sizeof(std::uint32_t), stream_);
    bag_pool_count_.resize(1U);
    cudaMemsetAsync(thrust::raw_pointer_cast(bag_pool_count_.data()), 0,
                    sizeof(std::uint32_t), stream_);
    for (auto d = 0U; d != 2U; ++d) {
      reuse_bags_[d].resize(static_cast<std::size_t>(n_locations) *
                            kMcReuseCap);
      cudaMemsetAsync(thrust::raw_pointer_cast(reuse_bags_[d].data()), 0xFF,
                      reuse_bags_[d].size() * sizeof(std::uint64_t), stream_);
    }
    bag_locks_.resize(n_locations);
    cudaMemsetAsync(thrust::raw_pointer_cast(bag_locks_.data()), 0,
                    bag_locks_.size() * sizeof(std::uint32_t), stream_);
    bag_hwm_.resize(n_locations);
    cudaMemsetAsync(thrust::raw_pointer_cast(bag_hwm_.data()), 0,
                    bag_hwm_.size() * sizeof(std::uint8_t), stream_);
    for (auto d = 0U; d != 2U; ++d) {
      dest_bag_[d].resize(kMcDestCap);
      dest_best_key_[d].resize(kMaxTransfers + 2U);
      dest_best_total_[d].resize(kMaxTransfers + 2U);
      dest_lock_[d].resize(1U);
      cudaMemsetAsync(thrust::raw_pointer_cast(dest_lock_[d].data()), 0,
                      sizeof(std::uint32_t), stream_);
    }

    // breadcrumbs of one start time; overflow trips the canary
    auto const arena_cap = std::min<std::size_t>(
        std::max<std::size_t>(static_cast<std::size_t>(n_locations) * 16U,
                              8'000'000U),
        32'000'000U);
    bc_.resize(arena_cap);
    bc_count_.resize(1U);

    touched_.resize(n_bitvec_words(n_locations));
    station_mark_.resize(n_bitvec_words(n_locations));
    prev_station_mark_.resize(n_bitvec_words(n_locations));
    route_mark_.resize(n_bitvec_words(tt_.n_routes_));
    thrust::fill(thrust::cuda::par.on(stream_), touched_.begin(),
                 touched_.end(), 0U);
    thrust::fill(thrust::cuda::par.on(stream_), station_mark_.begin(),
                 station_mark_.end(), 0U);
    thrust::fill(thrust::cuda::par.on(stream_), prev_station_mark_.begin(),
                 prev_station_mark_.end(), 0U);
    thrust::fill(thrust::cuda::par.on(stream_), route_mark_.begin(),
                 route_mark_.end(), 0U);

    route_list_.resize(tt_.n_routes_);
    route_list_count_.resize(1U);
    route_entry_count_.resize(tt_.n_routes_);
    route_single_entry_.resize(tt_.n_routes_);
    route_single_flat_.resize(tt_.n_routes_);
    auto const n_route_stops = tt_.route_of_stop_.size();
    // et tasks are bounded by the flat route-stops; the entry pool is reserved
    // per task at collect time. The clamp keeps tiny timetables above the
    // floor.
    et_tasks_cap_ = checked_cap(
        "et tasks", std::clamp<std::size_t>(n_route_stops, 4U, 64'000'000U),
        1U << 28U);
    et_task_list_.resize(et_tasks_cap_);
    et_task_off_.resize(et_tasks_cap_);
    et_task_cnt_.resize(et_tasks_cap_);
    et_task_count_.resize(1U);
    et_entry_count_.resize(1U);
    et_pool_cap_ =
        checked_cap("et pool",
                    std::min<std::size_t>(
                        std::max<std::size_t>(8U * n_route_stops, 64'000'000U),
                        224'000'000U),
                    1U << 30U);
    et_ent_key_.resize(et_pool_cap_);
    et_ent_ex_.resize(et_pool_cap_);
    et_ent_sl_.resize(et_pool_cap_);
    task_bits_.resize((n_route_stops >> 5U) + 2U);
    cudaMemsetAsync(thrust::raw_pointer_cast(task_bits_.data()), 0,
                    task_bits_.size() * sizeof(std::uint32_t), stream_);
    route_task_start_.resize(tt_.n_routes_);
    any_marked_.resize(1U);
    done_.resize(1U);
    overflow_.resize(1U);
    cudaMemsetAsync(thrust::raw_pointer_cast(overflow_.data()), 0,
                    sizeof(std::uint32_t), stream_);
  }

  ~impl() { cudaStreamDestroy(stream_); }

  // per query: the rt timetable may be absent or have grown
  void resize_rt(unsigned const n_rt_transports) {
    rt_transport_mark_.resize(n_bitvec_words(n_rt_transports));
  }

  void upload_query(
      unsigned const dir,
      nigiri::bitvec const& is_dest,
      std::vector<std::uint16_t> const& dist_to_dest,
      hash_map<location_idx_t, std::vector<td_offset>> const& td_dist_to_dest,
      std::vector<std::uint16_t> const& lb) {
    is_intermodal_dest_[dir] = !dist_to_dest.empty();

    // td dest offsets as sorted (loc, range, data) groups, as in gpu raptor
    if (td_dist_to_dest.empty()) {
      td_dest_locs_dev_[dir].clear();
      td_dest_ranges_dev_[dir].clear();
      td_dest_data_dev_[dir].clear();
    } else {
      auto locs = std::vector<location_idx_t>{};
      locs.reserve(td_dist_to_dest.size());
      for (auto const& [l, _] : td_dist_to_dest) {
        locs.push_back(l);
      }
      utl::sort(locs);  // hash_map order is not deterministic
      auto ranges = std::vector<std::uint32_t>{0U};
      auto td_data = std::vector<td_offset>{};
      for (auto const l : locs) {
        auto const& v = td_dist_to_dest.at(l);
        td_data.insert(end(td_data), begin(v), end(v));
        ranges.push_back(static_cast<std::uint32_t>(td_data.size()));
      }
      auto* const locs_pin = td_dest_locs_pin_[dir].ensure(locs.size());
      std::copy(locs.begin(), locs.end(), locs_pin);
      auto* const ranges_pin = td_dest_ranges_pin_[dir].ensure(ranges.size());
      std::copy(ranges.begin(), ranges.end(), ranges_pin);
      auto* const data_pin = td_dest_data_pin_[dir].ensure(td_data.size());
      std::copy(td_data.begin(), td_data.end(), data_pin);
      utl::verify(
          cudaSuccess ==
                  cudaMemcpyAsync(
                      td_dest_locs_dev_[dir].ensure(locs.size(), stream_),
                      locs_pin, locs.size() * sizeof(location_idx_t),
                      cudaMemcpyHostToDevice, stream_) &&
              cudaSuccess ==
                  cudaMemcpyAsync(
                      td_dest_ranges_dev_[dir].ensure(ranges.size(), stream_),
                      ranges_pin, ranges.size() * sizeof(std::uint32_t),
                      cudaMemcpyHostToDevice, stream_) &&
              cudaSuccess ==
                  cudaMemcpyAsync(
                      td_dest_data_dev_[dir].ensure(td_data.size(), stream_),
                      data_pin, td_data.size() * sizeof(td_offset),
                      cudaMemcpyHostToDevice, stream_),
          "gpu mcraptor: could not copy td dest offsets");
    }

    is_dest_[dir].resize(is_dest.blocks_.size());
    auto* const is_dest_pin = is_dest_pin_[dir].ensure(is_dest.blocks_.size());
    std::copy(is_dest.blocks_.begin(), is_dest.blocks_.end(), is_dest_pin);
    utl::verify(
        cudaSuccess ==
            cudaMemcpyAsync(thrust::raw_pointer_cast(is_dest_[dir].data()),
                            is_dest_pin,
                            is_dest.blocks_.size() * sizeof(std::uint64_t),
                            cudaMemcpyHostToDevice, stream_),
        "gpu mcraptor: could not copy is_dest");

    dist_to_dest_dev_[dir].resize(dist_to_dest.size());
    auto* const dd_pin = dist_to_dest_pin_[dir].ensure(dist_to_dest.size());
    std::copy(dist_to_dest.begin(), dist_to_dest.end(), dd_pin);
    utl::verify(cudaSuccess ==
                    cudaMemcpyAsync(
                        thrust::raw_pointer_cast(dist_to_dest_dev_[dir].data()),
                        dd_pin, dist_to_dest.size() * sizeof(std::uint16_t),
                        cudaMemcpyHostToDevice, stream_),
                "gpu mcraptor: could not copy dist_to_dest");

    if (lb.empty()) {  // no lower bounds supplied: inert zeros
      auto const n = static_cast<std::size_t>(tt_.n_locations_);
      cudaMemsetAsync(lb_dev_[dir].ensure(n, stream_), 0,
                      n * sizeof(std::uint16_t), stream_);
      return;
    }
    auto* const lb_pin = lb_pin_[dir].ensure(lb.size());
    std::copy(lb.begin(), lb.end(), lb_pin);
    utl::verify(cudaSuccess ==
                    cudaMemcpyAsync(lb_dev_[dir].ensure(lb.size(), stream_),
                                    lb_pin, lb.size() * sizeof(std::uint16_t),
                                    cudaMemcpyHostToDevice, stream_),
                "gpu mcraptor: could not copy lower bounds");
  }

  bool is_intermodal_dest_[2];  // per direction: [0]=fwd, [1]=bwd
  std::uint32_t bag_cap_;
  std::size_t et_pool_cap_;
  std::uint32_t et_tasks_cap_;

  device_timetable tt_;

  thrust::device_vector<mc_label_t> bags_;
  thrust::device_vector<mc_label_t> bag_pool_;
  thrust::device_vector<std::uint32_t> bag_next_;
  thrust::device_vector<std::uint32_t> bag_ovf_;
  thrust::device_vector<std::uint32_t> bag_pool_count_;
  std::uint32_t bag_pool_cap_;
  thrust::device_vector<std::uint64_t> reuse_bags_[2];
  thrust::device_vector<std::uint32_t> bag_locks_;
  thrust::device_vector<std::uint8_t> bag_hwm_;
  thrust::device_vector<mc_label_t> dest_bag_[2];
  thrust::device_vector<std::uint32_t> dest_lock_[2];
  thrust::device_vector<std::uint32_t> dest_best_key_[2];
  thrust::device_vector<std::uint32_t> dest_best_total_[2];
  thrust::device_vector<mc_breadcrumb> bc_;
  thrust::device_vector<std::uint32_t> bc_count_;

  thrust::device_vector<std::uint32_t> touched_;
  thrust::device_vector<std::uint32_t> station_mark_;
  thrust::device_vector<std::uint32_t> prev_station_mark_;
  thrust::device_vector<std::uint32_t> route_mark_;
  thrust::device_vector<std::uint32_t> rt_transport_mark_;

  thrust::device_vector<std::uint32_t> route_list_;
  thrust::device_vector<std::uint32_t> route_list_count_;
  thrust::device_vector<std::uint32_t> et_task_list_;
  thrust::device_vector<std::uint32_t> et_task_count_;
  thrust::device_vector<std::uint32_t> et_ent_key_;
  thrust::device_vector<std::uint16_t> et_ent_ex_;
  thrust::device_vector<std::uint8_t> et_ent_sl_;
  thrust::device_vector<std::uint32_t> task_bits_;
  thrust::device_vector<std::uint32_t> route_task_start_;
  thrust::device_vector<std::uint32_t> et_task_off_;
  thrust::device_vector<std::uint8_t> et_task_cnt_;
  thrust::device_vector<std::uint32_t> et_entry_count_;
  thrust::device_vector<std::uint32_t> route_entry_count_;
  thrust::device_vector<std::uint64_t> route_single_entry_;
  thrust::device_vector<std::uint32_t> route_single_flat_;
  thrust::device_vector<std::uint32_t> any_marked_;
  thrust::device_vector<std::uint32_t> done_;
  thrust::device_vector<std::uint32_t> overflow_;

  thrust::device_vector<std::uint64_t> is_dest_[2];
  pinned_host_buffer<std::uint64_t> is_dest_pin_[2];
  thrust::device_vector<std::uint16_t> dist_to_dest_dev_[2];
  pinned_host_buffer<std::uint16_t> dist_to_dest_pin_[2];
  // td egress offsets (q.td_dest_), sparse groups per direction
  device_buffer<location_idx_t> td_dest_locs_dev_[2];
  device_buffer<std::uint32_t> td_dest_ranges_dev_[2];
  device_buffer<td_offset> td_dest_data_dev_[2];
  pinned_host_buffer<location_idx_t> td_dest_locs_pin_[2];
  pinned_host_buffer<std::uint32_t> td_dest_ranges_pin_[2];
  pinned_host_buffer<td_offset> td_dest_data_pin_[2];
  device_buffer<std::uint16_t> lb_dev_[2];
  pinned_host_buffer<std::uint16_t> lb_pin_[2];

  pinned_host_buffer<std::pair<location_idx_t, delta_t>> starts_pin_;
  device_buffer<std::pair<location_idx_t, delta_t>> starts_dev_;

  pinned_host_buffer<location_idx_t> rec_dest_pin_;
  device_buffer<location_idx_t> rec_dest_;
  device_buffer<gpu_journey> rec_out_;
  pinned_host_buffer<gpu_journey> rec_host_out_;
  pinned_host_buffer<std::uint32_t> overflow_pin_;

  cudaStream_t stream_;
};

gpu_mcraptor_state::gpu_mcraptor_state(gpu_timetable const& gtt)
    : impl_{std::make_unique<impl>(gtt)} {}

gpu_mcraptor_state::~gpu_mcraptor_state() = default;

// ---- kernels ----------------------------------------------------------------

template <direction SearchDir, mc_crit Crit>
__global__ void mc_init_arrivals_kernel(mcraptor_impl<SearchDir, Crit> r,
                                        delta_t const d_start) {
  r.init_arrivals(d_start);
}

template <direction SearchDir, mc_crit Crit>
__global__ void mc_begin_round_kernel(mcraptor_impl<SearchDir, Crit> r) {
  if (*r.done_) {
    return;
  }
  if (get_global_thread_id() == 0U) {
    *r.any_marked_ = 0U;
  }
}

template <direction SearchDir, mc_crit Crit>
__global__ void mc_mark_routes_kernel(mcraptor_impl<SearchDir, Crit> r) {
  if (*r.done_) {
    return;
  }
  r.mark_routes();
}

// only launched when the query carries rt transports (see execute)
template <direction SearchDir, mc_crit Crit>
__global__ void mc_mark_rt_kernel(mcraptor_impl<SearchDir, Crit> r) {
  if (*r.done_) {
    return;
  }
  r.mark_rt_transports();
}

template <direction SearchDir,
          mc_crit Crit,
          bool WithClaszFilter,
          bool IsWheelchair>
__global__ void mc_scan_rt_kernel(mcraptor_impl<SearchDir, Crit> r,
                                  unsigned const k) {
  if (*r.done_) {
    return;
  }
  r.template scan_rt_transports<WithClaszFilter, IsWheelchair>(k);
}

template <direction SearchDir, mc_crit Crit>
__global__ void mc_begin_transit_kernel(mcraptor_impl<SearchDir, Crit> r) {
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

template <direction SearchDir, mc_crit Crit>
__global__ void mc_build_route_list_kernel(mcraptor_impl<SearchDir, Crit> r) {
  if (*r.done_) {
    return;
  }
  r.build_route_list();
}

template <direction SearchDir, mc_crit Crit, bool IsWheelchair>
__global__ void mc_et_collect_kernel(mcraptor_impl<SearchDir, Crit> r,
                                     unsigned const k) {
  if (*r.done_) {
    return;
  }
  r.template et_collect_tasks<IsWheelchair>(k);
}

template <direction SearchDir, mc_crit Crit>
__global__ void mc_et_lookups_kernel(mcraptor_impl<SearchDir, Crit> r,
                                     unsigned const k) {
  if (*r.done_) {
    return;
  }
  r.et_run_lookups(k);
}

template <direction SearchDir,
          mc_crit Crit,
          bool WithClaszFilter,
          bool IsWheelchair>
__global__ void __launch_bounds__(kMcScanThreads)
    mc_scan_routes_kernel(mcraptor_impl<SearchDir, Crit> r, unsigned const k) {
  extern __shared__ mc_seg seg_smem[];  // kMcMaxSegs per warp
  if (*r.done_) {
    return;
  }
  r.template scan_routes<WithClaszFilter, IsWheelchair>(k, seg_smem);
}

template <direction SearchDir, mc_crit Crit>
__global__ void mc_begin_footpath_kernel(mcraptor_impl<SearchDir, Crit> r) {
  if (*r.done_) {
    return;
  }
  if (*r.any_marked_ == 0U) {  // no location improved -> search converged
    if (get_global_thread_id() == 0U) {
      *r.done_ = 1U;
    }
    return;
  }
}

template <direction SearchDir, mc_crit Crit>
__global__ void mc_transfers_footpaths_kernel(mcraptor_impl<SearchDir, Crit> r,
                                              unsigned const k) {
  if (*r.done_) {
    return;
  }
  r.update_transfers_and_footpaths(k);
  r.route_mark_.reset();
}

template <direction SearchDir, mc_crit Crit>
__global__ void mc_clear_bags_kernel(mcraptor_impl<SearchDir, Crit> r) {
  r.clear_bags();
}

template <direction SearchDir, mc_crit Crit>
__global__ void mc_reconstruct_kernel(location_idx_t const* const dest_list,
                                      std::uint32_t const n_dest,
                                      mcraptor_impl<SearchDir, Crit> r,
                                      gpu_journey* const out) {
  auto const tid = blockIdx.x * blockDim.x + threadIdx.x;
  // a dest bag can use every slot the u8 hwm allows
  auto const per_dest = kMcMaxBagSlots;
  if (tid >= n_dest * per_dest) {
    return;
  }
  out[tid].state_ = reconstruction_result::kNotReconstructed;
  auto const dest = dest_list[tid / per_dest];
  auto const lab = r.bag_label(to_idx(dest), tid % per_dest);
  if (lab == kMcEmptySlot) {
    return;
  }
  r.reconstruct_label(dest, lab, &out[tid]);
}

// Keyed by kernel address: kernels sharing a signature need their own block
// size ("too many resources" otherwise).
template <typename Kernel>
std::pair<int, int> mc_launch_dims(Kernel kernel) {
  static thread_local std::unordered_map<void*, std::pair<int, int>> cache;
  auto const key = reinterpret_cast<void*>(kernel);
  if (auto const it = cache.find(key); it != end(cache)) {
    return it->second;
  }
  auto blocks = 0;
  auto threads = 0;
  cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, kernel, 0, 0);
  auto const dims = std::pair{blocks, threads};
  cache.emplace(key, dims);
  return dims;
}

template <typename Kernel, typename... Args>
void mc_launch(Kernel kernel, cudaStream_t stream, Args&&... args) {
  auto const [blocks, threads] = mc_launch_dims(kernel);
  kernel<<<blocks, threads, 0, stream>>>(std::forward<Args>(args)...);
}

// ---- algorithm --------------------------------------------------------------

template <direction SearchDir, mc_crit Crit>
gpu_mcraptor<SearchDir, Crit>::gpu_mcraptor(
    timetable const& tt,
    rt_timetable const* rtt,
    gpu_mcraptor_state& state,
    bitvec& is_dest,
    std::array<bitvec, kMaxVias> const& /* is_via */,
    std::vector<std::uint16_t> const& dist_to_dest,
    hash_map<location_idx_t, std::vector<td_offset>> const& td_dist_to_dest,
    std::vector<std::uint16_t> const& lb,
    std::vector<via_stop> const& via_stops,
    day_idx_t const base,
    clasz_mask_t const allowed_claszes,
    bool const require_bike_transport,
    bool const require_car_transport,
    bool const is_wheelchair,
    bool const no_compulsory_reservation,
    transfer_time_settings const& tts,
    profile_idx_t const prf_idx)
    : tt_{tt},
      rtt_{rtt},
      gpu_rtt_{rtt == nullptr ? nullptr
                              : static_cast<gpu_rt_timetable const*>(
                                    rtt->gpu_rtt_.ptr_.get())},
      state_{state},
      is_dest_{is_dest},
      base_{base},
      allowed_claszes_{allowed_claszes},
      is_wheelchair_{is_wheelchair},
      transfer_time_settings_{tts},
      prf_idx_{prf_idx},
      worst_at_dest_{kInvalidDelta<SearchDir>} {
  utl::verify(via_stops.empty(), "gpu mcraptor: via stops not supported");
  utl::verify(!require_bike_transport && !require_car_transport &&
                  !no_compulsory_reservation,
              "gpu mcraptor: bike/car transport, reservation filter not "
              "supported");
  utl::verify(rtt == nullptr || gpu_rtt_ != nullptr,
              "gpu mcraptor: rt search requires the uploaded device rt "
              "timetable (rt_timetable::gpu_rtt_)");
  utl::verify(lb.empty() || lb.size() == tt.n_locations(),
              "gpu mcraptor: lower bounds required (kUseLowerBounds)");
  state_.impl_->resize_rt(rtt == nullptr ? 0U : rtt->n_rt_transports());
  reset_arrivals();
  state_.impl_->upload_query(kDirIdx, is_dest, dist_to_dest, td_dist_to_dest,
                             lb);
}

template <direction SearchDir, mc_crit Crit>
void gpu_mcraptor<SearchDir, Crit>::reset_arrivals() {
  auto& s = *state_.impl_;
  worst_at_dest_ = kInvalidDelta<SearchDir>;
  cudaMemsetAsync(thrust::raw_pointer_cast(s.dest_bag_[kDirIdx].data()), 0xFF,
                  s.dest_bag_[kDirIdx].size() * sizeof(mc_label_t), s.stream_);
  cudaMemsetAsync(thrust::raw_pointer_cast(s.reuse_bags_[kDirIdx].data()), 0xFF,
                  s.reuse_bags_[kDirIdx].size() * sizeof(std::uint64_t),
                  s.stream_);
  cudaMemsetAsync(
      thrust::raw_pointer_cast(s.dest_best_key_[kDirIdx].data()), 0xFF,
      s.dest_best_key_[kDirIdx].size() * sizeof(std::uint32_t), s.stream_);
  cudaMemsetAsync(
      thrust::raw_pointer_cast(s.dest_best_total_[kDirIdx].data()), 0xFF,
      s.dest_best_total_[kDirIdx].size() * sizeof(std::uint32_t), s.stream_);
  cudaMemsetAsync(thrust::raw_pointer_cast(s.overflow_.data()), 0,
                  sizeof(std::uint32_t), s.stream_);
}

template <direction SearchDir, mc_crit Crit>
mcraptor_impl<SearchDir, Crit> make_impl(
    gpu_mcraptor_state::impl& s,
    unsigned const dir_idx,
    transfer_time_settings const& tts,
    clasz_mask_t const allowed_claszes,
    profile_idx_t const prf_idx,
    day_idx_t const base,
    delta_t const worst_at_dest,
    delta_t const dep,
    cuda::std::span<std::pair<location_idx_t, delta_t> const> starts) {
  return mcraptor_impl<SearchDir, Crit>{
      .any_marked_ = thrust::raw_pointer_cast(s.any_marked_.data()),
      .done_ = thrust::raw_pointer_cast(s.done_.data()),
      .overflow_ = thrust::raw_pointer_cast(s.overflow_.data()),
      .tt_ = s.tt_,
      .rtt_ = device_rt_timetable{},
      .transfer_time_settings_ = tts,
      .allowed_claszes_ = allowed_claszes,
      .prf_idx_ = prf_idx,
      .base_ = base,
      .worst_at_dest_ = worst_at_dest,
      .starts_ = starts,
      .is_dest_ = {to_view(s.is_dest_[dir_idx])},
      .dist_to_end_ = to_view(s.dist_to_dest_dev_[dir_idx]),
      .td_dest_locs_ = {s.td_dest_locs_dev_[dir_idx].data(),
                        s.td_dest_locs_dev_[dir_idx].size()},
      .td_dest_ = {.data_ = {s.td_dest_data_dev_[dir_idx].data(),
                             s.td_dest_data_dev_[dir_idx].size()},
                   .bucket_starts_ = {s.td_dest_ranges_dev_[dir_idx].data(),
                                      s.td_dest_ranges_dev_[dir_idx].size()}},
      .lb_ = {s.lb_dev_[dir_idx].data(), s.lb_dev_[dir_idx].size()},
      .bags_ = thrust::raw_pointer_cast(s.bags_.data()),
      .bag_pool_ = thrust::raw_pointer_cast(s.bag_pool_.data()),
      .bag_ovf_ = thrust::raw_pointer_cast(s.bag_ovf_.data()),
      .bag_next_ = thrust::raw_pointer_cast(s.bag_next_.data()),
      .bag_pool_count_ = thrust::raw_pointer_cast(s.bag_pool_count_.data()),
      .bag_pool_cap_ = s.bag_pool_cap_,
      .reuse_bags_ = thrust::raw_pointer_cast(s.reuse_bags_[dir_idx].data()),
      .bag_cap_ = s.bag_cap_,
      .dep_ = dep,
      .bag_locks_ = thrust::raw_pointer_cast(s.bag_locks_.data()),
      .bag_hwm_ = thrust::raw_pointer_cast(s.bag_hwm_.data()),
      .dest_bag_ = thrust::raw_pointer_cast(s.dest_bag_[dir_idx].data()),
      .dest_lock_ = thrust::raw_pointer_cast(s.dest_lock_[dir_idx].data()),
      .dest_best_key_ =
          thrust::raw_pointer_cast(s.dest_best_key_[dir_idx].data()),
      .dest_best_total_ =
          thrust::raw_pointer_cast(s.dest_best_total_[dir_idx].data()),
      .bc_ = thrust::raw_pointer_cast(s.bc_.data()),
      .bc_count_ = thrust::raw_pointer_cast(s.bc_count_.data()),
      .bc_cap_ = static_cast<std::uint32_t>(s.bc_.size()),
      .touched_ = {to_mutable_view(s.touched_)},
      .station_mark_ = {to_mutable_view(s.station_mark_)},
      .prev_station_mark_ = {to_mutable_view(s.prev_station_mark_)},
      .route_mark_ = {to_mutable_view(s.route_mark_)},
      .rt_transport_mark_ = {to_mutable_view(s.rt_transport_mark_)},
      .route_list_ = to_mutable_view(s.route_list_),
      .route_list_count_ = thrust::raw_pointer_cast(s.route_list_count_.data()),
      .et_task_list_ = to_mutable_view(s.et_task_list_),
      .et_task_count_ = thrust::raw_pointer_cast(s.et_task_count_.data()),
      .et_ent_key_ = thrust::raw_pointer_cast(s.et_ent_key_.data()),
      .et_ent_ex_ = thrust::raw_pointer_cast(s.et_ent_ex_.data()),
      .et_ent_sl_ = thrust::raw_pointer_cast(s.et_ent_sl_.data()),
      .task_bits_ = thrust::raw_pointer_cast(s.task_bits_.data()),
      .route_task_start_ = thrust::raw_pointer_cast(s.route_task_start_.data()),
      .et_task_off_ = thrust::raw_pointer_cast(s.et_task_off_.data()),
      .et_task_cnt_ = thrust::raw_pointer_cast(s.et_task_cnt_.data()),
      .et_entry_count_ = thrust::raw_pointer_cast(s.et_entry_count_.data()),
      .et_pool_cap_ = s.et_pool_cap_,
      .et_tasks_cap_ = s.et_tasks_cap_,
      .route_entry_count_ =
          thrust::raw_pointer_cast(s.route_entry_count_.data()),
      .route_single_entry_ =
          thrust::raw_pointer_cast(s.route_single_entry_.data()),
      .route_single_flat_ =
          thrust::raw_pointer_cast(s.route_single_flat_.data())};
}

template <direction SearchDir, mc_crit Crit>
void gpu_mcraptor<SearchDir, Crit>::next_start_time() {
  starts_.clear();
  auto& s = *state_.impl_;
  auto const r = make_impl<SearchDir, Crit>(s, kDirIdx, transfer_time_settings_,
                                            allowed_claszes_, 0U, base_,
                                            worst_at_dest_, 0, {});
  mc_launch(mc_clear_bags_kernel<SearchDir, Crit>, s.stream_, r);
  cudaMemsetAsync(thrust::raw_pointer_cast(s.bag_pool_count_.data()), 0,
                  sizeof(std::uint32_t), s.stream_);
  thrust::fill(thrust::cuda::par.on(s.stream_), s.station_mark_.begin(),
               s.station_mark_.end(), 0U);
  thrust::fill(thrust::cuda::par.on(s.stream_), s.prev_station_mark_.begin(),
               s.prev_station_mark_.end(), 0U);
  thrust::fill(thrust::cuda::par.on(s.stream_), s.route_mark_.begin(),
               s.route_mark_.end(), 0U);
  thrust::fill(thrust::cuda::par.on(s.stream_), s.rt_transport_mark_.begin(),
               s.rt_transport_mark_.end(), 0U);
}

template <direction SearchDir, mc_crit Crit>
void gpu_mcraptor<SearchDir, Crit>::add_start(location_idx_t const l,
                                              unixtime_t const t,
                                              via_offset_t) {
  starts_.emplace_back(l, t);
}

template <direction SearchDir, mc_crit Crit>
void gpu_mcraptor<SearchDir, Crit>::execute(unixtime_t const start_time,
                                            std::uint8_t const max_transfers,
                                            unixtime_t const worst_time_at_dest,
                                            pareto_set<journey>& results) {
  auto& s = *state_.impl_;
  constexpr auto const kFwd = SearchDir == direction::kForward;

  auto const d_worst = unix_to_delta(base(), worst_time_at_dest);
  worst_at_dest_ = kFwd ? std::min(d_worst, worst_at_dest_)
                        : std::max(d_worst, worst_at_dest_);

  // upload the seeds (stop, seeded stop time in delta units)
  auto* const starts_pin = s.starts_pin_.ensure(starts_.size());
  for (auto i = std::size_t{0U}; i != starts_.size(); ++i) {
    starts_pin[i] = {starts_[i].first,
                     unix_to_delta(base(), starts_[i].second)};
  }
  auto* const starts_dev = s.starts_dev_.ensure(starts_.size(), s.stream_);
  CUDA_CHECK(cudaMemcpyAsync(
      starts_dev, starts_pin,
      starts_.size() * sizeof(std::pair<location_idx_t, delta_t>),
      cudaMemcpyHostToDevice, s.stream_));

  auto const d_start_dep = unix_to_delta(base(), start_time);
  auto r = make_impl<SearchDir, Crit>(
      s, kDirIdx, transfer_time_settings_, allowed_claszes_, prf_idx_, base_,
      worst_at_dest_, d_start_dep,
      cuda::std::span<std::pair<location_idx_t, delta_t> const>{
          starts_dev, starts_.size()});

  // rt: the rt traffic days make rt-updated static transports inactive (the rt
  // scan covers them); tt_.bitfields_ stays static (is_route_active reads it).
  auto const rt_active = gpu_rtt_ != nullptr;
  if (rt_active) {
    r.rtt_ = gpu_rtt_->impl_->to_device_rt_timetable();
    r.tt_.transport_traffic_days_ = r.rtt_.transport_traffic_days_;
  }
  auto const with_rt_scan = rt_active && r.rtt_.n_rt_transports_ != 0U;

  auto const end_k =
      static_cast<std::uint32_t>(std::min(max_transfers, kMaxTransfers) + 2U);

  // === ROUTING KERNELS ===
  mc_launch(mc_init_arrivals_kernel<SearchDir, Crit>, s.stream_, r,
            d_start_dep);
  for (auto k = 1U; k != end_k; ++k) {
    mc_launch(mc_begin_round_kernel<SearchDir, Crit>, s.stream_, r);
    mc_launch(mc_mark_routes_kernel<SearchDir, Crit>, s.stream_, r);
    if (with_rt_scan) {
      mc_launch(mc_mark_rt_kernel<SearchDir, Crit>, s.stream_, r);
    }
    mc_launch(mc_begin_transit_kernel<SearchDir, Crit>, s.stream_, r);
    mc_launch(mc_build_route_list_kernel<SearchDir, Crit>, s.stream_, r);
    if (is_wheelchair_) {
      mc_launch(mc_et_collect_kernel<SearchDir, Crit, true>, s.stream_, r, k);
    } else {
      mc_launch(mc_et_collect_kernel<SearchDir, Crit, false>, s.stream_, r, k);
    }
    mc_launch(mc_et_lookups_kernel<SearchDir, Crit>, s.stream_, r, k);
    // fixed geometry: occupancy launch cannot size the shared segment slab
    auto const scan_blocks = 512U;
    auto const scan_shared =
        (kMcScanThreads / 32U) * kMcMaxSegs * sizeof(mc_seg);
    auto const with_clasz = allowed_claszes_ != all_clasz_allowed();
    if (with_clasz) {
      if (is_wheelchair_) {
        mc_scan_routes_kernel<SearchDir, Crit, true, true>
            <<<scan_blocks, kMcScanThreads, scan_shared, s.stream_>>>(r, k);
      } else {
        mc_scan_routes_kernel<SearchDir, Crit, true, false>
            <<<scan_blocks, kMcScanThreads, scan_shared, s.stream_>>>(r, k);
      }
    } else {
      if (is_wheelchair_) {
        mc_scan_routes_kernel<SearchDir, Crit, false, true>
            <<<scan_blocks, kMcScanThreads, scan_shared, s.stream_>>>(r, k);
      } else {
        mc_scan_routes_kernel<SearchDir, Crit, false, false>
            <<<scan_blocks, kMcScanThreads, scan_shared, s.stream_>>>(r, k);
      }
    }
    // both read round k - 1 and write round k: order does not matter
    if (with_rt_scan) {
      if (with_clasz) {
        if (is_wheelchair_) {
          mc_launch(mc_scan_rt_kernel<SearchDir, Crit, true, true>, s.stream_,
                    r, k);
        } else {
          mc_launch(mc_scan_rt_kernel<SearchDir, Crit, true, false>, s.stream_,
                    r, k);
        }
      } else {
        if (is_wheelchair_) {
          mc_launch(mc_scan_rt_kernel<SearchDir, Crit, false, true>, s.stream_,
                    r, k);
        } else {
          mc_launch(mc_scan_rt_kernel<SearchDir, Crit, false, false>, s.stream_,
                    r, k);
        }
      }
    }
    mc_launch(mc_begin_footpath_kernel<SearchDir, Crit>, s.stream_, r);
    mc_launch(mc_transfers_footpaths_kernel<SearchDir, Crit>, s.stream_, r, k);
  }
  cudaStreamSynchronize(s.stream_);
  CUDA_CHECK(cudaPeekAtLastError());

  // === DEVICE RECONSTRUCT ===
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
  auto const total = n_dest * kMcMaxBagSlots;

  auto* const dest_pin = s.rec_dest_pin_.ensure(dest_list.size());
  std::copy(dest_list.begin(), dest_list.end(), dest_pin);
  auto* const dest_dev = s.rec_dest_.ensure(dest_list.size(), s.stream_);
  CUDA_CHECK(cudaMemcpyAsync(dest_dev, dest_pin,
                             dest_list.size() * sizeof(location_idx_t),
                             cudaMemcpyHostToDevice, s.stream_));
  auto* const rec_out_dev = s.rec_out_.ensure(total, s.stream_);
  auto* const rec_host = s.rec_host_out_.ensure(total);
  auto* const overflow_pin = s.overflow_pin_.ensure(1U);

  {
    auto const threads = 128U;
    auto const blocks = (total + threads - 1U) / threads;
    mc_reconstruct_kernel<SearchDir, Crit>
        <<<blocks, threads, 0, s.stream_>>>(dest_dev, n_dest, r, rec_out_dev);
    CUDA_CHECK(cudaMemcpyAsync(rec_host, rec_out_dev,
                               total * sizeof(gpu_journey),
                               cudaMemcpyDeviceToHost, s.stream_));
    CUDA_CHECK(cudaMemcpyAsync(
        overflow_pin, thrust::raw_pointer_cast(s.overflow_.data()),
        sizeof(std::uint32_t), cudaMemcpyDeviceToHost, s.stream_));
    cudaStreamSynchronize(s.stream_);
  }
  CUDA_CHECK(cudaPeekAtLastError());

  // capacity canaries: a lossy search must never go unnoticed
  utl::verify(*overflow_pin == 0U,
              "gpu mcraptor: capacity overflow (mask={}): bag={} arena={} "
              "route_bag={} rec={} et_block={} et_tasks={} task_cap={}",
              *overflow_pin, (*overflow_pin & kMcOverflowBag) != 0U,
              (*overflow_pin & kMcOverflowArena) != 0U,
              (*overflow_pin & kMcOverflowRouteBag) != 0U,
              (*overflow_pin & kMcOverflowRec) != 0U,
              (*overflow_pin & kMcOverflowEtBlock) != 0U,
              (*overflow_pin & kMcOverflowEtTasks) != 0U,
              (*overflow_pin & kMcOverflowTaskCap) != 0U);

  // === CONVERT DEVICE JOURNEYS TO HOST JOURNEYS ===
  for (auto idx = std::uint32_t{0U}; idx != total; ++idx) {
    auto const& gj = rec_host[idx];

    if (gj.state_ == reconstruction_result::kReconstructionFailed) {
      log(log_lvl::error, "search",
          "gpu mcraptor reconstruct failed: breadcrumb chain "
          "unreconstructable (dest={})",
          to_idx(gj.dest_l_));
      continue;
    }
    if (gj.state_ != reconstruction_result::kOk || gj.n_legs_ == 0U) {
      continue;
    }

    auto j = journey{};
    j.start_time_ = start_time;
    j.dest_time_ = delta_to_unix(base(), gj.dest_time_);
    j.dest_ = gj.dest_l_;
    j.transfers_ = gj.transfers_;
    if constexpr (kHasNonTransit) {
      j.criteria_cost_ = gj.criteria_cost_;
    }
    if constexpr (kHasModeFilter) {
      j.criteria_mode_filter_ = gj.criteria_mode_filter_ != 0U;
    }
    if constexpr (kHasModeSwitches) {
      j.criteria_mode_switches_ = gj.criteria_mode_switches_;
    }

    for (auto li2 = 0U; li2 != gj.n_legs_; ++li2) {
      auto const li = (SearchDir == direction::kForward)
                          ? (static_cast<unsigned>(gj.n_legs_) - 1U - li2)
                          : li2;
      auto const& gl = gj.legs_[li];
      auto const from = gl.from_l_;
      auto const to = gl.to_l_;
      auto const dep = delta_to_unix(base(), gl.dep_);
      auto const arr = delta_to_unix(base(), gl.arr_);

      if (gl.is_footpath_) {
        j.legs_.emplace_back(journey::leg{
            SearchDir, from, to, dep, arr,
            footpath{to, duration_t{
                             static_cast<duration_t::rep>(gl.fp_duration_)}}});
      } else if (gl.rt_transport_ != rt_transport_idx_t::invalid()) {
        auto const rt_t = gl.rt_transport_;
        auto const run =
            rt::run{.t_ = rtt_->resolve_static(rt_t),
                    .stop_range_ =
                        interval<stop_idx_t>{
                            stop_idx_t{0},
                            static_cast<stop_idx_t>(
                                rtt_->rt_transport_location_seq_[rt_t].size())},
                    .rt_ = rt_t};
        j.legs_.emplace_back(journey::leg{
            SearchDir, from, to, dep, arr,
            journey::run_enter_exit{run, gl.enter_stop_, gl.exit_stop_}});
      } else {
        auto const t_idx = gl.transport_;
        auto const route = tt_.transport_route_[t_idx];
        auto const route_len =
            static_cast<stop_idx_t>(tt_.route_location_seq_[route].size());
        auto const run = rt::run{
            .t_ = transport{t_idx, gl.day_},
            .stop_range_ = interval<stop_idx_t>{stop_idx_t{0}, route_len}};
        j.legs_.emplace_back(journey::leg{
            SearchDir, from, to, dep, arr,
            journey::run_enter_exit{run, gl.enter_stop_, gl.exit_stop_}});
      }
    }

    // backward: re-anchor footpaths to the previous trip's arrival
    for (auto i = std::size_t{1U}; i < j.legs_.size(); ++i) {
      if (std::holds_alternative<footpath>(j.legs_[i].uses_)) {
        auto const dur = std::get<footpath>(j.legs_[i].uses_).duration();
        j.legs_[i].dep_time_ = j.legs_[i - 1U].arr_time_;
        j.legs_[i].arr_time_ = j.legs_[i].dep_time_ + dur;
      }
    }

    results.add(std::move(j));
  }
}

// Offset legs and the start footpath, on the host as for the CPU mcraptor.
template <direction SearchDir, mc_crit Crit>
void gpu_mcraptor<SearchDir, Crit>::reconstruct(query const& q, journey& j) {
  mc_reconstruct_offsets<SearchDir>(tt_, rtt_, q, j);
  j.is_reconstructed_ = true;
}

#define NIGIRI_GPU_MCRAPTOR_INSTANTIATE(C)             \
  template class gpu_mcraptor<direction::kForward, C>; \
  template class gpu_mcraptor<direction::kBackward, C>;
NIGIRI_GPU_MCRAPTOR_INSTANTIATE(mc_crit::arr)
NIGIRI_GPU_MCRAPTOR_INSTANTIATE(mc_crit::non_transit)
NIGIRI_GPU_MCRAPTOR_INSTANTIATE(mc_crit::mode_filter)
NIGIRI_GPU_MCRAPTOR_INSTANTIATE(mc_crit::non_transit_mode_filter)
NIGIRI_GPU_MCRAPTOR_INSTANTIATE(mc_crit::mode_switches)
NIGIRI_GPU_MCRAPTOR_INSTANTIATE(mc_crit::non_transit_mode_switches)
NIGIRI_GPU_MCRAPTOR_INSTANTIATE(mc_crit::mode_filter_mode_switches)
NIGIRI_GPU_MCRAPTOR_INSTANTIATE(mc_crit::non_transit_mode_filter_mode_switches)
#undef NIGIRI_GPU_MCRAPTOR_INSTANTIATE

}  // namespace nigiri::routing::gpu
