#include "nigiri/routing/gpu/mcraptor.h"

#include <algorithm>
#include <type_traits>
#include <unordered_map>

// hide date.h's NOEXCEPT from CCCL's token pasting, see device_times.h
#pragma push_macro("NOEXCEPT")
#undef NOEXCEPT
#include "cuda/std/span"
#pragma pop_macro("NOEXCEPT")

#include "thrust/device_vector.h"

#include "utl/helpers/algorithm.h"
#include "utl/verify.h"

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

// async byte fill (thrust::fill would synchronize the stream)
template <typename V>
void fill_bytes(V& v, int const byte, cudaStream_t const stream) {
  if (!v.empty()) {  // no rt transports: the rt marks are empty
    CUDA_CHECK(cudaMemsetAsync(thrust::raw_pointer_cast(v.data()), byte,
                               v.size() * sizeof(typename V::value_type),
                               stream));
  }
}

// host range -> pinned staging -> device, async
template <typename T, typename Range>
void upload(T* const dev,
            pinned_host_buffer<T>& pin,
            Range const& src,
            cudaStream_t const stream,
            char const* what) {
  auto* const p = pin.ensure(src.size());
  std::copy(begin(src), end(src), p);
  utl::verify(cudaMemcpyAsync(dev, p, src.size() * sizeof(T),
                              cudaMemcpyHostToDevice, stream) == cudaSuccess,
              "gpu mcraptor: could not copy {}", what);
}

struct gpu_mcraptor_state::impl {
  explicit impl(gpu_timetable const& gtt)
      : tt_{gtt.impl_->to_device_timetable()} {
    cudaStreamCreate(&stream_);

    auto const n_locations = tt_.n_locations_;
    bags_.resize(static_cast<std::size_t>(n_locations) * kMcBagCap);
    fill_bytes(bags_, 0xFF, stream_);
    // chained blocks: the pool scales with the labels, not a worst-case row
    bag_pool_cap_ = std::clamp(n_locations / 2U, kMcBagPoolMin, kMcBagPoolMax);
    bag_pool_.resize(static_cast<std::size_t>(bag_pool_cap_) * kMcBagBlock);
    fill_bytes(bag_pool_, 0xFF, stream_);
    bag_next_.resize(bag_pool_cap_);  // free blocks link to kMcInvalid
    fill_bytes(bag_next_, 0xFF, stream_);
    bag_ovf_.resize(n_locations);
    fill_bytes(bag_ovf_, 0xFF, stream_);
    bag_pool_count_.resize(1U);
    fill_bytes(bag_pool_count_, 0, stream_);
    bag_locks_.resize(n_locations);
    fill_bytes(bag_locks_, 0, stream_);
    bag_hwm_.resize(n_locations);
    fill_bytes(bag_hwm_, 0, stream_);
    for (auto d = 0U; d != 2U; ++d) {
      dest_bag_[d].resize(kMcDestCap);
      dest_best_key_[d].resize(kMaxTransfers + 2U);
      dest_best_total_[d].resize(kMaxTransfers + 2U);
      dest_lock_[d].resize(1U);
      fill_bytes(dest_lock_[d], 0, stream_);
    }

    auto const arena_cap =
        std::clamp(static_cast<std::size_t>(n_locations) * kMcArenaPerLocation,
                   std::size_t{kMcArenaMin}, std::size_t{kMcArenaMax});
    bc_.resize(arena_cap);
    bc_count_.resize(1U);

    touched_.resize(n_bitvec_words(n_locations));
    station_mark_.resize(n_bitvec_words(n_locations));
    prev_station_mark_.resize(n_bitvec_words(n_locations));
    route_mark_.resize(n_bitvec_words(tt_.n_routes_));
    fill_bytes(touched_, 0, stream_);
    fill_bytes(station_mark_, 0, stream_);
    fill_bytes(prev_station_mark_, 0, stream_);
    fill_bytes(route_mark_, 0, stream_);

    route_list_.resize(tt_.n_routes_);
    route_list_count_.resize(1U);
    auto const n_route_stops = tt_.route_of_stop_.size();
    et_task_list_.resize(n_route_stops);
    et_task_count_.resize(1U);
    sb_n_.resize(n_route_stops);
    sb_et_.resize(n_route_stops * kMcPrefixK);
    sb_par_.resize(n_route_stops * kMcPrefixK);
    sb_bx_.resize(n_route_stops * kMcPrefixK);
    any_marked_.resize(1U);
    done_.resize(1U);
    overflow_.resize(1U);
    fill_bytes(overflow_, 0, stream_);
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
      upload(td_dest_locs_dev_[dir].ensure(locs.size(), stream_),
             td_dest_locs_pin_[dir], locs, stream_, "td dest locations");
      upload(td_dest_ranges_dev_[dir].ensure(ranges.size(), stream_),
             td_dest_ranges_pin_[dir], ranges, stream_, "td dest ranges");
      upload(td_dest_data_dev_[dir].ensure(td_data.size(), stream_),
             td_dest_data_pin_[dir], td_data, stream_, "td dest offsets");
    }

    is_dest_[dir].resize(is_dest.blocks_.size());
    upload(thrust::raw_pointer_cast(is_dest_[dir].data()), is_dest_pin_[dir],
           is_dest.blocks_, stream_, "is_dest");
    dist_to_dest_dev_[dir].resize(dist_to_dest.size());
    upload(thrust::raw_pointer_cast(dist_to_dest_dev_[dir].data()),
           dist_to_dest_pin_[dir], dist_to_dest, stream_, "dist_to_dest");
    if (lb.empty()) {  // no lower bounds supplied: inert zeros
      auto const n = static_cast<std::size_t>(tt_.n_locations_);
      cudaMemsetAsync(lb_dev_[dir].ensure(n, stream_), 0,
                      n * sizeof(std::uint16_t), stream_);
    } else {
      upload(lb_dev_[dir].ensure(lb.size(), stream_), lb_pin_[dir], lb, stream_,
             "lower bounds");
    }
  }

  bool is_intermodal_dest_[2];  // per direction: [0]=fwd, [1]=bwd

  device_timetable tt_;

  thrust::device_vector<mc_label_t> bags_;
  thrust::device_vector<mc_label_t> bag_pool_;
  thrust::device_vector<std::uint32_t> bag_next_;
  thrust::device_vector<std::uint32_t> bag_ovf_;
  thrust::device_vector<std::uint32_t> bag_pool_count_;
  std::uint32_t bag_pool_cap_;
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
  thrust::device_vector<std::uint32_t> sb_n_;
  thrust::device_vector<std::uint32_t> sb_et_;
  thrust::device_vector<std::uint32_t> sb_par_;
  thrust::device_vector<std::uint32_t> sb_bx_;
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

  pinned_host_buffer<mc_seed> starts_pin_;
  device_buffer<mc_seed> starts_dev_;

  pinned_host_buffer<location_idx_t> rec_dest_pin_;
  device_buffer<location_idx_t> rec_dest_;
  device_buffer<gpu_journey> rec_out_;
  pinned_host_buffer<gpu_journey> rec_host_out_;
  pinned_host_buffer<std::uint32_t> overflow_pin_;
  device_buffer<std::uint32_t> rec_count_;
  pinned_host_buffer<std::uint32_t> rec_count_pin_;

  cudaStream_t stream_;
};

gpu_mcraptor_state::gpu_mcraptor_state(gpu_timetable const& gtt)
    : impl_{std::make_unique<impl>(gtt)} {}

gpu_mcraptor_state::~gpu_mcraptor_state() = default;

// ---- kernels ----------------------------------------------------------------
// The rounds run without host syncs: every kernel after the init returns once
// done_ is set.

template <direction SearchDir, mc_crit Crit>
__global__ void mc_init_arrivals_kernel(mcraptor_impl<SearchDir, Crit> r) {
  r.init_arrivals();
}

template <direction SearchDir, mc_crit Crit>
__global__ void mc_mark_routes_kernel(mcraptor_impl<SearchDir, Crit> r) {
  if (!*r.done_) {
    r.mark_routes();
  }
}

// only launched when the query carries rt transports (see execute)
template <direction SearchDir, mc_crit Crit>
__global__ void mc_mark_rt_transports_kernel(mcraptor_impl<SearchDir, Crit> r) {
  if (!*r.done_) {
    r.mark_rt_transports();
  }
}

template <direction SearchDir, mc_crit Crit>
__global__ void mc_begin_transit_phase_kernel(
    mcraptor_impl<SearchDir, Crit> r) {
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
__global__ void mc_et_build_route_list_kernel(
    mcraptor_impl<SearchDir, Crit> r) {
  if (!*r.done_) {
    r.et_build_route_list();
  }
}

template <direction SearchDir, mc_crit Crit, bool IsWheelchair>
__global__ void mc_et_collect_tasks_kernel(mcraptor_impl<SearchDir, Crit> r) {
  if (!*r.done_) {
    r.template et_collect_tasks<IsWheelchair>();
  }
}

template <direction SearchDir, mc_crit Crit>
__global__ void mc_et_run_lookups_kernel(mcraptor_impl<SearchDir, Crit> r,
                                         unsigned const k) {
  if (!*r.done_) {
    r.et_run_lookups(k);
  }
}

template <direction SearchDir,
          mc_crit Crit,
          bool WithClaszFilter,
          bool IsWheelchair>
__global__ void mc_loop_routes_kernel(mcraptor_impl<SearchDir, Crit> r,
                                      unsigned const k) {
  if (!*r.done_) {
    r.template loop_routes<WithClaszFilter, IsWheelchair>(k);
  }
}

// only launched when the query carries rt transports (see execute)
template <direction SearchDir,
          mc_crit Crit,
          bool WithClaszFilter,
          bool IsWheelchair>
__global__ void mc_update_rt_transports_kernel(mcraptor_impl<SearchDir, Crit> r,
                                               unsigned const k) {
  if (!*r.done_) {
    r.template update_rt_transports<WithClaszFilter, IsWheelchair>(k);
  }
}

template <direction SearchDir, mc_crit Crit>
__global__ void mc_begin_footpath_phase_kernel(
    mcraptor_impl<SearchDir, Crit> r) {
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

template <direction SearchDir, mc_crit Crit>
__global__ void mc_footpaths_kernel(mcraptor_impl<SearchDir, Crit> r,
                                    unsigned const k) {
  if (!*r.done_) {
    r.update_footpaths(k);
    r.route_mark_.reset();
    r.rt_transport_mark_.reset();
  }
}

template <direction SearchDir, mc_crit Crit>
__global__ void mc_clear_bags_kernel(mcraptor_impl<SearchDir, Crit> r) {
  r.clear_bags();
}

template <direction SearchDir, mc_crit Crit>
__global__ void mc_demote_bags_kernel(mcraptor_impl<SearchDir, Crit> r) {
  r.demote_bags();
}

// one thread per destination bag slot; the journeys are packed at the front
template <direction SearchDir, mc_crit Crit>
__global__ void mc_reconstruct_kernel(location_idx_t const* const dest_list,
                                      std::uint32_t const n_dest,
                                      mcraptor_impl<SearchDir, Crit> r,
                                      gpu_journey* const out,
                                      std::uint32_t* const out_count) {
  auto const tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_dest * kMcMaxBagSlots) {
    return;
  }
  auto const dest = dest_list[tid / kMcMaxBagSlots];
  auto const lab = r.bag_slots(to_idx(dest))[tid % kMcMaxBagSlots];
  if (lab == kMcEmptySlot || mc_round(lab) == 0U || mc_bc(lab) == kMcDemoted) {
    return;  // empty, a round-0 seed (no legs) or an earlier start's label
  }
  r.reconstruct_label(dest, lab, &out[atomicAdd(out_count, 1U)]);
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
      avoided_{mode_filter_dim::avoided_mask()},
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
mcraptor_impl<SearchDir, Crit> make_impl(
    gpu_mcraptor_state::impl& s,
    unsigned const dir_idx,
    transfer_time_settings const& tts,
    clasz_mask_t const allowed_claszes,
    clasz_mask_t const avoided,
    profile_idx_t const prf_idx,
    day_idx_t const base,
    delta_t const worst_at_dest,
    cuda::std::span<mc_seed const> starts) {
  auto const ptr = [](auto& v) { return thrust::raw_pointer_cast(v.data()); };
  return mcraptor_impl<SearchDir, Crit>{
      .any_marked_ = ptr(s.any_marked_),
      .done_ = ptr(s.done_),
      .overflow_ = ptr(s.overflow_),
      .tt_ = s.tt_,
      .rtt_ = device_rt_timetable{},
      .transfer_time_settings_ = tts,
      .allowed_claszes_ = allowed_claszes,
      .avoided_ = avoided,
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
      .bags_ = ptr(s.bags_),
      .bag_pool_ = ptr(s.bag_pool_),
      .bag_ovf_ = ptr(s.bag_ovf_),
      .bag_next_ = ptr(s.bag_next_),
      .bag_pool_count_ = ptr(s.bag_pool_count_),
      .bag_pool_cap_ = s.bag_pool_cap_,
      .bag_hwm_ = ptr(s.bag_hwm_),
      .bag_locks_ = ptr(s.bag_locks_),
      .dest_bag_ = ptr(s.dest_bag_[dir_idx]),
      .dest_lock_ = ptr(s.dest_lock_[dir_idx]),
      .dest_best_key_ = ptr(s.dest_best_key_[dir_idx]),
      .dest_best_total_ = ptr(s.dest_best_total_[dir_idx]),
      .bc_ = ptr(s.bc_),
      .bc_count_ = ptr(s.bc_count_),
      .bc_cap_ = static_cast<std::uint32_t>(s.bc_.size()),
      .touched_ = {to_mutable_view(s.touched_)},
      .station_mark_ = {to_mutable_view(s.station_mark_)},
      .prev_station_mark_ = {to_mutable_view(s.prev_station_mark_)},
      .route_mark_ = {to_mutable_view(s.route_mark_)},
      .rt_transport_mark_ = {to_mutable_view(s.rt_transport_mark_)},
      .route_list_ = to_mutable_view(s.route_list_),
      .route_list_count_ = ptr(s.route_list_count_),
      .et_task_list_ = to_mutable_view(s.et_task_list_),
      .et_task_count_ = ptr(s.et_task_count_),
      .sb_n_ = ptr(s.sb_n_),
      .sb_et_ = ptr(s.sb_et_),
      .sb_par_ = ptr(s.sb_par_),
      .sb_bx_ = ptr(s.sb_bx_)};
}

template <direction SearchDir, mc_crit Crit>
void gpu_mcraptor<SearchDir, Crit>::reset_arrivals() {
  auto& s = *state_.impl_;
  worst_at_dest_ = kInvalidDelta<SearchDir>;
  fill_bytes(s.dest_bag_[kDirIdx], 0xFF, s.stream_);
  fill_bytes(s.dest_best_key_[kDirIdx], 0xFF, s.stream_);
  fill_bytes(s.dest_best_total_[kDirIdx], 0xFF, s.stream_);
  fill_bytes(s.overflow_, 0, s.stream_);
  mc_launch(mc_clear_bags_kernel<SearchDir, Crit>, s.stream_,
            make_impl<SearchDir, Crit>(s, kDirIdx, transfer_time_settings_,
                                       allowed_claszes_, avoided_, 0U, base_,
                                       worst_at_dest_, {}));
  fill_bytes(s.bag_pool_count_, 0, s.stream_);
}

template <direction SearchDir, mc_crit Crit>
void gpu_mcraptor<SearchDir, Crit>::next_start_time() {
  starts_.clear();
  auto& s = *state_.impl_;
  mc_launch(mc_demote_bags_kernel<SearchDir, Crit>, s.stream_,
            make_impl<SearchDir, Crit>(s, kDirIdx, transfer_time_settings_,
                                       allowed_claszes_, avoided_, 0U, base_,
                                       worst_at_dest_, {}));
  fill_bytes(s.station_mark_, 0, s.stream_);
  fill_bytes(s.prev_station_mark_, 0, s.stream_);
  fill_bytes(s.route_mark_, 0, s.stream_);
  fill_bytes(s.rt_transport_mark_, 0, s.stream_);
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

  auto const d_start_dep = unix_to_delta(base(), start_time);

  // upload the seeds
  auto* const starts_pin = s.starts_pin_.ensure(starts_.size());
  for (auto i = std::size_t{0U}; i != starts_.size(); ++i) {
    auto const arr = unix_to_delta(base(), starts_[i].second);
    auto const elapsed = kFwd ? arr - d_start_dep : d_start_dep - arr;
    starts_pin[i] = {
        starts_[i].first, arr,
        seed_walk<SearchDir>(td_start_, starts_[i].first, start_time,
                             static_cast<std::uint16_t>(elapsed))};
  }
  auto* const starts_dev = s.starts_dev_.ensure(starts_.size(), s.stream_);
  CUDA_CHECK(cudaMemcpyAsync(starts_dev, starts_pin,
                             starts_.size() * sizeof(mc_seed),
                             cudaMemcpyHostToDevice, s.stream_));

  auto r = make_impl<SearchDir, Crit>(
      s, kDirIdx, transfer_time_settings_, allowed_claszes_, avoided_, prf_idx_,
      base_, worst_at_dest_,
      cuda::std::span<mc_seed const>{starts_dev, starts_.size()});

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

  // calls f(clasz filter?, wheelchair?) with both as std::bool_constant
  auto const with_clasz = allowed_claszes_ != all_clasz_allowed();
  auto const with_flags = [&](auto&& f) {
    using t = std::true_type;
    using n = std::false_type;
    with_clasz ? (is_wheelchair_ ? f(t{}, t{}) : f(t{}, n{}))
               : (is_wheelchair_ ? f(n{}, t{}) : f(n{}, n{}));
  };

  // === ROUTING KERNELS (as gpu raptor) ===
  mc_launch(mc_init_arrivals_kernel<SearchDir, Crit>, s.stream_, r);
  for (auto k = 1U; k != end_k; ++k) {
    mc_launch(mc_mark_routes_kernel<SearchDir, Crit>, s.stream_, r);
    if (with_rt_scan) {
      mc_launch(mc_mark_rt_transports_kernel<SearchDir, Crit>, s.stream_, r);
    }
    mc_launch(mc_begin_transit_phase_kernel<SearchDir, Crit>, s.stream_, r);
    mc_launch(mc_et_build_route_list_kernel<SearchDir, Crit>, s.stream_, r);
    if (is_wheelchair_) {
      mc_launch(mc_et_collect_tasks_kernel<SearchDir, Crit, true>, s.stream_,
                r);
    } else {
      mc_launch(mc_et_collect_tasks_kernel<SearchDir, Crit, false>, s.stream_,
                r);
    }
    mc_launch(mc_et_run_lookups_kernel<SearchDir, Crit>, s.stream_, r, k);
    // route and rt scans read round k - 1 and write round k (any order)
    with_flags([&](auto const clasz_filter, auto const wheelchair) {
      constexpr auto kClasz = decltype(clasz_filter)::value;
      constexpr auto kWheelchair = decltype(wheelchair)::value;
      mc_launch(mc_loop_routes_kernel<SearchDir, Crit, kClasz, kWheelchair>,
                s.stream_, r, k);
      if (with_rt_scan) {
        mc_launch(mc_update_rt_transports_kernel<SearchDir, Crit, kClasz,
                                                 kWheelchair>,
                  s.stream_, r, k);
      }
    });
    mc_launch(mc_begin_footpath_phase_kernel<SearchDir, Crit>, s.stream_, r);
    mc_launch(mc_footpaths_kernel<SearchDir, Crit>, s.stream_, r, k);
  }

  // === DEVICE RECONSTRUCT ===
  auto dest_list = std::vector<location_idx_t>{};
  if (s.is_intermodal_dest_[kDirIdx]) {
    dest_list.push_back(get_special_station(special_station::kEnd));
  } else {
    is_dest_.for_each_set_bit(
        [&](auto const i) { dest_list.push_back(location_idx_t{i}); });
  }
  if (dest_list.empty()) {
    cudaStreamSynchronize(s.stream_);
    CUDA_CHECK(cudaPeekAtLastError());
    return;
  }

  auto const n_dest = static_cast<std::uint32_t>(dest_list.size());
  auto const total = n_dest * kMcMaxBagSlots;

  auto* const dest_dev = s.rec_dest_.ensure(dest_list.size(), s.stream_);
  upload(dest_dev, s.rec_dest_pin_, dest_list, s.stream_, "destinations");
  auto* const rec_out_dev = s.rec_out_.ensure(total, s.stream_);
  auto* const rec_host = s.rec_host_out_.ensure(total);
  auto* const overflow_pin =
      s.overflow_pin_.ensure(3U);  // flags, bc + pool use
  auto* const rec_count_dev = s.rec_count_.ensure(1U, s.stream_);
  auto* const rec_count_pin = s.rec_count_pin_.ensure(1U);

  // The kernel packs the journeys at the front: copy the count and a prefix
  // that usually holds all of them, so one sync suffices.
  auto const prefix = std::min(total, kMcRecPrefix);
  CUDA_CHECK(
      cudaMemsetAsync(rec_count_dev, 0, sizeof(std::uint32_t), s.stream_));
  {
    auto const threads = 128U;
    auto const blocks = (total + threads - 1U) / threads;
    mc_reconstruct_kernel<SearchDir, Crit><<<blocks, threads, 0, s.stream_>>>(
        dest_dev, n_dest, r, rec_out_dev, rec_count_dev);
  }
  CUDA_CHECK(cudaMemcpyAsync(rec_count_pin, rec_count_dev,
                             sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                             s.stream_));
  CUDA_CHECK(cudaMemcpyAsync(rec_host, rec_out_dev,
                             prefix * sizeof(gpu_journey),
                             cudaMemcpyDeviceToHost, s.stream_));
  CUDA_CHECK(cudaMemcpyAsync(
      overflow_pin, thrust::raw_pointer_cast(s.overflow_.data()),
      sizeof(std::uint32_t), cudaMemcpyDeviceToHost, s.stream_));
  CUDA_CHECK(cudaMemcpyAsync(
      overflow_pin + 1U, thrust::raw_pointer_cast(s.bc_count_.data()),
      sizeof(std::uint32_t), cudaMemcpyDeviceToHost, s.stream_));
  CUDA_CHECK(cudaMemcpyAsync(
      overflow_pin + 2U, thrust::raw_pointer_cast(s.bag_pool_count_.data()),
      sizeof(std::uint32_t), cudaMemcpyDeviceToHost, s.stream_));
  cudaStreamSynchronize(s.stream_);
  CUDA_CHECK(cudaPeekAtLastError());
  auto const n_rec = *rec_count_pin;
  if (n_rec > prefix) {
    CUDA_CHECK(cudaMemcpyAsync(rec_host + prefix, rec_out_dev + prefix,
                               (n_rec - prefix) * sizeof(gpu_journey),
                               cudaMemcpyDeviceToHost, s.stream_));
    cudaStreamSynchronize(s.stream_);
  }

  // capacity canaries: a lossy search must never go unnoticed
  utl::verify(*overflow_pin == 0U,
              "gpu mcraptor: capacity overflow: bag={} (pool {} of {} blocks) "
              "arena={} ({} of {} breadcrumbs) route_bag={}",
              (*overflow_pin & kMcOverflowBag) != 0U, overflow_pin[2],
              s.bag_pool_cap_, (*overflow_pin & kMcOverflowArena) != 0U,
              overflow_pin[1], s.bc_.size(),
              (*overflow_pin & kMcOverflowRouteBag) != 0U);

  // === CONVERT DEVICE JOURNEYS TO HOST JOURNEYS ===
  for (auto idx = std::uint32_t{0U}; idx != n_rec; ++idx) {
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
