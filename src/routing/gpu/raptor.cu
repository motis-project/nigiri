#include <unordered_map>

#include "nigiri/routing/gpu/mcraptor.h"
#include "nigiri/routing/gpu/raptor.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <array>
#include <iostream>
#include <optional>

// hide date.h's NOEXCEPT from CCCL's token pasting, see device_times.h
#pragma push_macro("NOEXCEPT")
#undef NOEXCEPT
#include "cuda/std/array"
#include "cuda/std/span"
#pragma pop_macro("NOEXCEPT")

#include "thrust/copy.h"
#include "thrust/device_vector.h"
#include "thrust/fill.h"
#include "thrust/host_vector.h"

#include "utl/helpers/algorithm.h"
#include "utl/timer.h"

#include "nigiri/for_each_meta.h"
#include "utl/verify.h"

#include "nigiri/logging.h"
#include "nigiri/routing/gpu/cuda_check.cuh"
#include "nigiri/routing/gpu/device_buffer.cuh"
#include "nigiri/routing/gpu/device_timetable.cuh"
#include "nigiri/routing/gpu/pinned_host_buffer.cuh"
#include "nigiri/routing/gpu/raptor_impl.cuh"
#include "nigiri/routing/gpu/timetable_impl.cuh"
#include "nigiri/routing/gpu/types.cuh"
#include "nigiri/td_footpath.h"

namespace nigiri::routing::gpu {



gpu_timetable::gpu_timetable(timetable const& tt)
    : impl_{std::make_unique<impl>(tt)} {}

gpu_timetable::~gpu_timetable() = default;

gpu_rt_timetable::gpu_rt_timetable(timetable const& tt, rt_timetable const& rtt)
    : impl_{std::make_unique<impl>(tt, rtt)} {}

gpu_rt_timetable::~gpu_rt_timetable() = default;

std::unique_ptr<void, void (*)(void*)> make_gpu_rtt(timetable const& tt,
                                                    rt_timetable const& rtt) {
  return {new gpu_rt_timetable{tt, rtt},
          [](void* p) { delete static_cast<gpu_rt_timetable*>(p); }};
}

struct gpu_raptor_state::impl {
  explicit impl(gpu_timetable const& gtt)
      : gtt_{&gtt}, tt_{gtt.impl_->to_device_timetable()} {
    cudaStreamCreate(&stream_);

    auto const n_route_stops = tt_.route_of_stop_.size();
    et_result_.resize(n_route_stops);
    et_task_list_.resize(n_route_stops);
    et_task_count_.resize(1U);
    route_list_.resize(tt_.n_routes_);
    route_list_count_.resize(1U);

    // the state is bound to this timetable for its whole lifetime -> all
    // n_locations/n_routes-sized buffers are allocated once here; the
    // selective-clear scheme requires an all-invalid (0xFF) start
    time_at_dest_.resize(kMaxTransfers + 2);
    tmp_.resize(tt_.n_locations_);
    best_.resize(tt_.n_locations_);
    round_times_.resize(tt_.n_locations_ * (kMaxTransfers + 2));
    cudaMemsetAsync(thrust::raw_pointer_cast(round_times_.data()), 0xFF,
                    round_times_.size() * sizeof(std::uint64_t), stream_);
    cudaMemsetAsync(thrust::raw_pointer_cast(best_.data()), 0xFF,
                    best_.size() * sizeof(std::uint64_t), stream_);
    cudaMemsetAsync(thrust::raw_pointer_cast(tmp_.data()), 0xFF,
                    tmp_.size() * sizeof(std::uint64_t), stream_);
    station_mark_.resize(tt_.n_locations_ / 32U + 1U);
    prev_station_mark_.resize(tt_.n_locations_ / 32U + 1U);
    route_mark_.resize(tt_.n_routes_ / 32U + 1U);
    any_marked_.resize(1U);
    done_.resize(1U);

    // is_dest + dist_to_dest differ between ping vs pong -> handled in
    // upload_query (called by the gpu_raptor ctor -> one per dir)
  }

  ~impl() { cudaStreamDestroy(stream_); }

  // the only per-query sizing: the rt timetable may be absent or may have
  // grown (rt updates) since the last query on this state
  void resize_rt(unsigned const n_rt_transports) {
    rt_transport_mark_.resize(n_rt_transports / 32U + 1U);
  }

  void upload_query(
      unsigned const dir /* fwd=0 bwd=1 -> ping/pong can coexist */,
      void const* owner,
      nigiri::bitvec const& is_dest,
      std::vector<std::uint16_t> const& dist_to_dest,
      hash_map<location_idx_t, std::vector<td_offset>> const& td_dist_to_dest) {
    q_owner_[dir] = owner;
    is_intermodal_dest_[dir] = !dist_to_dest.empty();

    // td dest offsets: flatten into sorted (loc, range, data) groups.
    if (td_dist_to_dest.empty()) {
      // reset used sizes: the state may have served a flex query before
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
          "could not copy td dest offsets");
    }

    // Copy is_dest.
    is_dest_[dir].resize(is_dest.blocks_.size());
    auto* const is_dest_pin = is_dest_pin_[dir].ensure(is_dest.blocks_.size());
    std::copy(is_dest.blocks_.begin(), is_dest.blocks_.end(), is_dest_pin);
    utl::verify(
        cudaSuccess == cudaMemcpyAsync(
                           thrust::raw_pointer_cast(is_dest_[dir].data()),
                           is_dest_pin,
                           is_dest.blocks_.size() *
                               sizeof(std::decay_t<decltype(is_dest)>::block_t),
                           cudaMemcpyHostToDevice, stream_),
        "could not copy is_dest");

    // Copy dist_to_dest.
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

  gpu_timetable const* gtt_;
  std::unique_ptr<gpu_mcraptor_state> mc_state_;  // see try_mc_state()
  bool mc_state_failed_{false};  // do not retry a failed allocation

  bool is_intermodal_dest_[2];  // per direction: [0]=fwd, [1]=bwd
  // which gpu_raptor last uploaded into each direction slot
  void const* q_owner_[2]{nullptr, nullptr};
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
  thrust::device_vector<std::uint32_t> rt_transport_mark_;

  // per-direction query data: [0]=fwd, [1]=bwd (ping/pong interleave on the
  // shared state, each direction uploads its slot once in the ctor)
  thrust::device_vector<std::uint64_t> is_dest_[2];
  pinned_host_buffer<std::uint64_t> is_dest_pin_[2];

  thrust::device_vector<std::uint16_t> dist_to_dest_dev_[2];

  // td egress offsets (q.td_dest_), sparse groups per direction;
  // used size 0 = no td offsets (device never touched)
  device_buffer<location_idx_t> td_dest_locs_dev_[2];
  device_buffer<std::uint32_t> td_dest_ranges_dev_[2];
  device_buffer<td_offset> td_dest_data_dev_[2];
  pinned_host_buffer<location_idx_t> td_dest_locs_pin_[2];
  pinned_host_buffer<std::uint32_t> td_dest_ranges_pin_[2];
  pinned_host_buffer<td_offset> td_dest_data_pin_[2];

  pinned_host_buffer<std::uint16_t> dist_to_dest_[2];
  pinned_host_buffer<std::pair<location_idx_t, unixtime_t>> starts_;
  thrust::device_vector<std::pair<location_idx_t, unixtime_t>> starts_dev_;

  // reusable reconstruction buffers
  pinned_host_buffer<location_idx_t> rec_dest_pin_;
  device_buffer<location_idx_t> rec_dest_;
  device_buffer<gpu_journey> rec_out_;
  pinned_host_buffer<gpu_journey> rec_host_out_;

  device_timetable tt_;

  thrust::device_vector<std::uint32_t> et_result_;
  thrust::device_vector<std::uint32_t> et_task_list_;
  thrust::device_vector<std::uint32_t> et_task_count_;
  thrust::device_vector<std::uint32_t> route_list_;
  thrust::device_vector<std::uint32_t> route_list_count_;

  // BM-RAPTOR bound matrix, (budget+1) x n_locations delta_t, uploaded by
  // gpu_raptor::set_bounds()
  thrust::device_vector<delta_t> bmrap_bounds_;
  // Scratch for build_reach_bounds(), separate from bmrap_bounds_ so building
  // a matrix cannot clobber the one this search is currently pruning with.
  // One slot per matrix kind (0 = tau_dep^<-, 1 = tau_arr^->) because both are
  // live at once: PHASE 2a's matrix is still feeding the pruning search when
  // PHASE 2b builds its own. reach_tag_ is what set_bounds() matches against.
  thrust::device_vector<delta_t> reach_out_[2];
  std::uint64_t reach_tag_[2]{0U, 0U};
  std::size_t reach_n_[2]{0U, 0U};
  // staging buffer for copy_round_times()
  std::vector<std::uint64_t> round_times_host_;

  cudaStream_t stream_;
};

gpu_raptor_state::gpu_raptor_state(gpu_timetable const& gtt)
    : impl_{std::make_unique<impl>(gtt)} {}

gpu_raptor_state::~gpu_raptor_state() = default;

gpu_mcraptor_state* gpu_raptor_state::try_mc_state() {
  if (impl_->mc_state_ == nullptr && !impl_->mc_state_failed_) {
    try {
      impl_->mc_state_ = std::make_unique<gpu_mcraptor_state>(*impl_->gtt_);
    } catch (std::exception const& e) {
      impl_->mc_state_failed_ = true;
      log(log_lvl::info, "gpu",
          "no device mcraptor state ({}) - multicriteria phases stay on the "
          "CPU",
          e.what());
    }
  }
  return impl_->mc_state_.get();
}

gpu_mcraptor_state& gpu_raptor_state::mc_state() {
  auto* const s = try_mc_state();
  utl::verify(s != nullptr, "gpu_raptor_state: no device mcraptor state");
  return *s;
}

template <direction SearchDir>
gpu_raptor<SearchDir>::gpu_raptor(
    timetable const& tt,
    rt_timetable const* rtt,
    gpu_raptor_state& state,
    nigiri::bitvec& is_dest,
    std::array<nigiri::bitvec, kMaxVias> const& /* is_via (GPU: no vias) */,
    std::vector<std::uint16_t> const& dist_to_dest,
    hash_map<location_idx_t, std::vector<td_offset>> const& td_dist_to_dest,
    std::vector<std::uint16_t> const& /* lb (GPU: no lower bounds) */,
    std::vector<via_stop> const& via_stops,
    day_idx_t const base,
    clasz_mask_t const allowed_claszes,
    bool const require_bike_transport,
    bool const require_car_transport,
    bool const is_wheelchair,
    transfer_time_settings const& tts)
    : tt_{tt},
      rtt_{rtt},
      gpu_rtt_{rtt == nullptr ? nullptr
                              : static_cast<gpu_rt_timetable const*>(
                                    rtt->gpu_rtt_.ptr_.get())},
      n_locations_{tt_.n_locations()},
      state_{state},
      is_dest_{is_dest},
      dist_to_dest_{&dist_to_dest},
      td_dist_to_dest_{&td_dist_to_dest},
      base_{base},
      allowed_claszes_{allowed_claszes},
      require_bike_transport_{require_bike_transport},
      require_car_transport_{require_car_transport},
      is_wheelchair_{is_wheelchair},
      transfer_time_settings_{tts} {
  utl::verify(rtt == nullptr || gpu_rtt_ != nullptr,
              "GPU raptor: rt search requires the uploaded device rt "
              "timetable (rt_timetable::gpu_rtt_)");
  state_.impl_->resize_rt(rtt == nullptr ? 0U : rtt->n_rt_transports());
  reset_arrivals();
  state_.impl_->upload_query(kDirIdx, this, is_dest, dist_to_dest,
                             td_dist_to_dest);
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
__global__ void mark_rt_transports_kernel(raptor_impl<SearchDir> r,
                                          unsigned const k) {
  if (*r.done_) {
    return;
  }
  r.mark_rt_transports(k);
}

template <direction SearchDir,
          bool WithClaszFilter,
          bool IsWheelchair,
          bool WithFilters>
__global__ void update_rt_transports_kernel(raptor_impl<SearchDir> r,
                                            unsigned const k) {
  if (*r.done_) {
    return;
  }
  r.template update_rt_transports<WithClaszFilter, IsWheelchair, WithFilters>(
      k);
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

template <direction SearchDir, bool IsWheelchair>
__global__ void et_collect_tasks_kernel(raptor_impl<SearchDir> r,
                                        unsigned const k) {
  if (*r.done_) {
    return;
  }
  r.template et_collect_tasks<IsWheelchair>(k);
}

template <direction SearchDir>
__global__ void et_run_lookups_kernel(raptor_impl<SearchDir> r,
                                      unsigned const k) {
  if (*r.done_) {
    return;
  }
  r.et_run_lookups(k);
}

template <direction SearchDir,
          bool WithClaszFilter,
          bool IsWheelchair,
          bool WithFilters>
__global__ void loop_routes_kernel(raptor_impl<SearchDir> r, unsigned const k) {
  if (*r.done_) {
    return;
  }
  r.template loop_routes<WithClaszFilter, IsWheelchair, WithFilters>(k);
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

template <direction SearchDir, bool WithTdDest, bool WithTdFootpaths>
__global__ void transfers_footpaths_kernel(raptor_impl<SearchDir> r,
                                           unsigned const k) {
  if (*r.done_) {
    return;
  }
  r.template update_transfers_and_footpaths<WithTdDest, WithTdFootpaths>(k);
  r.route_mark_.reset();
  r.rt_transport_mark_.reset();
}

// NOTE: the cache must be keyed by the kernel ADDRESS, not by the template
// parameter: every kernel sharing a signature - and most here are
// (impl, unsigned) - instantiates the SAME launch_dims, so a static-per-type
// cache hands them all the block size of whichever ran first. The heaviest
// kernels here use 138 registers, which caps a block at ~474 threads on
// sm_75, so inheriting a lighter kernel's 1024 is "too many resources
// requested for launch". Same fix as mc_launch_dims() in mcraptor.cu.
template <typename Kernel>
std::pair<int, int> launch_dims(Kernel kernel) {
  static thread_local std::unordered_map<void*, std::pair<int, int>> cache;
  auto const key = reinterpret_cast<void*>(kernel);
  if (auto const it = cache.find(key); it != end(cache)) {
    return it->second;
  }
  auto blocks = 0;
  auto threads = 0;
  // half + quarter benchmarked with less throughput
  cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, kernel, 0, 0);
  auto const dims = std::pair{blocks, threads};
  cache.emplace(key, dims);
  return dims;
}

template <typename Kernel, typename... Args>
void launch(Kernel kernel, cudaStream_t stream, Args&&... args) {
  auto const [blocks, threads] = launch_dims(kernel);
  kernel<<<blocks, threads, 0, stream>>>(std::forward<Args>(args)...);
}

template <typename Fn>
void dispatch_filtered(bool const with_clasz,
                       bool const is_wheelchair,
                       bool const with_filters,
                       Fn&& fn) {
  switch ((static_cast<unsigned>(with_clasz) << 2) |
          (static_cast<unsigned>(is_wheelchair) << 1) |
          static_cast<unsigned>(with_filters)) {
    case 0b000: fn.template operator()<false, false, false>(); break;
    case 0b001: fn.template operator()<false, false, true>(); break;
    case 0b011: fn.template operator()<false, true, true>(); break;
    case 0b100: fn.template operator()<true, false, false>(); break;
    case 0b101: fn.template operator()<true, false, true>(); break;
    case 0b111: fn.template operator()<true, true, true>(); break;
    default: __builtin_unreachable();  // no 010/110 (wheelchair => filters)
  }
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
  out[tid].state_ = reconstruction_result::kNotReconstructed;
  auto const k = tid % end_k;
  if (k == 0U) {
    return;
  }
  r.reconstruct_journey(dest_list[tid / end_k], k, &out[tid]);
}

// PHASE 2 bound build on the device. `at(i, l)` is the best round time at l
// over rounds 0..i, less the location's transfer buffer; the host version is
// build_reach_matrix() in bmrap_common.h and this must stay identical to it.
// Running it here means only the (budget + 1) rows the caller keeps cross
// PCIe rather than all (kMaxTransfers + 2) rounds of packed round times.
//
// The prefix runs over the raw round times, so a location's transfer buffer
// is subtracted exactly once - same as build_reach_matrix(), which this must
// stay identical to.
template <direction SearchDir>
__global__ void reach_bounds_kernel(device_times<SearchDir, 1U> round_times,
                                    device_timetable tt,
                                    transfer_time_settings const tts,
                                    delta_t* out,
                                    std::uint32_t const n_locations,
                                    unsigned const budget,
                                    bool const sub_transfer) {
  for (auto l = get_global_thread_id(); l < n_locations;
       l += get_global_stride()) {
    auto const li = location_idx_t{l};
    auto const buf =
        sub_transfer
            ? (kFwd ? 1 : -1) *
                  adjusted_transfer_time(tts, tt.transfer_time_[li].count())
            : 0;
    auto prefix = kInvalid;
    for (auto i = 0U; i <= budget; ++i) {
      auto const cur =
          round_times.get(static_cast<std::uint8_t>(i), li, via_offset_t{0U});
      // "better in SearchDir" == looser as a bound (build_reach_matrix's
      // is_looser); kInvalid is the worst value either way, so it loses
      prefix = (kFwd ? cur < prefix : cur > prefix) ? cur : prefix;
      out[i * n_locations + l] =
          prefix == kInvalid ? kInvalid : static_cast<delta_t>(prefix - buf);
    }
  }
}

// Provenance stamps for bmrap_bounds::device_tag_. Monotone and process-wide,
// so a stamp can never match a matrix built by a different state or an older
// generation of the same slot.
static std::uint64_t next_reach_tag() {
  static auto counter = std::atomic<std::uint64_t>{0U};
  return counter.fetch_add(1U, std::memory_order_relaxed) + 1U;
}

template <direction SearchDir>
void gpu_raptor<SearchDir>::build_reach_bounds(bmrap_bounds& out,
                                               std::uint8_t const budget,
                                               bool const sub_transfer) {
  auto& s = *state_.impl_;
  auto const slot = sub_transfer ? 1U : 0U;
  auto const n = static_cast<std::size_t>(n_locations_) * (budget + 1U);
  if (s.reach_out_[slot].size() < n) {
    s.reach_out_[slot].resize(n);
  }

  auto const [blocks, threads] = launch_dims(reach_bounds_kernel<SearchDir>);
  reach_bounds_kernel<SearchDir><<<blocks, threads, 0, s.stream_>>>(
      device_times<SearchDir, 1U>{to_mutable_view(s.round_times_),
                                  n_locations_},
      s.tt_, transfer_time_settings_,
      thrust::raw_pointer_cast(s.reach_out_[slot].data()), n_locations_, budget,
      sub_transfer);

  out.n_locations_ = n_locations_;
  out.budget_ = budget;
  out.device_tag_ = s.reach_tag_[slot] = next_reach_tag();
  s.reach_n_[slot] = n;
  // Straight into the caller's buffer: staging through pinned memory would
  // buy back some copy bandwidth and then spend more than that on the extra
  // host-to-host pass, and on a large timetable this is tens of MB. The host
  // copy itself is not optional - the multicriteria engines read `lat_`
  // directly unless they too are on the device.
  out.lat_.resize(n);
  auto const* const src = thrust::raw_pointer_cast(s.reach_out_[slot].data());
  CUDA_CHECK(cudaMemcpyAsync(out.lat_.data(), src, n * sizeof(delta_t),
                             cudaMemcpyDeviceToHost, s.stream_));
  CUDA_CHECK(cudaStreamSynchronize(s.stream_));
  CUDA_CHECK(cudaPeekAtLastError());
}

template <direction SearchDir>
void gpu_raptor<SearchDir>::execute(unixtime_t start_time,
                                    std::uint8_t max_transfers,
                                    unixtime_t worst_time_at_dest,
                                    profile_idx_t prf_idx,
                                    pareto_set<journey>& results) {
  auto& s = *state_.impl_;

  // BM-RAPTOR runs its pong and its pruning search in the same direction on
  // one state, so the later ctor overwrites the earlier one's destination
  // slot. Re-claim it here (a few KB) rather than giving every search its own
  // device state.
  if (s.q_owner_[kDirIdx] != this) {
    s.upload_query(kDirIdx, this, is_dest_, *dist_to_dest_, *td_dist_to_dest_);
  }

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

  auto const rt_active = gpu_rtt_ != nullptr;
  auto const with_td_dest = s.td_dest_locs_dev_[kDirIdx].size() > 0U;
  auto const with_td_fps =
      rt_active && prf_idx != 0U && gpu_rtt_->impl_->has_td_fps_[prf_idx];
  auto r = raptor_impl<SearchDir>{
      .bounds_ = has_bounds_
                     ? cuda::std::span<delta_t const>{
                           thrust::raw_pointer_cast(s.bmrap_bounds_.data()),
                           s.bmrap_bounds_.size()}
                     : cuda::std::span<delta_t const>{},
      .bounds_n_locations_ = bounds_n_locations_,
      .bounds_budget_ = bounds_budget_,
      .has_bounds_ = has_bounds_,
      .start_round_ = start_round_,
      .relax_origin_ = relax_on_ ? unix_to_delta(base(), relax_origin_)
                                 : delta_t{0},
      .relax_factor_ = relax_factor_,
      .relax_floor_ = relax_floor_,
      .relax_cap_ = relax_cap_,
      .relax_add_ = relax_add_,
      .relax_on_ = relax_on_,
      .any_marked_ = thrust::raw_pointer_cast(s.any_marked_.data()),
      .done_ = thrust::raw_pointer_cast(s.done_.data()),
      .tt_ = s.tt_,
      .rtt_ = rt_active ? gpu_rtt_->impl_->to_device_rt_timetable()
                        : device_rt_timetable{},
      .transfer_time_settings_ = transfer_time_settings_,
      .max_transfers_ = max_transfers,
      .allowed_claszes_ = allowed_claszes_,
      .prf_idx_ = prf_idx,
      .require_bike_transport_ = require_bike_transport_,
      .require_car_transport_ = require_car_transport_,
      .base_ = base_,
      .starts_ = starts,
      .is_dest_ = {to_view(s.is_dest_[kDirIdx])},
      .dist_to_end_ = to_view(s.dist_to_dest_dev_[kDirIdx]),
      .td_dest_locs_ = {s.td_dest_locs_dev_[kDirIdx].data(),
                        s.td_dest_locs_dev_[kDirIdx].size()},
      .td_dest_ = {.data_ = {s.td_dest_data_dev_[kDirIdx].data(),
                             s.td_dest_data_dev_[kDirIdx].size()},
                   .bucket_starts_ = {s.td_dest_ranges_dev_[kDirIdx].data(),
                                      s.td_dest_ranges_dev_[kDirIdx].size()}},
      .round_times_ = {to_mutable_view(s.round_times_), n_locations_},
      .best_ = {to_mutable_view(s.best_), n_locations_},
      .tmp_ = {to_mutable_view(s.tmp_), n_locations_},
      .time_at_dest_ = {to_mutable_view(s.time_at_dest_), n_locations_},
      .station_mark_ = {to_mutable_view(s.station_mark_)},
      .prev_station_mark_ = {to_mutable_view(s.prev_station_mark_)},
      .route_mark_ = {to_mutable_view(s.route_mark_)},
      .rt_transport_mark_ = {to_mutable_view(s.rt_transport_mark_)},
      .et_result_ = to_mutable_view(s.et_result_),
      .et_task_list_ = to_mutable_view(s.et_task_list_),
      .et_task_count_ = thrust::raw_pointer_cast(s.et_task_count_.data()),
      .route_list_ = to_mutable_view(s.route_list_),
      .route_list_count_ =
          thrust::raw_pointer_cast(s.route_list_count_.data())};

  if (rt_active) {
    r.tt_.transport_traffic_days_ = r.rtt_.transport_traffic_days_;
    r.tt_.bitfields_ = r.rtt_.bitfields_;
  }

  // mirrors raptor::execute(): a staggered run writes its round k into slot
  // k + start_round_, so the scan is shifted by the same amount and clipped
  // to the matrix
  auto const end_k = static_cast<std::uint32_t>(
      std::min(std::min(max_transfers, kMaxTransfers) + 2U + start_round_,
               static_cast<unsigned>(kMaxTransfers) + 2U));

  // === ROUTING KERNELS ===
  launch(init_arrivals_kernel<SearchDir>, s.stream_, r, worst_time_at_dest);
  for (auto k = start_round_ + 1U; k != end_k; ++k) {
    launch(reuse_previous_arrivals_kernel<SearchDir>, s.stream_, r, k);
    launch(mark_routes_kernel<SearchDir>, s.stream_, r, k);
    if (rt_active) {
      launch(mark_rt_transports_kernel<SearchDir>, s.stream_, r, k);
    }
    launch(begin_transit_phase_kernel<SearchDir>, s.stream_, r);
    launch(et_build_route_list_kernel<SearchDir>, s.stream_, r);
    if (is_wheelchair_) {
      launch(et_collect_tasks_kernel<SearchDir, true>, s.stream_, r, k);
    } else {
      launch(et_collect_tasks_kernel<SearchDir, false>, s.stream_, r, k);
    }
    launch(et_run_lookups_kernel<SearchDir>, s.stream_, r, k);
    {
      auto const with_clasz = allowed_claszes_ != all_clasz_allowed();
      auto const with_filters =
          is_wheelchair_ || require_bike_transport_ || require_car_transport_;
      dispatch_filtered(
          with_clasz, is_wheelchair_, with_filters,
          [&]<bool WithClasz, bool IsWheelchair, bool WithFilters>() {
            launch(loop_routes_kernel<SearchDir, WithClasz, IsWheelchair,
                                      WithFilters>,
                   s.stream_, r, k);
          });
      if (rt_active) {
        dispatch_filtered(
            with_clasz, is_wheelchair_, with_filters,
            [&]<bool WithClasz, bool IsWheelchair, bool WithFilters>() {
              launch(update_rt_transports_kernel<SearchDir, WithClasz,
                                                 IsWheelchair, WithFilters>,
                     s.stream_, r, k);
            });
      }
    }
    launch(begin_footpath_phase_kernel<SearchDir>, s.stream_, r);
    if (!with_td_dest && !with_td_fps) {
      launch(transfers_footpaths_kernel<SearchDir, false, false>, s.stream_, r,
             k);
    } else if (with_td_dest && !with_td_fps) {
      launch(transfers_footpaths_kernel<SearchDir, true, false>, s.stream_, r,
             k);
    } else if (!with_td_dest && with_td_fps) {
      launch(transfers_footpaths_kernel<SearchDir, false, true>, s.stream_, r,
             k);
    } else {
      launch(transfers_footpaths_kernel<SearchDir, true, true>, s.stream_, r,
             k);
    }
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
  auto const total = n_dest * end_k;

  auto* const dest_pin = s.rec_dest_pin_.ensure(dest_list.size());
  std::copy(dest_list.begin(), dest_list.end(), dest_pin);
  auto* const dest_dev = s.rec_dest_.ensure(dest_list.size(), s.stream_);
  CUDA_CHECK(cudaMemcpyAsync(dest_dev, dest_pin,
                             dest_list.size() * sizeof(location_idx_t),
                             cudaMemcpyHostToDevice, s.stream_));
  auto* const rec_out_dev = s.rec_out_.ensure(total, s.stream_);
  auto* const rec_host = s.rec_host_out_.ensure(total);

  {
    auto const threads = 128U;
    auto const blocks = (total + threads - 1U) / threads;
    reconstruct_kernel<SearchDir><<<blocks, threads, 0, s.stream_>>>(
        dest_dev, n_dest, end_k, r, rec_out_dev);
    CUDA_CHECK(cudaMemcpyAsync(rec_host, rec_out_dev,
                               total * sizeof(gpu_journey),
                               cudaMemcpyDeviceToHost, s.stream_));
    cudaStreamSynchronize(s.stream_);
  }
  CUDA_CHECK(cudaPeekAtLastError());

  // === CONVERT DEVICE JOURNEYS TO HOST JOURNEYS ===
  for (auto idx = std::uint32_t{0U}; idx != total; ++idx) {
    auto const& gj = rec_host[idx];

    if (gj.state_ == reconstruction_result::kReconstructionFailed) {
      // should not happen
      log(log_lvl::error, "search",
          "reconstruct failed: gpu breadcrumb chain unreconstructable "
          "(dest={}, k={})",
          to_idx(gj.dest_l_), idx % end_k);
      continue;
    }

    if (gj.state_ != reconstruction_result::kOk || gj.n_legs_ == 0U) {
      continue;
    }

    // Construct host journey from device journey.
    // This also reverses legs for forward search.
    // -> Legs are in chronological order from here on.
    auto j = journey{};
    j.start_time_ = start_time;
    j.dest_time_ = delta_to_unix(base(), gj.dest_time_);
    j.dest_ = gj.dest_l_;
    j.transfers_ = gj.transfers_;

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

    // Backward search requires to re-anchor footpath durations
    // to the arrival of the previous trip
    // instead of the departure of the next trip.
    // No-op for forward search.
    for (auto i = std::size_t{1U}; i < j.legs_.size(); ++i) {
      if (std::holds_alternative<footpath>(j.legs_[i].uses_)) {
        auto const dur = std::get<footpath>(j.legs_[i].uses_).duration();
        j.legs_[i].dep_time_ = j.legs_[i - 1U].arr_time_;
        j.legs_[i].arr_time_ = j.legs_[i].dep_time_ + dur;
      }
    }

    // Shorten td footpaths to their actual duration (excluding waiting).
    if (rtt_ != nullptr && prf_idx != 0U &&
        !(SearchDir == direction::kForward ? rtt_->td_footpaths_in_[prf_idx]
                                           : rtt_->td_footpaths_out_[prf_idx])
             .empty()) {
      constexpr auto const kIsFwd = SearchDir == direction::kForward;
      auto const& has_td = kIsFwd ? rtt_->has_td_footpaths_in_[prf_idx]
                                  : rtt_->has_td_footpaths_out_[prf_idx];
      auto const& td_fps = kIsFwd ? rtt_->td_footpaths_in_[prf_idx]
                                  : rtt_->td_footpaths_out_[prf_idx];
      for (auto& lg : j.legs_) {
        if (!std::holds_alternative<footpath>(lg.uses_)) {
          continue;
        }
        auto const key_l = kIsFwd ? lg.to_ : lg.from_;
        auto const target_l = kIsFwd ? lg.from_ : lg.to_;
        if (!has_td.test(key_l)) {
          continue;
        }
        auto const t = lg.arr_time_;
        for_each_footpath<SearchDir>(td_fps[key_l], t, [&](footpath const fp) {
          if (fp.target() != target_l) {
            return utl::cflow::kContinue;
          }
          lg.dep_time_ = t - fp.duration();
          lg.arr_time_ = t;
          lg.uses_ = footpath{lg.to_, fp.duration()};
          return utl::cflow::kBreak;
        });
      }
    }

    // is_reconstructed_ stays false
    // -> only reconstruct() called by pong/search.h might set it to true
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
  thrust::fill(thrust::cuda::par.on(state_.impl_->stream_),
               begin(state_.impl_->rt_transport_mark_),
               end(state_.impl_->rt_transport_mark_), 0U);
}

// First/last mile mumo offset and start footpath legs are added here
// on the host, where the query offsets live.
template <direction SearchDir>
void gpu_raptor<SearchDir>::reconstruct(query const& q, journey& j) {
  // The core legs (boarding station -> alighting station) reconstructed by the
  // GPU kernel (breadcrumb pointer chase).
  utl::verify(!j.legs_.empty(), "gpu reconstruct: journey without core legs");

  // The legs are chronological (materialization reverses forward device order).
  constexpr auto const is_fwd = SearchDir == direction::kForward;

  // Front-side mumo leg: special_station -> first transit stop.
  auto const from = j.legs_.front().from_;
  auto const dep_time = j.legs_.front().dep_time_;
  auto const front_match_mode =
      is_fwd ? q.start_match_mode_ : q.dest_match_mode_;
  if (front_match_mode == location_match_mode::kIntermodal) {
    auto const& offsets = is_fwd ? q.start_ : q.destination_;
    auto const& td_offsets = is_fwd ? q.td_start_ : q.td_dest_;
    auto const special = get_special_station(is_fwd ? special_station::kStart
                                                    : special_station::kEnd);
    auto const o = utl::find_if(offsets, [&](offset const& x) {
      return matches(tt_, front_match_mode, x.target(), from) &&
             (is_fwd
                  // fwd: query start, check feasibility (allows ontrip start)
                  ? dep_time - x.duration() >= j.start_time_
                  // bwd: destination, anchored exactly at j.dest_time_
                  : dep_time - x.duration() == j.dest_time_);
    });
    if (o != end(offsets)) {
      auto const dep = dep_time - o->duration();
      j.legs_.insert(begin(j.legs_), journey::leg{direction::kForward, special,
                                                  from, dep, dep_time, *o});
    } else {
      // td offset (e.g. flex), mirrors the CPU reconstruct:
      // - fwd = query start side (find_start_footpath): evaluated backward
      //   from the first transit departure, feasible if not before the
      //   journey start
      // - bwd = td egress side (get_legs intermodal dest): evaluated forward
      //   from the final label j.dest_time_
      // specify_td_offsets() refines both to the raw entry duration below.
      auto inserted = false;
      for (auto const& [target, tds] : td_offsets) {
        if (!matches(tt_, front_match_mode, target, from)) {
          continue;
        }
        if (is_fwd) {
          auto const fp = get_td_duration<direction::kBackward>(tds, dep_time);
          if (!fp.has_value() || dep_time - fp->first < j.start_time_) {
            continue;
          }
          j.legs_.insert(begin(j.legs_),
                         journey::leg{direction::kForward, special, from,
                                      dep_time - fp->first, dep_time,
                                      offset{target, fp->first,
                                             fp->second.transport_mode_id_}});
        } else {
          auto const t = j.dest_time_;
          auto const fp = get_td_duration<direction::kForward>(tds, t);
          if (!fp.has_value() || t + fp->first > dep_time) {
            continue;  // must reach the stop before the transit departure
          }
          j.legs_.insert(
              begin(j.legs_),
              journey::leg{
                  direction::kForward, special, from, t, t + fp->first,
                  offset{target, fp->first, fp->second.transport_mode_id_}});
        }
        inserted = true;
        break;
      }
      utl::verify(inserted, "gpu reconstruct: no front mumo offset");
    }
  }

  // offset: last transit stop -> special_station.
  auto const to = j.legs_.back().to_;
  auto const arr_time = j.legs_.back().arr_time_;
  auto const back_match_mode =
      is_fwd ? q.dest_match_mode_ : q.start_match_mode_;
  if (back_match_mode == location_match_mode::kIntermodal) {
    auto const& offsets = is_fwd ? q.destination_ : q.start_;
    auto const& td_offsets = is_fwd ? q.td_dest_ : q.td_start_;
    auto const special = get_special_station(is_fwd ? special_station::kEnd
                                                    : special_station::kStart);
    auto const o = utl::find_if(offsets, [&](offset const& x) {
      return matches(tt_, back_match_mode, x.target(), to) &&
             (is_fwd
                  // fwd: destination, anchored exactly at j.dest_time_
                  ? arr_time + x.duration() == j.dest_time_
                  // bwd: query start, anchored by feasibility
                  : arr_time + x.duration() <= j.start_time_);
    });
    if (o != end(offsets)) {
      auto const arr = arr_time + o->duration();
      j.legs_.push_back(
          journey::leg{direction::kForward, to, special, arr_time, arr, *o});
      j.dest_ = special;
    } else {
      // dest td offset
      auto inserted = false;
      for (auto const& [target, tds] : td_offsets) {
        if (!matches(tt_, back_match_mode, target, to)) {
          continue;
        }

        if (is_fwd) {
          auto const fp =
              get_td_duration<direction::kBackward>(tds, j.dest_time_);
          if (!fp.has_value() ||
              j.dest_time_ - fp->first /* duration*/ < arr_time) {
            continue;
          }

          j.legs_.push_back(journey::leg{
              direction::kForward, to, special, j.dest_time_ - fp->first,
              j.dest_time_,
              offset{target, fp->first, fp->second.transport_mode_id_}});
        } else {
          auto const fp = get_td_duration<direction::kForward>(tds, arr_time);
          if (!fp.has_value() || arr_time + fp->first > j.start_time_) {
            continue;
          }

          j.legs_.push_back(journey::leg{
              direction::kForward, to, special, arr_time, arr_time + fp->first,
              offset{target, fp->first, fp->second.transport_mode_id_}});
        }

        j.dest_ = special;
        inserted = true;

        break;
      }
      utl::verify(inserted, "gpu reconstruct: no back mumo offset");
    }
  }

  // use_start_footpaths_ == true:
  // reconstruct the start footpath that seeded round k=0 at the first stop.
  if (q.start_match_mode_ != location_match_mode::kIntermodal) {
    auto const is_journey_start = [&](location_idx_t const l) {
      for (auto const& o : q.start_) {
        if (matches(tt_, q.start_match_mode_, o.target(), l)) {
          return true;
        }
      }
      return false;
    };
    auto const start_l = is_fwd ? j.legs_.front().from_ : j.legs_.back().to_;
    auto const start_t =
        is_fwd ? j.legs_.front().dep_time_ : j.legs_.back().arr_time_;
    auto const direct_start_ok =
        is_fwd ? j.start_time_ <= start_t : j.start_time_ >= start_t;
    if (!is_journey_start(start_l) || !direct_start_ok) {
      auto const fps = is_fwd
                           ? tt_.locations_.footpaths_in_[q.prf_idx_][start_l]
                           : tt_.locations_.footpaths_out_[q.prf_idx_][start_l];
      auto best = std::optional<footpath>{};
      for (auto const fp : fps) {
        if ((!best.has_value() || fp.duration() < best->duration()) &&
            is_journey_start(fp.target())) {
          best = fp;
        }
      }
      if (best.has_value()) {
        auto const dur = duration_t{adjusted_transfer_time(
            q.transfer_time_settings_, best->duration().count())};
        auto const fp_arr = j.start_time_ + (is_fwd ? dur : -dur);
        if (is_fwd ? fp_arr <= start_t : fp_arr >= start_t) {
          auto const lg = journey::leg{
              SearchDir,     best->target(), start_l,
              j.start_time_, fp_arr,         footpath{best->target(), dur}};
          if (is_fwd) {
            j.legs_.insert(begin(j.legs_), lg);
          } else {
            j.legs_.push_back(lg);
          }
        }
      }
    }
  }

  if constexpr (is_fwd) {
    optimize_footpaths(tt_, rtt_, q, j);
    specify_td_offsets(q, j);
  } else {
    // The journey's legs are chronological, but q is in search direction.
    auto journey_q = q;
    journey_q.flip_dir();
    optimize_footpaths(tt_, rtt_, journey_q, j);
    specify_td_offsets(journey_q, j);
  }

  j.is_reconstructed_ = true;
}

template <direction SearchDir>
void gpu_raptor<SearchDir>::add_start(location_idx_t const l,
                                      unixtime_t const t) {
  starts_.emplace_back(l, t);
}

template <direction SearchDir>
void gpu_raptor<SearchDir>::set_bounds(bmrap_bounds const* b) {
  if (b == nullptr || b->empty()) {
    has_bounds_ = false;
    return;
  }
  auto& s = *state_.impl_;
  if (s.bmrap_bounds_.size() < b->lat_.size()) {
    s.bmrap_bounds_.resize(b->lat_.size());
  }
  // If this is a matrix we built ourselves it is still on the device, so take
  // it from there rather than pushing the host copy back over PCIe - which is
  // the common case for PHASE 2a's matrix feeding the PHASE 2b pruning search.
  auto const from_device = [&] {
    for (auto slot = 0U; slot != 2U; ++slot) {
      if (b->device_tag_ != 0U && b->device_tag_ == s.reach_tag_[slot] &&
          s.reach_n_[slot] == b->lat_.size()) {
        return static_cast<int>(slot);
      }
    }
    return -1;
  }();
  if (from_device >= 0) {
    CUDA_CHECK(cudaMemcpyAsync(
        thrust::raw_pointer_cast(s.bmrap_bounds_.data()),
        thrust::raw_pointer_cast(s.reach_out_[from_device].data()),
        b->lat_.size() * sizeof(delta_t), cudaMemcpyDeviceToDevice, s.stream_));
  } else {
    CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(s.bmrap_bounds_.data()),
                               b->lat_.data(),
                               b->lat_.size() * sizeof(delta_t),
                               cudaMemcpyHostToDevice, s.stream_));
  }
  CUDA_CHECK(cudaStreamSynchronize(s.stream_));
  bounds_n_locations_ = b->n_locations_;
  bounds_budget_ = b->budget_;
  has_bounds_ = true;
}

template <direction SearchDir>
void gpu_raptor<SearchDir>::set_dest_relax(unixtime_t const origin,
                                           double const factor,
                                           int const add_minutes,
                                           double const floor_min,
                                           double const cap_min) {
  relax_origin_ = origin;
  relax_factor_ = factor;
  relax_add_ = add_minutes;
  relax_cap_ = cap_min;
  relax_floor_ = std::min(floor_min, cap_min);
  relax_on_ = true;
}

template <direction SearchDir>
void gpu_raptor<SearchDir>::copy_round_times(
    std::vector<std::array<delta_t, 1>>& out) {
  auto& s = *state_.impl_;
  auto const n = static_cast<std::size_t>(n_locations_) * (kMaxTransfers + 2U);
  s.round_times_host_.resize(n);
  CUDA_CHECK(cudaMemcpyAsync(s.round_times_host_.data(),
                             thrust::raw_pointer_cast(s.round_times_.data()),
                             n * sizeof(std::uint64_t), cudaMemcpyDeviceToHost,
                             s.stream_));
  CUDA_CHECK(cudaStreamSynchronize(s.stream_));
  // unpack: the device stores (biased time key << 48) | breadcrumb
  out.resize(n);
  using times_t = device_times<SearchDir, 1U>;
  for (auto i = std::size_t{0U}; i != n; ++i) {
    auto const w = s.round_times_host_[i];
    out[i][0] = w == times_t::invalid_packed()
                    ? kInvalidDelta<SearchDir>
                    : times_t::from_key(static_cast<std::uint16_t>(w >> 48U));
  }
}

bool gpu_available() {
  auto n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

template class gpu_raptor<direction::kForward>;
template class gpu_raptor<direction::kBackward>;

}  // namespace nigiri::routing::gpu