#include "nigiri/routing/gpu/mcraptor.h"

#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <optional>
#include <unordered_map>

#include "cuda/std/span"

#include "thrust/device_vector.h"

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
#include "nigiri/routing/raptor/reconstruct.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/special_stations.h"

namespace nigiri::routing::gpu {

struct gpu_mcraptor_state::impl {
  explicit impl(gpu_timetable const& gtt)
      : tt_{gtt.impl_->to_device_timetable()} {
    cudaStreamCreate(&stream_);

    auto const n_locations = tt_.n_locations_;
    auto const env_cap = [](char const* name, std::uint32_t const def,
                            std::uint32_t const max) {
      auto const* v = std::getenv(name);
      auto const x = v == nullptr ? def
                                  : static_cast<std::uint32_t>(std::atoi(v));
      utl::verify(x >= 8U && x <= max, "{}={} out of range [8, {}]", name, x,
                  max);
      return x;
    };
    // inline slots per stop; the hwm byte bounds inline + overflow block
    bag_cap_ =
        env_cap("NIGIRI_GPU_MC_BAG_CAP", kMcBagCapDefault, 255U - kMcBagBlock);
    bags_.resize(static_cast<std::size_t>(n_locations) * bag_cap_);
    cudaMemsetAsync(thrust::raw_pointer_cast(bags_.data()), 0xFF,
                    bags_.size() * sizeof(mc_label_t), stream_);
    bag_pool_cap_ = env_cap(
        "NIGIRI_GPU_MC_BAG_POOL",
        static_cast<std::uint32_t>(std::min<std::size_t>(
            std::max<std::size_t>(n_locations / 16U, 131'072U),
            393'216U)),
        1U << 22U);
    bag_pool_.resize(static_cast<std::size_t>(bag_pool_cap_) * kMcBagBlock);
    cudaMemsetAsync(thrust::raw_pointer_cast(bag_pool_.data()), 0xFF,
                    bag_pool_.size() * sizeof(mc_label_t), stream_);
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

    // breadcrumb arena: transit arrivals + footpath copies of one start
    // time; overflow trips the device canary -> raise via env if it fires
    // (sized for the strict-dominance frontier: ~2x the bounded rule's)
    auto arena_cap = std::min<std::size_t>(
        std::max<std::size_t>(static_cast<std::size_t>(n_locations) * 16U,
                              8'000'000U),
        32'000'000U);
    if (auto const* env = std::getenv("NIGIRI_GPU_MC_ARENA");
        env != nullptr) {
      arena_cap = static_cast<std::size_t>(std::atoll(env));
    }
    bc_pay_lo_.resize(arena_cap);
    bc_hi_arr_.resize(arena_cap);
    bc_par_.resize(arena_cap);
    bc_count_.resize(1U);

    touched_.resize(n_locations / 32U + 1U);
    station_mark_.resize(n_locations / 32U + 1U);
    prev_station_mark_.resize(n_locations / 32U + 1U);
    route_mark_.resize(tt_.n_routes_ / 32U + 1U);
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
    // exact-size et rows: task list/map/offsets are per flat route-stop
    // (task count can never exceed that), the entry pool is reserved at
    // collect time as hwm+1 per task - it scales with the actual frontier
    // (measured ~3 entries/task mean) instead of a worst-case row width
    // measured WW peaks (n=20): 38M tasks, 26M arena entries; caps ~1.7x
    et_tasks_cap_ = env_cap(
        "NIGIRI_GPU_MC_ET_TASKS",
        static_cast<std::uint32_t>(
            std::min<std::size_t>(n_route_stops, 64'000'000U)),
        1U << 28U);
    et_task_list_.resize(et_tasks_cap_);
    et_task_off_.resize(et_tasks_cap_);
    et_task_count_.resize(1U);
    et_entry_count_.resize(1U);
    et_pool_cap_ = env_cap("NIGIRI_GPU_MC_ET_POOL",
                           static_cast<std::uint32_t>(std::min<std::size_t>(
                               std::max<std::size_t>(8U * n_route_stops,
                                                     64'000'000U),
                               224'000'000U)),
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

    // instrumentation: NIGIRI_GPU_MC_BAG_HIST=1 accumulates a bag
    // occupancy histogram (non-empty slots per touched bag, sampled after
    // the last round of every execute) and prints it at teardown
    hwm_stats_ = std::getenv("NIGIRI_GPU_MC_STATS") != nullptr;
    if (hwm_stats_) {
      hwm_stats_dev_.resize(3U);
      cudaMemsetAsync(thrust::raw_pointer_cast(hwm_stats_dev_.data()), 0,
                      3U * sizeof(std::uint32_t), stream_);
    }
    validate_ = std::getenv("NIGIRI_GPU_MC_VALIDATE") != nullptr;
    if (validate_) {
      validate_claim_.resize(bag_pool_cap_);
    }
    lock_stats_ = std::getenv("NIGIRI_GPU_MC_LOCK_STATS") != nullptr;
    if (lock_stats_) {
      lock_stats_dev_.resize(4U);
      cudaMemsetAsync(thrust::raw_pointer_cast(lock_stats_dev_.data()), 0,
                      4U * sizeof(unsigned long long), stream_);
    }
    seg_hist_ = std::getenv("NIGIRI_GPU_MC_SEG_HIST") != nullptr;
    if (seg_hist_) {
      seg_hist_dev_.resize(kMcMaxSegs + 2U);
      cudaMemsetAsync(thrust::raw_pointer_cast(seg_hist_dev_.data()), 0,
                      seg_hist_dev_.size() * sizeof(std::uint32_t), stream_);
      livebag_hist_dev_.resize(kMcRouteBagCap + 1U);
      cudaMemsetAsync(thrust::raw_pointer_cast(livebag_hist_dev_.data()), 0,
                      livebag_hist_dev_.size() * sizeof(std::uint32_t),
                      stream_);
      len_hist_dev_.resize(257U);
      cudaMemsetAsync(thrust::raw_pointer_cast(len_hist_dev_.data()), 0,
                      len_hist_dev_.size() * sizeof(std::uint32_t), stream_);
    }
    bag_hist_ = std::getenv("NIGIRI_GPU_MC_BAG_HIST") != nullptr;
    if (bag_hist_) {
      hist_dev_.resize(bag_cap_ + kMcBagBlock + 1U);
      cudaMemsetAsync(thrust::raw_pointer_cast(hist_dev_.data()), 0,
                      hist_dev_.size() * sizeof(std::uint32_t), stream_);
      hist_acc_.assign(bag_cap_ + kMcBagBlock + 1U, 0ULL);
    }
  }

  ~impl() {
    if (hwm_stats_) {
      auto v = std::vector<std::uint32_t>(3U);
      cudaMemcpy(v.data(), thrust::raw_pointer_cast(hwm_stats_dev_.data()),
                 3U * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
      auto c = std::uint32_t{0U};  // the last query's count was never swept
      cudaMemcpy(&c, thrust::raw_pointer_cast(bag_pool_count_.data()),
                 sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
      std::fprintf(stderr,
                   "MCHWM tasks_per_round=%u pool_entries_per_round=%u "
                   "arena_per_start=%u bag_blocks=%u\n",
                   v[0], v[1], v[2], std::max(bag_blocks_hwm_, c));
    }
    if (lock_stats_) {
      auto v = std::vector<unsigned long long>(4U);
      cudaMemcpy(v.data(), thrust::raw_pointer_cast(lock_stats_dev_.data()),
                 4U * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
      std::fprintf(stderr,
                   "MCLOCKSTATS acq=%llu cas_retries=%llu prescan_rej=%llu "
                   "accepts=%llu\n",
                   v[0], v[1], v[2], v[3]);
    }
    if (seg_hist_) {
      auto h = std::vector<std::uint32_t>(seg_hist_dev_.size());
      cudaMemcpy(h.data(), thrust::raw_pointer_cast(seg_hist_dev_.data()),
                 h.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
      std::fprintf(stderr, "MCSEGHIST fallback=%u\n", h[kMcMaxSegs + 1U]);
      for (auto n = std::size_t{1U}; n <= kMcMaxSegs; ++n) {
        if (h[n] != 0U) {
          std::fprintf(stderr, "MCSEGHIST %zu %u\n", n, h[n]);
        }
      }
      auto ln = std::vector<std::uint32_t>(len_hist_dev_.size());
      cudaMemcpy(ln.data(), thrust::raw_pointer_cast(len_hist_dev_.data()),
                 ln.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
      for (auto n = std::size_t{2U}; n != ln.size(); ++n) {
        if (ln[n] != 0U) {
          std::fprintf(stderr, "MCLENHIST %zu %u\n", n, ln[n]);
        }
      }
      auto lb = std::vector<std::uint32_t>(livebag_hist_dev_.size());
      cudaMemcpy(lb.data(),
                 thrust::raw_pointer_cast(livebag_hist_dev_.data()),
                 lb.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
      for (auto n = std::size_t{1U}; n != lb.size(); ++n) {
        if (lb[n] != 0U) {
          std::fprintf(stderr, "MCLIVEBAG %zu %u\n", n, lb[n]);
        }
      }
    }
    if (bag_hist_) {
      auto total = 0ULL;
      for (auto const v : hist_acc_) {
        total += v;
      }
      std::fprintf(stderr, "MCBAGHIST total_nonempty=%llu cap=%u\n", total,
                   bag_cap_);
      for (auto n = std::size_t{1U}; n != hist_acc_.size(); ++n) {
        if (hist_acc_[n] != 0ULL) {
          std::fprintf(stderr, "MCBAGHIST %zu %llu\n", n, hist_acc_[n]);
        }
      }
    }
    cudaStreamDestroy(stream_);
  }

  void upload_query(unsigned const dir,
                    nigiri::bitvec const& is_dest,
                    std::vector<std::uint16_t> const& dist_to_dest,
                    std::vector<std::uint16_t> const& lb) {
    is_intermodal_dest_[dir] = !dist_to_dest.empty();

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
    utl::verify(
        cudaSuccess ==
            cudaMemcpyAsync(
                thrust::raw_pointer_cast(dist_to_dest_dev_[dir].data()),
                dd_pin, dist_to_dest.size() * sizeof(std::uint16_t),
                cudaMemcpyHostToDevice, stream_),
        "gpu mcraptor: could not copy dist_to_dest");

    if (lb.empty()) {  // kUseLowerBounds=false experiment: inert zeros
      auto const n = static_cast<std::size_t>(tt_.n_locations_);
      cudaMemsetAsync(lb_dev_[dir].ensure(n, stream_), 0,
                      n * sizeof(std::uint16_t), stream_);
      return;
    }
    auto* const lb_pin = lb_pin_[dir].ensure(lb.size());
    std::copy(lb.begin(), lb.end(), lb_pin);
    // experiment switch: zero the lower bounds (= disable the lb
    // projection while keeping plain dest pruning and the host dijkstra)
    static bool const no_lb = std::getenv("NIGIRI_GPU_MC_NO_LB") != nullptr;
    if (no_lb) {
      std::fill(lb_pin, lb_pin + lb.size(), std::uint16_t{0U});
    }
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
  thrust::device_vector<std::uint32_t> bc_pay_lo_;
  thrust::device_vector<std::uint32_t> bc_hi_arr_;
  thrust::device_vector<std::uint32_t> bc_par_;
  thrust::device_vector<std::uint32_t> bc_count_;

  thrust::device_vector<std::uint32_t> touched_;
  thrust::device_vector<std::uint32_t> station_mark_;
  thrust::device_vector<std::uint32_t> prev_station_mark_;
  thrust::device_vector<std::uint32_t> route_mark_;

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
  thrust::device_vector<std::uint32_t> et_entry_count_;
  thrust::device_vector<std::uint32_t> route_entry_count_;
  thrust::device_vector<std::uint64_t> route_single_entry_;
  thrust::device_vector<std::uint32_t> route_single_flat_;
  thrust::device_vector<std::uint32_t> any_marked_;
  thrust::device_vector<std::uint32_t> done_;
  thrust::device_vector<std::uint32_t> overflow_;

  bool bag_hist_{false};
  thrust::device_vector<std::uint32_t> hist_dev_;
  std::vector<unsigned long long> hist_acc_;
  bool seg_hist_{false};
  bool hwm_stats_{false};
  thrust::device_vector<std::uint32_t> hwm_stats_dev_;
  std::uint32_t bag_blocks_hwm_{0U};
  // debug (NIGIRI_GPU_MC_VALIDATE): bag invariant checks + block claim map
  bool validate_{false};
  thrust::device_vector<std::uint32_t> validate_claim_;
  bool lock_stats_{false};
  thrust::device_vector<unsigned long long> lock_stats_dev_;
  thrust::device_vector<std::uint32_t> seg_hist_dev_;
  thrust::device_vector<std::uint32_t> livebag_hist_dev_;
  thrust::device_vector<std::uint32_t> len_hist_dev_;

  thrust::device_vector<std::uint64_t> is_dest_[2];
  pinned_host_buffer<std::uint64_t> is_dest_pin_[2];
  thrust::device_vector<std::uint16_t> dist_to_dest_dev_[2];
  pinned_host_buffer<std::uint16_t> dist_to_dest_pin_[2];
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

template <direction SearchDir, bool WithCost>
__global__ void mc_init_arrivals_kernel(mcraptor_impl<SearchDir, WithCost> r,
                                        delta_t const d_start) {
  r.init_arrivals(d_start);
}

template <direction SearchDir, bool WithCost>
__global__ void mc_begin_round_kernel(mcraptor_impl<SearchDir, WithCost> r) {
  if (*r.done_) {
    return;
  }
  if (get_global_thread_id() == 0U) {
    *r.any_marked_ = 0U;
  }
}

template <direction SearchDir, bool WithCost>
__global__ void mc_mark_routes_kernel(mcraptor_impl<SearchDir, WithCost> r) {
  if (*r.done_) {
    return;
  }
  r.mark_routes();
}

template <direction SearchDir, bool WithCost>
__global__ void mc_begin_transit_kernel(mcraptor_impl<SearchDir, WithCost> r) {
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

// debug (NIGIRI_GPU_MC_VALIDATE): bag-storage invariant checks.
// tag 0 = post-sweep: every stop must be pristine (hwm==0, no block).
// tag 1 = mid-round: a pool block may be owned by at most ONE stop and
// an owned block implies hwm > inline capacity.
template <direction SearchDir, bool WithCost>
__global__ void mc_bag_validate_kernel(mcraptor_impl<SearchDir, WithCost> r,
                                       std::uint32_t const n_locations,
                                       std::uint32_t* const claim,
                                       std::uint32_t const tag,
                                       std::uint32_t const round) {
  auto const gid = blockIdx.x * blockDim.x + threadIdx.x;
  auto const stride = gridDim.x * blockDim.x;
  for (auto l = gid; l < n_locations; l += stride) {
    auto const ovf = r.bag_ovf_[l];
    auto const hwm = static_cast<std::uint32_t>(r.bag_hwm_[l]);
    if (tag == 0U) {
      if (hwm != 0U || ovf != ~0U) {
        printf("VALIDBG post-sweep dirty l=%u hwm=%u ovf=%u\n", l, hwm, ovf);
      }
    } else if (ovf != ~0U) {
      if (hwm <= r.bag_cap_) {
        printf("VALIDBG k=%u stranded l=%u hwm=%u ovf=%u\n", round, l, hwm,
               ovf);
      }
      auto const prev = atomicExch(claim + ovf, l);
      if (prev != ~0U) {
        printf("VALIDBG k=%u DUP block=%u l1=%u l2=%u\n", round, ovf, prev,
               l);
      }
    }
  }
}

template <direction SearchDir, bool WithCost>
__global__ void mc_build_route_list_kernel(
    mcraptor_impl<SearchDir, WithCost> r) {
  if (*r.done_) {
    return;
  }
  r.build_route_list();
}

template <direction SearchDir, bool WithCost, bool IsWheelchair>
__global__ void mc_et_collect_kernel(mcraptor_impl<SearchDir, WithCost> r,
                                     unsigned const k) {
  if (*r.done_) {
    return;
  }
  r.template et_collect_tasks<IsWheelchair>(k);
}

template <direction SearchDir, bool WithCost>
__global__ void mc_et_lookups_kernel(mcraptor_impl<SearchDir, WithCost> r,
                                     unsigned const k) {
  if (*r.done_) {
    return;
  }
  r.et_run_lookups(k);
}

template <direction SearchDir, bool WithCost, bool WithClaszFilter,
          bool IsWheelchair>
__global__ void __launch_bounds__(kMcScanThreads)
    mc_scan_routes_kernel(mcraptor_impl<SearchDir, WithCost> r,
                          unsigned const k) {
  extern __shared__ mc_seg seg_smem[];  // kMcMaxSegs per warp
  if (*r.done_) {
    return;
  }
  r.template scan_routes<WithClaszFilter, IsWheelchair>(k, seg_smem);
}

template <direction SearchDir, bool WithCost>
__global__ void mc_begin_footpath_kernel(mcraptor_impl<SearchDir, WithCost> r) {
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

template <direction SearchDir, bool WithCost>
__global__ void mc_transfers_footpaths_kernel(
    mcraptor_impl<SearchDir, WithCost> r, unsigned const k) {
  if (*r.done_) {
    return;
  }
  r.update_transfers_and_footpaths(k);
  r.route_mark_.reset();
}

template <direction SearchDir, bool WithCost>
__global__ void mc_clear_bags_kernel(mcraptor_impl<SearchDir, WithCost> r) {
  r.clear_bags();
}

template <direction SearchDir, bool WithCost>
__global__ void mc_trace_kernel(mcraptor_impl<SearchDir, WithCost> r,
                                unsigned const k) {
  r.dump_traced(k);
}

// debug tracing (NIGIRI_MC_TRACE="l1,l2,..." + NIGIRI_MC_TRACE_START=
// unixtime minutes): dump the traced stops' bags after every round of the
// matching start
struct mc_trace_cfg {
  std::vector<location_idx_t> locs_;
  std::int64_t start_minutes_{-1};
};
inline mc_trace_cfg const& get_trace_cfg() {
  static auto const cfg = [] {
    auto c = mc_trace_cfg{};
    if (auto const* v = std::getenv("NIGIRI_MC_TRACE"); v != nullptr) {
      auto str = std::string{v};
      auto pos = std::size_t{0U};
      while (pos < str.size()) {
        auto end = str.find(',', pos);
        if (end == std::string::npos) {
          end = str.size();
        }
        c.locs_.push_back(location_idx_t{static_cast<std::uint32_t>(
            std::atoll(str.substr(pos, end - pos).c_str()))});
        pos = end + 1U;
      }
    }
    if (auto const* v = std::getenv("NIGIRI_MC_TRACE_START"); v != nullptr) {
      c.start_minutes_ = std::atoll(v);
    }
    return c;
  }();
  return cfg;
}

template <direction SearchDir, bool WithCost>
__global__ void mc_reconstruct_kernel(
    location_idx_t const* const dest_list,
    std::uint32_t const n_dest,
    mcraptor_impl<SearchDir, WithCost> r,
    gpu_journey* const out) {
  auto const tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto const per_dest = r.bag_cap_ + kMcBagBlock;
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

__global__ void mc_bag_hist_kernel(std::uint64_t const* const bags,
                                   std::uint64_t const* const pool,
                                   std::uint32_t const* const ovf,
                                   std::uint32_t const n_locations,
                                   std::uint32_t const cap,
                                   std::uint32_t* const hist) {
  auto const gid = blockIdx.x * blockDim.x + threadIdx.x;
  auto const stride = gridDim.x * blockDim.x;
  for (auto l = gid; l < n_locations; l += stride) {
    auto n = 0U;
    auto const* const b = bags + static_cast<std::size_t>(l) * cap;
    for (auto i = 0U; i != cap; ++i) {
      if (b[i] != kMcEmptySlot) {
        ++n;
      }
    }
    if (ovf[l] != ~0U) {
      auto const* const o =
          pool + static_cast<std::size_t>(ovf[l]) * kMcBagBlock;
      for (auto i = 0U; i != kMcBagBlock; ++i) {
        if (o[i] != kMcEmptySlot) {
          ++n;
        }
      }
    }
    if (n != 0U) {
      atomicAdd(hist + n, 1U);
    }
  }
}

// NOTE: the cache must be keyed by the kernel ADDRESS, not by the
// template parameter: all kernels sharing one signature instantiate the
// SAME mc_launch_dims, and a static-per-type cache would reuse the first
// kernel's occupancy for all of them (a 6-register kernel's 1024-thread
// block size launched a 70-register kernel -> "too many resources" on
// sm_75).
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

template <direction SearchDir, bool WithCost>
gpu_mcraptor<SearchDir, WithCost>::gpu_mcraptor(
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
    transfer_time_settings const& tts)
    : tt_{tt},
      n_locations_{tt_.n_locations()},
      state_{state},
      is_dest_{is_dest},
      base_{base},
      allowed_claszes_{allowed_claszes},
      is_wheelchair_{is_wheelchair},
      transfer_time_settings_{tts},
      worst_at_dest_{kInvalidDelta<SearchDir>} {
  utl::verify(via_stops.empty(), "gpu mcraptor: via stops not supported");
  utl::verify(td_dist_to_dest.empty(),
              "gpu mcraptor: time-dependent offsets not supported");
  utl::verify(!require_bike_transport && !require_car_transport,
              "gpu mcraptor: bike/car transport not supported");
  utl::verify(rtt == nullptr || rtt->n_rt_transports() == 0U,
              "gpu mcraptor: realtime not supported");
  utl::verify(lb.empty() || lb.size() == tt.n_locations(),
              "gpu mcraptor: lower bounds required (kUseLowerBounds)");
  reset_arrivals();
  state_.impl_->upload_query(kDirIdx, is_dest, dist_to_dest, lb);
}

template <direction SearchDir, bool WithCost>
void gpu_mcraptor<SearchDir, WithCost>::reset_arrivals() {
  auto& s = *state_.impl_;
  worst_at_dest_ = kInvalidDelta<SearchDir>;
  cudaMemsetAsync(thrust::raw_pointer_cast(s.dest_bag_[kDirIdx].data()), 0xFF,
                  s.dest_bag_[kDirIdx].size() * sizeof(mc_label_t), s.stream_);
  cudaMemsetAsync(thrust::raw_pointer_cast(s.reuse_bags_[kDirIdx].data()),
                  0xFF,
                  s.reuse_bags_[kDirIdx].size() * sizeof(std::uint64_t),
                  s.stream_);
  cudaMemsetAsync(thrust::raw_pointer_cast(s.dest_best_key_[kDirIdx].data()),
                  0xFF,
                  s.dest_best_key_[kDirIdx].size() * sizeof(std::uint32_t),
                  s.stream_);
  cudaMemsetAsync(
      thrust::raw_pointer_cast(s.dest_best_total_[kDirIdx].data()), 0xFF,
      s.dest_best_total_[kDirIdx].size() * sizeof(std::uint32_t), s.stream_);
  cudaMemsetAsync(thrust::raw_pointer_cast(s.overflow_.data()), 0,
                  sizeof(std::uint32_t), s.stream_);
}

template <direction SearchDir, bool WithCost>
mcraptor_impl<SearchDir, WithCost> make_impl(
    gpu_mcraptor_state::impl& s,
    unsigned const dir_idx,
    transfer_time_settings const& tts,
    clasz_mask_t const allowed_claszes,
    profile_idx_t const prf_idx,
    day_idx_t const base,
    delta_t const worst_at_dest,
    std::uint32_t const walk_surcharge,
    delta_t const dep,
    cuda::std::span<std::pair<location_idx_t, delta_t> const> starts,
    bool const reuse_same_dep = false) {
  return mcraptor_impl<SearchDir, WithCost>{
      .any_marked_ = thrust::raw_pointer_cast(s.any_marked_.data()),
      .done_ = thrust::raw_pointer_cast(s.done_.data()),
      .overflow_ = thrust::raw_pointer_cast(s.overflow_.data()),
      .seg_hist_ = s.seg_hist_
                       ? thrust::raw_pointer_cast(s.seg_hist_dev_.data())
                       : nullptr,
      .hwm_stats_ =
          s.hwm_stats_ ? thrust::raw_pointer_cast(s.hwm_stats_dev_.data())
                       : nullptr,
      .lock_stats_ =
          s.lock_stats_
              ? thrust::raw_pointer_cast(s.lock_stats_dev_.data())
              : nullptr,
      .livebag_hist_ =
          s.seg_hist_ ? thrust::raw_pointer_cast(s.livebag_hist_dev_.data())
                      : nullptr,
      .len_hist_ = s.seg_hist_
                       ? thrust::raw_pointer_cast(s.len_hist_dev_.data())
                       : nullptr,
      .tt_ = s.tt_,
      .transfer_time_settings_ = tts,
      .allowed_claszes_ = allowed_claszes,
      .prf_idx_ = prf_idx,
      .base_ = base,
      .worst_at_dest_ = worst_at_dest,
      .walk_surcharge_ = walk_surcharge,
      .starts_ = starts,
      .is_dest_ = {to_view(s.is_dest_[dir_idx])},
      .dist_to_end_ = to_view(s.dist_to_dest_dev_[dir_idx]),
      .lb_ = {s.lb_dev_[dir_idx].data(), s.lb_dev_[dir_idx].size()},
      .bags_ = thrust::raw_pointer_cast(s.bags_.data()),
      .bag_pool_ = thrust::raw_pointer_cast(s.bag_pool_.data()),
      .bag_ovf_ = thrust::raw_pointer_cast(s.bag_ovf_.data()),
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
      .dest_prune_disabled_ =
          [] {
            static bool const v =
                std::getenv("NIGIRI_NO_DEST_PRUNING") != nullptr;
            return v;
          }(),
      .prefix_disabled_ =
          [] {
            static bool const v =
                std::getenv("NIGIRI_GPU_MC_NO_PREFIX") != nullptr;
            return v;
          }(),
      .reuse_disabled_ =
          [] {
            static bool const v =
                std::getenv("NIGIRI_GPU_MC_NO_REUSE") != nullptr;
            return v;
          }(),
      .reuse_same_dep_only_ = reuse_same_dep,
      .bc_pay_lo_ = thrust::raw_pointer_cast(s.bc_pay_lo_.data()),
      .bc_hi_arr_ = thrust::raw_pointer_cast(s.bc_hi_arr_.data()),
      .bc_par_ = thrust::raw_pointer_cast(s.bc_par_.data()),
      .bc_count_ = thrust::raw_pointer_cast(s.bc_count_.data()),
      .bc_cap_ = static_cast<std::uint32_t>(s.bc_pay_lo_.size()),
      .touched_ = {to_mutable_view(s.touched_)},
      .station_mark_ = {to_mutable_view(s.station_mark_)},
      .prev_station_mark_ = {to_mutable_view(s.prev_station_mark_)},
      .route_mark_ = {to_mutable_view(s.route_mark_)},
      .route_list_ = to_mutable_view(s.route_list_),
      .route_list_count_ =
          thrust::raw_pointer_cast(s.route_list_count_.data()),
      .et_task_list_ = to_mutable_view(s.et_task_list_),
      .et_task_count_ = thrust::raw_pointer_cast(s.et_task_count_.data()),
      .et_ent_key_ = thrust::raw_pointer_cast(s.et_ent_key_.data()),
      .et_ent_ex_ = thrust::raw_pointer_cast(s.et_ent_ex_.data()),
      .et_ent_sl_ = thrust::raw_pointer_cast(s.et_ent_sl_.data()),
      .task_bits_ = thrust::raw_pointer_cast(s.task_bits_.data()),
      .route_task_start_ =
          thrust::raw_pointer_cast(s.route_task_start_.data()),
      .et_task_off_ = thrust::raw_pointer_cast(s.et_task_off_.data()),
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

template <direction SearchDir, bool WithCost>
void gpu_mcraptor<SearchDir, WithCost>::next_start_time() {
  starts_.clear();
  auto& s = *state_.impl_;
  auto const r = make_impl<SearchDir, WithCost>(
      s, kDirIdx, transfer_time_settings_, allowed_claszes_, 0U, base_,
      worst_at_dest_, 0U, 0, {});
  mc_launch(mc_clear_bags_kernel<SearchDir, WithCost>, s.stream_, r);
  if (s.hwm_stats_) {  // counter is monotone within a query - read = max
    auto c = std::uint32_t{0U};
    cudaMemcpyAsync(&c, thrust::raw_pointer_cast(s.bag_pool_count_.data()),
                    sizeof(std::uint32_t), cudaMemcpyDeviceToHost, s.stream_);
    cudaStreamSynchronize(s.stream_);
    s.bag_blocks_hwm_ = std::max(s.bag_blocks_hwm_, c);
  }
  cudaMemsetAsync(thrust::raw_pointer_cast(s.bag_pool_count_.data()), 0,
                  sizeof(std::uint32_t), s.stream_);
  if (s.validate_) {  // debug: everything must be pristine after the sweep
    mc_bag_validate_kernel<SearchDir, WithCost>
        <<<512, 256, 0, s.stream_>>>(r, n_locations_, nullptr, 0U, 0U);
  }
  thrust::fill(thrust::cuda::par.on(s.stream_), s.station_mark_.begin(),
               s.station_mark_.end(), 0U);
  thrust::fill(thrust::cuda::par.on(s.stream_), s.prev_station_mark_.begin(),
               s.prev_station_mark_.end(), 0U);
  thrust::fill(thrust::cuda::par.on(s.stream_), s.route_mark_.begin(),
               s.route_mark_.end(), 0U);
}

template <direction SearchDir, bool WithCost>
void gpu_mcraptor<SearchDir, WithCost>::add_start(location_idx_t const l,
                                                  unixtime_t const t) {
  starts_.emplace_back(l, t);
}

template <direction SearchDir, bool WithCost>
void gpu_mcraptor<SearchDir, WithCost>::execute(
    unixtime_t const start_time,
    std::uint8_t const max_transfers,
    unixtime_t const worst_time_at_dest,
    profile_idx_t const prf_idx,
    pareto_set<journey>& results) {
  auto& s = *state_.impl_;
  constexpr auto const kFwd = SearchDir == direction::kForward;

  auto const d_worst = unix_to_delta(base(), worst_time_at_dest);
  worst_at_dest_ = kFwd ? std::min(d_worst, worst_at_dest_)
                        : std::max(d_worst, worst_at_dest_);

  // mirror of the CPU arr_cost_criteria::kWalkSurcharge configuration
  auto const walk_surcharge =
      WithCost ? arr_cost_criteria::kWalkSurcharge : 0U;

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
  auto const r = make_impl<SearchDir, WithCost>(
      s, kDirIdx, transfer_time_settings_, allowed_claszes_, prf_idx, base_,
      worst_at_dest_, walk_surcharge, d_start_dep,
      cuda::std::span<std::pair<location_idx_t, delta_t> const>{
          starts_dev, starts_.size()},
      reuse_same_dep_);

  auto const end_k =
      static_cast<std::uint32_t>(std::min(max_transfers, kMaxTransfers) + 2U);
  auto const d_start = unix_to_delta(base(), start_time);

  // debug tracing: enabled iff this execute's start matches the env
  auto const& trace_cfg = get_trace_cfg();
  auto trace_locs_dev = thrust::device_vector<location_idx_t>{};
  auto trace = r;
  auto const tracing =
      !trace_cfg.locs_.empty() &&
      (trace_cfg.start_minutes_ < 0 ||
       trace_cfg.start_minutes_ == start_time.time_since_epoch().count());
  if (tracing) {
    trace_locs_dev.assign(trace_cfg.locs_.begin(), trace_cfg.locs_.end());
    trace.trace_locs_ = {thrust::raw_pointer_cast(trace_locs_dev.data()),
                         trace_locs_dev.size()};
    std::printf("GPUTRACE begin start=%lld worst=%d\n",
                static_cast<long long>(start_time.time_since_epoch().count()),
                static_cast<int>(worst_at_dest_));
  }

  // === ROUTING KERNELS ===
  // (all launches use the `trace` copy: identical to r, plus trace_locs_
  // when NIGIRI_MC_TRACE is active - the debug probes live in the phases)
  mc_launch(mc_init_arrivals_kernel<SearchDir, WithCost>, s.stream_, trace,
            d_start);
  for (auto k = 1U; k != end_k; ++k) {
    mc_launch(mc_begin_round_kernel<SearchDir, WithCost>, s.stream_, trace);
    mc_launch(mc_mark_routes_kernel<SearchDir, WithCost>, s.stream_, trace);
    mc_launch(mc_begin_transit_kernel<SearchDir, WithCost>, s.stream_, trace);
    mc_launch(mc_build_route_list_kernel<SearchDir, WithCost>, s.stream_,
              trace);
    if (is_wheelchair_) {
      mc_launch(mc_et_collect_kernel<SearchDir, WithCost, true>, s.stream_,
                trace, k);
    } else {
      mc_launch(mc_et_collect_kernel<SearchDir, WithCost, false>, s.stream_,
                trace, k);
    }
    mc_launch(mc_et_lookups_kernel<SearchDir, WithCost>, s.stream_, trace, k);
    // warp-per-route two-pass scan: fixed geometry + per-warp shared
    // segment slab (occupancy-launch cannot size dynamic shared memory)
    auto const scan_blocks = 512U;
    auto const scan_shared =
        (kMcScanThreads / 32U) * kMcMaxSegs * sizeof(mc_seg);
    auto const with_clasz = allowed_claszes_ != all_clasz_allowed();
    if (with_clasz) {
      if (is_wheelchair_) {
        mc_scan_routes_kernel<SearchDir, WithCost, true, true>
            <<<scan_blocks, kMcScanThreads, scan_shared, s.stream_>>>(trace,
                                                                      k);
      } else {
        mc_scan_routes_kernel<SearchDir, WithCost, true, false>
            <<<scan_blocks, kMcScanThreads, scan_shared, s.stream_>>>(trace,
                                                                      k);
      }
    } else {
      if (is_wheelchair_) {
        mc_scan_routes_kernel<SearchDir, WithCost, false, true>
            <<<scan_blocks, kMcScanThreads, scan_shared, s.stream_>>>(trace,
                                                                      k);
      } else {
        mc_scan_routes_kernel<SearchDir, WithCost, false, false>
            <<<scan_blocks, kMcScanThreads, scan_shared, s.stream_>>>(trace,
                                                                      k);
      }
    }
    mc_launch(mc_begin_footpath_kernel<SearchDir, WithCost>, s.stream_,
              trace);
    mc_launch(mc_transfers_footpaths_kernel<SearchDir, WithCost>, s.stream_,
              trace, k);
    if (s.validate_) {  // debug: block ownership must be unique per stop
      cudaMemsetAsync(thrust::raw_pointer_cast(s.validate_claim_.data()),
                      0xFF, s.validate_claim_.size() * sizeof(std::uint32_t),
                      s.stream_);
      mc_bag_validate_kernel<SearchDir, WithCost><<<512, 256, 0, s.stream_>>>(
          trace, n_locations_,
          thrust::raw_pointer_cast(s.validate_claim_.data()), 1U, k);
    }
    if (tracing) {
      mc_trace_kernel<SearchDir, WithCost><<<1, 1, 0, s.stream_>>>(trace, k);
    }
  }
  cudaStreamSynchronize(s.stream_);
  CUDA_CHECK(cudaPeekAtLastError());

  if (s.bag_hist_) {
    mc_bag_hist_kernel<<<512, 256, 0, s.stream_>>>(
        thrust::raw_pointer_cast(s.bags_.data()),
        thrust::raw_pointer_cast(s.bag_pool_.data()),
        thrust::raw_pointer_cast(s.bag_ovf_.data()), s.tt_.n_locations_,
        s.bag_cap_, thrust::raw_pointer_cast(s.hist_dev_.data()));
    auto hist = std::vector<std::uint32_t>(s.bag_cap_ + kMcBagBlock + 1U);
    CUDA_CHECK(cudaMemcpyAsync(
        hist.data(), thrust::raw_pointer_cast(s.hist_dev_.data()),
        hist.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
        s.stream_));
    cudaStreamSynchronize(s.stream_);
    for (auto n = std::size_t{0U}; n != hist.size(); ++n) {
      s.hist_acc_[n] += hist[n];
    }
    cudaMemsetAsync(thrust::raw_pointer_cast(s.hist_dev_.data()), 0,
                    hist.size() * sizeof(std::uint32_t), s.stream_);
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
    return;
  }

  auto const n_dest = static_cast<std::uint32_t>(dest_list.size());
  auto const total = n_dest * (s.bag_cap_ + kMcBagBlock);

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
    mc_reconstruct_kernel<SearchDir, WithCost>
        <<<blocks, threads, 0, s.stream_>>>(dest_dev, n_dest, r, rec_out_dev);
    CUDA_CHECK(cudaMemcpyAsync(rec_host, rec_out_dev,
                               total * sizeof(gpu_journey),
                               cudaMemcpyDeviceToHost, s.stream_));
    CUDA_CHECK(cudaMemcpyAsync(overflow_pin,
                               thrust::raw_pointer_cast(s.overflow_.data()),
                               sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                               s.stream_));
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
    if constexpr (WithCost) {
      // full generalized cost: common elapsed part + stored extras
      j.criteria_cost_ = static_cast<std::uint16_t>(
          gj.criteria_cost_ +
          static_cast<std::uint16_t>(j.travel_time().count()));
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

    // Backward search: re-anchor footpath durations to the arrival of the
    // previous trip (no-op for forward search).
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

// First/last mile mumo offset and start footpath legs are added here on the
// host, where the query offsets live (mirror of the CPU
// mcraptor::reconstruct, transit-anchored offset legs).
template <direction SearchDir, bool WithCost>
void gpu_mcraptor<SearchDir, WithCost>::reconstruct(query const& q,
                                                    journey& j) {
  utl::verify(!j.legs_.empty(),
              "gpu mcraptor reconstruct: journey without legs");

  constexpr auto const is_fwd = SearchDir == direction::kForward;

  // Front-side mumo leg: special_station -> first transit stop.
  auto const from = j.legs_.front().from_;
  auto const dep_time = j.legs_.front().dep_time_;
  auto const front_match_mode =
      is_fwd ? q.start_match_mode_ : q.dest_match_mode_;
  if (front_match_mode == location_match_mode::kIntermodal) {
    auto const& offsets = is_fwd ? q.start_ : q.destination_;
    auto const special = get_special_station(is_fwd ? special_station::kStart
                                                    : special_station::kEnd);
    auto const o = utl::find_if(offsets, [&](offset const& x) {
      return matches(tt_, front_match_mode, x.target(), from) &&
             (is_fwd ? dep_time - x.duration() >= j.start_time_
                     : dep_time - x.duration() == j.dest_time_);
    });
    utl::verify(o != end(offsets),
                "gpu mcraptor reconstruct: no front offset");
    auto const dep = dep_time - o->duration();
    j.legs_.insert(begin(j.legs_), journey::leg{direction::kForward, special,
                                                from, dep, dep_time, *o});
  }

  // Back-side mumo leg: last transit stop -> special_station.
  auto const to = j.legs_.back().to_;
  auto const arr_time = j.legs_.back().arr_time_;
  auto const back_match_mode =
      is_fwd ? q.dest_match_mode_ : q.start_match_mode_;
  if (back_match_mode == location_match_mode::kIntermodal) {
    auto const& offsets = is_fwd ? q.destination_ : q.start_;
    auto const special = get_special_station(is_fwd ? special_station::kEnd
                                                    : special_station::kStart);
    auto const o = utl::find_if(offsets, [&](offset const& x) {
      return matches(tt_, back_match_mode, x.target(), to) &&
             (is_fwd ? arr_time + x.duration() == j.dest_time_
                     : arr_time + x.duration() <= j.start_time_);
    });
    utl::verify(o != end(offsets), "gpu mcraptor reconstruct: no back offset");
    j.legs_.push_back(journey::leg{direction::kForward, to, special, arr_time,
                                   arr_time + o->duration(), *o});
    j.dest_ = special;
  }

  // Reconstruct the start footpath that seeded round k=0 at the first stop.
  if (q.start_match_mode_ != location_match_mode::kIntermodal) {
    auto const is_journey_start = [&](location_idx_t const l) {
      return utl::any_of(q.start_, [&](offset const& o) {
        return matches(tt_, q.start_match_mode_, o.target(), l);
      });
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
    optimize_footpaths(tt_, nullptr, q, j);
  } else {
    auto journey_q = q;
    journey_q.flip_dir();
    optimize_footpaths(tt_, nullptr, journey_q, j);
  }

  j.is_reconstructed_ = true;
}

template class gpu_mcraptor<direction::kForward, false>;
template class gpu_mcraptor<direction::kBackward, false>;
template class gpu_mcraptor<direction::kForward, true>;
template class gpu_mcraptor<direction::kBackward, true>;

}  // namespace nigiri::routing::gpu
