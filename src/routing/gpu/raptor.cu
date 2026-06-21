#include "nigiri/routing/gpu/raptor.h"

#include <algorithm>
#include <cstdio>
#include <iostream>

#include "cuda/std/array"
#include "cuda/std/span"

#include "cooperative_groups.h"

#include "thrust/copy.h"
#include "thrust/device_vector.h"
#include "thrust/fill.h"
#include "thrust/host_vector.h"

#include "nigiri/routing/gpu/device_timetable.cuh"
#include "nigiri/routing/gpu/raptor_impl.cuh"
#include "nigiri/routing/gpu/types.cuh"
#include "utl/timer.h"

namespace cg = cooperative_groups;

namespace nigiri::routing::gpu {

// Pinned host staging buffer (raw cudaMallocHost) -> fast, async-capable H2D
// transfers vs slow pageable memory. Grows on demand, freed on destruction.
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

// packed time||breadcrumb sentinel: all-ones. The high 16 bits are 0xFFFF =
// worst time key (== kInvalid for both directions, atomicMin always beats it);
// the low 48 breadcrumb bits are 1s but are only ever read for valid entries.
// Making it all-ones lets reset_arrivals/next_start_time clear with a single
// cudaMemset(0xFF) instead of a thrust::fill kernel.
static constexpr std::uint64_t kInvalidPacked = ~std::uint64_t{0};

#define CUDA_CHECK(code)                                              \
  if ((code) != cudaSuccess) {                                        \
    std::cerr << "CUDA error: " << cudaGetErrorString(code) << " at " \
              << __FILE__ << ":" << __LINE__;                         \
    std::terminate();                                                 \
  }

struct gpu_timetable::impl {
  using t = timetable;

  static std::vector<std::uint32_t> build_route_stop_offset(timetable const& tt) {
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
        route_bikes_allowed_{to_device(tt.route_bikes_allowed_)},
        route_bikes_allowed_per_section_{tt.route_bikes_allowed_per_section_},
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
            .route_bikes_allowed_ = {to_view(route_bikes_allowed_)},
            .route_bikes_allowed_per_section_ =
                to_view(route_bikes_allowed_per_section_),
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
  thrust::device_vector<std::uint64_t> route_bikes_allowed_;
  device_vecvec<decltype(t{}.route_bikes_allowed_per_section_)>
      route_bikes_allowed_per_section_;

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

struct gpu_rt_timetable::impl {
  using rtt = rt_timetable;

  thrust::device_vector<bitfield_idx_t> transport_traffic_days_;
  thrust::device_vector<bitfield> bitfields_;
  device_vecvec<decltype(rtt{}.rt_transport_stop_times_)>
      rt_transport_stop_times_;
  device_vecvec<decltype(rtt{}.rt_transport_location_seq_)>
      rt_transport_location_seq_;
  thrust::device_vector<std::uint64_t> rt_transport_bikes_allowed_;
  device_vecvec<decltype(rtt{}.rt_bikes_allowed_per_section_)>
      rt_bikes_allowed_per_section_;
  device_vecvec<vecvec<location_idx_t, rt_transport_idx_t>>
      location_rt_transports_;
  date::sys_days base_day_;
  day_idx_t base_day_idx_;
};

struct gpu_raptor_state::impl {
  explicit impl(gpu_timetable const& gtt)
      : tt_{gtt.impl_->to_device_timetable()} {
    cudaStreamCreate(&stream_);
    // compute_et boarding pre-pass buffers, sized to the flat route-stop space.
    auto const n_route_stops = tt_.route_of_stop_.size();
    et_result_.resize(n_route_stops);
    et_task_list_.resize(n_route_stops);
    et_task_count_.resize(1U);
  }

  ~impl() { cudaStreamDestroy(stream_); }

  void resize(unsigned n_locations,
              unsigned n_routes,
              unsigned n_rt_transports,
              std::array<nigiri::bitvec, kMaxVias> const& is_via,
              std::vector<via_stop> const& via_stops,
              nigiri::bitvec const& is_dest,
              std::vector<std::uint16_t> const& dist_to_dest) {
    n_locations_ = host_state_.n_locations_ = n_locations;

    // The GPU search is station-to-station only (Vias == 0), so the state needs
    // just one via-slot, not kMaxVias+1. Sizing to 1 cuts round_times/best/tmp
    // ~3x — important to fit large timetables on small GPUs.
    static constexpr auto kGpuViaSlots = std::uint32_t{1};
    time_at_dest_.resize(kMaxTransfers + 2);
    tmp_.resize(n_locations * kGpuViaSlots);
    best_.resize(n_locations * kGpuViaSlots);
    auto const first_time = round_times_.empty();
    round_times_.resize(n_locations * kGpuViaSlots * (kMaxTransfers + 2));
    host_round_times_.resize(round_times_.size());
    // selective-clear: 1 dirty bit per entry for round_times / best_ / tmp_;
    // reset only clears the marked entries. Needs an initial full clear so the
    // first search starts all-invalid (and the bitfields start all-zero).
    dirty_bits_.resize((round_times_.size() + 31U) / 32U);
    best_dirty_bits_.resize((best_.size() + 31U) / 32U);
    tmp_dirty_bits_.resize((tmp_.size() + 31U) / 32U);
    cudaMemsetAsync(thrust::raw_pointer_cast(dirty_bits_.data()), 0,
                    dirty_bits_.size() * sizeof(std::uint32_t), stream_);
    cudaMemsetAsync(thrust::raw_pointer_cast(best_dirty_bits_.data()), 0,
                    best_dirty_bits_.size() * sizeof(std::uint32_t), stream_);
    cudaMemsetAsync(thrust::raw_pointer_cast(tmp_dirty_bits_.data()), 0,
                    tmp_dirty_bits_.size() * sizeof(std::uint32_t), stream_);
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
    rt_transport_mark_.resize(n_rt_transports / 32U + 1U);
    end_reachable_.resize(n_locations / 32U + 1U);

    // lower bounds are intentionally not used on the GPU (no dijkstra; the
    // search explores without lb pruning -- same results, and it avoids the
    // per-query upload).

    any_marked_.resize(1U);

    // Per-search params (is_dest/is_via/via_stops/dist_to_dest) live in the
    // shared state but differ per gpu_raptor instance. upload_query() (re)uploads
    // them; execute() calls it too, so a sibling instance sharing this state
    // (e.g. pong's ping vs pong) can't leak its destination into our search.
    upload_query(is_via, via_stops, is_dest, dist_to_dest);
  }

  void upload_query(std::array<nigiri::bitvec, kMaxVias> const& is_via,
                    std::vector<via_stop> const& via_stops,
                    nigiri::bitvec const& is_dest,
                    std::vector<std::uint16_t> const& dist_to_dest) {
    is_intermodal_dest_ = !dist_to_dest.empty();

    for (auto i = 0U; i != is_via.size(); ++i) {
      is_via_[i].resize(is_via[i].blocks_.size());
      utl::verify(
          cudaSuccess ==
              cudaMemcpyAsync(
                  thrust::raw_pointer_cast(is_via_[i].data()),
                  is_via[i].blocks_.data(),
                  is_via[i].blocks_.size() *
                      sizeof(std::decay_t<decltype(is_via[0])>::block_t),
                  cudaMemcpyHostToDevice, stream_),
          "could not copy is_via[{}]", i);
    }

    via_stops_.resize(via_stops.size());
    utl::verify(cudaSuccess ==
                    cudaMemcpyAsync(thrust::raw_pointer_cast(via_stops_.data()),
                                    via_stops.data(),
                                    via_stops.size() * sizeof(via_stop),
                                    cudaMemcpyHostToDevice, stream_),
                "could not copy via_stops");

    is_dest_.resize(is_dest.blocks_.size());
    utl::verify(
        cudaSuccess == cudaMemcpyAsync(
                           thrust::raw_pointer_cast(is_dest_.data()),
                           is_dest.blocks_.data(),
                           is_dest.blocks_.size() *
                               sizeof(std::decay_t<decltype(is_dest)>::block_t),
                           cudaMemcpyHostToDevice, stream_),
        "could not copy is_dest");

    dist_to_dest_.resize(dist_to_dest.size());
    auto* const dd_pinned = dist_to_dest_pinned_.ensure(dist_to_dest.size());
    std::copy(dist_to_dest.begin(), dist_to_dest.end(), dd_pinned);
    utl::verify(
        cudaSuccess ==
            cudaMemcpyAsync(thrust::raw_pointer_cast(dist_to_dest_.data()),
                            dd_pinned,
                            dist_to_dest.size() * sizeof(std::uint16_t),
                            cudaMemcpyHostToDevice, stream_),
        "could not copy dist to dest");
  }

  std::uint32_t n_locations_;
  bool is_intermodal_dest_;
  thrust::device_vector<std::uint32_t> any_marked_;
  // packed time||breadcrumb words (see device_times / breadcrumb.h)
  thrust::device_vector<std::uint64_t> time_at_dest_;
  thrust::device_vector<std::uint64_t> tmp_;
  thrust::device_vector<std::uint64_t> best_;
  thrust::device_vector<std::uint64_t> round_times_;
  thrust::device_vector<std::uint32_t> dirty_bits_;  // 1 bit / round_times entry
  thrust::device_vector<std::uint32_t> best_dirty_bits_;  // 1 bit / best_ entry
  thrust::device_vector<std::uint32_t> tmp_dirty_bits_;  // 1 bit / tmp_ entry
  thrust::device_vector<std::uint32_t> station_mark_;
  thrust::device_vector<std::uint32_t> prev_station_mark_;
  thrust::device_vector<std::uint32_t> route_mark_;
  thrust::device_vector<std::uint32_t> rt_transport_mark_;

  thrust::device_vector<std::uint32_t> end_reachable_;
  thrust::device_vector<std::uint64_t> is_dest_;
  std::array<thrust::device_vector<std::uint64_t>, kMaxVias> is_via_;
  thrust::device_vector<via_stop> via_stops_;
  thrust::device_vector<std::uint16_t> dist_to_dest_;

  // pinned staging + device buffers for fast per-query H2D
  pinned_buffer<std::uint16_t> dist_to_dest_pinned_;
  pinned_buffer<std::pair<location_idx_t, unixtime_t>> starts_pinned_;
  thrust::device_vector<std::pair<location_idx_t, unixtime_t>> starts_dev_;

  // reusable reconstruction buffers (avoid per-query cudaMalloc)
  thrust::device_vector<location_idx_t> rec_dest_;
  thrust::device_vector<gpu_journey> rec_out_;
  thrust::host_vector<gpu_journey> rec_host_out_;

  device_timetable tt_;

  thrust::host_vector<std::uint64_t> host_round_times_;
  raptor_state host_state_;

  // compute_et boarding pre-pass: precomputed earliest-transport per flat
  // (route,stop), and a compacted task list of boarding candidates per round.
  thrust::device_vector<std::uint64_t> et_result_;
  thrust::device_vector<std::uint32_t> et_task_list_;
  thrust::device_vector<std::uint32_t> et_task_count_;

  cudaStream_t stream_;
};

gpu_raptor_state::gpu_raptor_state(gpu_timetable const& gtt)
    : impl_{std::make_unique<impl>(gtt)} {}

gpu_raptor_state::~gpu_raptor_state() = default;

// selective reset: scan the dirty bitfield and set only the marked round_times
// entries back to invalid (zeroing the bitfield as we go), instead of memset-ing
// the whole buffer. The search touches a tiny fraction of entries, so the scan
// (size/8 bytes) + a handful of writes beats the full clear (size*8 bytes).
__global__ void clear_dirty_kernel(std::uint64_t* const data,
                                   std::uint32_t* const bits,
                                   std::uint32_t const n_words,
                                   std::uint64_t const invalid) {
  for (auto w = blockIdx.x * blockDim.x + threadIdx.x; w < n_words;
       w += gridDim.x * blockDim.x) {
    auto word = bits[w];
    if (word == 0U) {
      continue;
    }
    bits[w] = 0U;
    auto const base = w * 32U;
    while (word != 0U) {
      auto const b = __ffs(static_cast<int>(word)) - 1;
      data[base + static_cast<unsigned>(b)] = invalid;
      word &= word - 1U;  // clear lowest set bit
    }
  }
}

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
    bool const require_car_transport,
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
      via_stops_{via_stops},
      base_{base},
      allowed_claszes_{allowed_claszes},
      require_bike_transport_{require_bike_transport},
      require_car_transport_{require_car_transport},
      is_wheelchair_{is_wheelchair},
      transfer_time_settings_{tts} {
  state_.impl_->resize(tt.n_locations(), tt.n_routes(),
                       rtt ? rtt->n_rt_transports() : 0U, is_via, via_stops,
                       is_dest, dist_to_dest);
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
__global__ void reconstruct_kernel(location_idx_t const* const dest_list,
                                   std::uint32_t const n_dest,
                                   std::uint32_t const end_k,
                                   raptor_impl<SearchDir, Rt, Vias> r,
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

template <direction SearchDir, bool Rt, via_offset_t Vias>
void gpu_raptor<SearchDir, Rt, Vias>::execute(unixtime_t start_time,
                                              std::uint8_t max_transfers,
                                              unixtime_t worst_time_at_dest,
                                              profile_idx_t,
                                              pareto_set<journey>& results) {
  auto& s = *state_.impl_;
  // Re-upload this search's params: ping and pong share one gpu_raptor_state,
  // and the pong instance's ctor would otherwise leave its destination (= the
  // original start) in the shared device is_dest_, killing ping's propagation.
  s.upload_query(is_via_, via_stops_, is_dest_, dist_to_end_);
  auto* const starts_pinned = s.starts_pinned_.ensure(starts_.size());
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

  auto r = raptor_impl<SearchDir, Rt, Vias>{
      .any_marked_ = thrust::raw_pointer_cast(s.any_marked_.data()),
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
      .starts_ = starts,
      .is_via_ =
          [&]() {
            auto ret = cuda::std::array<device_bitvec<std::uint64_t const>,
                                        kMaxVias>{};
            for (auto i = 0U; i != kMaxVias; ++i) {
              ret[i] = {to_view(s.is_via_[i])};
            }
            return ret;
          }(),
      .via_stops_ = to_view(s.via_stops_),
      .is_dest_ = {to_view(s.is_dest_)},
      .end_reachable_ = {to_mutable_view(s.end_reachable_)},
      .dist_to_end_ = to_view(s.dist_to_dest_),
      .round_times_ = {to_mutable_view(s.round_times_), n_locations_,
                       thrust::raw_pointer_cast(s.dirty_bits_.data())},
      .best_ = {to_mutable_view(s.best_), n_locations_,
                thrust::raw_pointer_cast(s.best_dirty_bits_.data())},
      .tmp_ = {to_mutable_view(s.tmp_), n_locations_,
               thrust::raw_pointer_cast(s.tmp_dirty_bits_.data())},
      .time_at_dest_ = {to_mutable_view(s.time_at_dest_), n_locations_},
      .station_mark_ = {to_mutable_view(s.station_mark_)},
      .prev_station_mark_ = {to_mutable_view(s.prev_station_mark_)},
      .route_mark_ = {to_mutable_view(s.route_mark_)},
      .et_result_ = to_mutable_view(s.et_result_),
      .et_task_list_ = to_mutable_view(s.et_task_list_),
      .et_task_count_ = thrust::raw_pointer_cast(s.et_task_count_.data())};

  auto blocks = 0;
  auto threads = 0;
  auto const kernel = reinterpret_cast<void*>(exec_raptor<SearchDir, Rt, Vias>);
  cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, kernel, 0, 0);
  {
    //    auto t = utl::scoped_timer{"kernel launch"};
    void* args[] = {reinterpret_cast<void*>(&start_time),
                    reinterpret_cast<void*>(&max_transfers),
                    reinterpret_cast<void*>(&worst_time_at_dest),
                    reinterpret_cast<void*>(&r)};
    cudaLaunchCooperativeKernel(kernel, blocks, threads, args, 0, s.stream_);

    cudaStreamSynchronize(s.stream_);
  }
  CUDA_CHECK(cudaPeekAtLastError());

  // --- GPU reconstruction: build journeys (with legs) on the device, then
  // materialize them on the host. No full round-time transfer. ---
  auto const end_k =
      static_cast<std::uint32_t>(std::min(max_transfers, kMaxTransfers) + 2U);

  auto dest_list = std::vector<location_idx_t>{};
  if (s.is_intermodal_dest_) {
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
  // reuse preallocated buffers (resize only grows; no per-query cudaMalloc once
  // warmed up)
  s.rec_dest_ = dest_list;  // small host->device copy
  if (s.rec_out_.size() < total) {
    s.rec_out_.resize(total);
  }
  {
    auto const threads = 128U;
    auto const blocks = (total + threads - 1U) / threads;
    reconstruct_kernel<SearchDir, Rt, Vias><<<blocks, threads, 0, s.stream_>>>(
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
    if (gj.valid_ == 0U) {
      continue;
    }
    auto j = journey{};
    j.start_time_ = start_time;
    j.dest_time_ = delta_to_unix(base(), gj.dest_time_);
    j.dest_ = location_idx_t{gj.dest_l_};
    j.transfers_ = gj.transfers_;
    // POD legs are emitted in walk order: reverse-chronological for forward
    // (walk dest->start), chronological for backward (walk start->dest). Emit
    // the journey in chronological order accordingly.
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
        j.legs_.emplace_back(journey::leg{SearchDir, from, to, dep, arr,
                                          footpath{to, duration_t{static_cast<
                                              duration_t::rep>(
                                              gl.fp_duration_)}}});
      } else {
        auto const t_idx = transport_idx_t{gl.transport_};
        auto const route = tt_.transport_route_[t_idx];
        auto const route_len =
            static_cast<stop_idx_t>(tt_.route_location_seq_[route].size());
        auto const run =
            rt::run{.t_ = transport{t_idx, day_idx_t{gl.day_}},
                    .stop_range_ = interval<stop_idx_t>{stop_idx_t{0},
                                                        route_len}};
        j.legs_.emplace_back(journey::leg{
            SearchDir, from, to, dep, arr,
            journey::run_enter_exit{run, gl.enter_stop_, gl.exit_stop_}});
      }
    }
    // Re-anchor transfer/footpath legs to the preceding leg's arrival (CPU
    // convention). The footpath duration is correct in both directions, but for
    // backward search round_times holds latest-departure times, so the device
    // anchors the transfer to the next departure instead of the previous
    // arrival. Snapping dep to the preceding leg's arrival fixes the instant
    // (no-op for forward, where they already coincide).
    for (auto i = std::size_t{1U}; i < j.legs_.size(); ++i) {
      if (std::holds_alternative<footpath>(j.legs_[i].uses_)) {
        auto const dur = std::get<footpath>(j.legs_[i].uses_).duration();
        j.legs_[i].dep_time_ = j.legs_[i - 1U].arr_time_;
        j.legs_[i].arr_time_ = j.legs_[i].dep_time_ + dur;
      }
    }
    // Add the fully-reconstructed journey. is_reconstructed_ stays false here so
    // it still goes through search.h's reconstruct-loop filtering (interval /
    // travel-time checks); reconstruct(q, j) then marks it. The pareto_set
    // dedups via journey::dominates (start_time, dest_time, transfers).
    results.add(std::move(j));
  }
}

template <direction SearchDir, bool Rt, via_offset_t Vias>
void gpu_raptor<SearchDir, Rt, Vias>::sync_round_times() {
  auto& s = *state_.impl_;
  utl::verify(
      cudaSuccess ==
          cudaMemcpy(thrust::raw_pointer_cast(s.host_round_times_.data()),
                     thrust::raw_pointer_cast(s.round_times_.data()),
                     s.round_times_.size() * sizeof(std::uint64_t),
                     cudaMemcpyDeviceToHost),
      "could not sync round times");
  s.host_state_.round_times_storage_.resize(s.host_round_times_.size());
  // unpack the time component from the packed time||breadcrumb words
  for (auto i = std::size_t{0U}; i != s.host_round_times_.size(); ++i) {
    s.host_state_.round_times_storage_[i] =
        device_times<SearchDir, Vias + 1U>::from_key(
            static_cast<std::uint16_t>(s.host_round_times_[i] >> 48U));
  }
}

template <direction SearchDir, bool Rt, via_offset_t Vias>
void gpu_raptor<SearchDir, Rt, Vias>::reset_arrivals() {
  auto& s = *state_.impl_;
  cudaMemsetAsync(thrust::raw_pointer_cast(s.time_at_dest_.data()), 0xFF,
                  s.time_at_dest_.size() * sizeof(std::uint64_t), s.stream_);
  // Full reset (kInvalidPacked == all-ones == 0xFF bytes). The selective
  // dirty-clear optimization was incorrect on state reuse (it left stale
  // arrivals that pruned later searches -> degraded/empty results); correctness
  // first. Re-introducing a correct selective clear is a throughput TODO.
  cudaMemsetAsync(thrust::raw_pointer_cast(s.round_times_.data()), 0xFF,
                  s.round_times_.size() * sizeof(std::uint64_t), s.stream_);
}

template <direction SearchDir, bool Rt, via_offset_t Vias>
void gpu_raptor<SearchDir, Rt, Vias>::next_start_time() {
  starts_.clear();
  auto& s = *state_.impl_;
  // Full reset (see reset_arrivals): the selective dirty-clear was incorrect on
  // state reuse. Correctness first; selective clear is a throughput TODO.
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
  if constexpr (Rt) {
    thrust::fill(thrust::cuda::par.on(state_.impl_->stream_),
                 begin(state_.impl_->rt_transport_mark_),
                 end(state_.impl_->rt_transport_mark_), 0U);
  }
}

template <direction SearchDir, bool Rt, via_offset_t Vias>
void gpu_raptor<SearchDir, Rt, Vias>::reconstruct(query const&, journey& j) {
  // The legs were already filled on the GPU inside execute(); just mark the
  // journey reconstructed so search.h keeps it (and runs its filtering).
  j.is_reconstructed_ = true;
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