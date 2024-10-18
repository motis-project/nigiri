#pragma once

#include <iostream>
#include <cinttypes>
#include "nigiri/common/linear_lower_bound.h"
#include "nigiri/routing/gpu_raptor_state.h"
#include "nigiri/routing/gpu_timetable.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/raptor/debug.h"
#include "utl/helpers/algorithm.h"
#include <variant>
extern "C"{
void copy_to_devices(gpu_clasz_mask_t const& allowed_claszes,
                     std::vector<std::uint16_t> const& dist_to_dest,
                     gpu_day_idx_t const& base,
                     std::vector<uint8_t> const& is_dest,
                     std::vector<std::uint16_t> const& lb,
                     int const& n_days,
                     std::uint16_t const& kUnreachable,
                     short const& kMaxTravelTimeTicks,
                     unsigned int const& kIntermodalTarget,
                     gpu_clasz_mask_t*& allowed_claszes_ptr,
                     std::uint16_t* & dist_to_end_ptr,
                     std::uint32_t* & dist_to_end_size_ptr,
                     gpu_day_idx_t* & base_ptr,
                     bool* & is_dest_ptr,
                     std::uint16_t* & lb_ptr,
                     int* & n_days_ptr,
                     std::uint16_t* & kUnreachable_ptr,
                     gpu_location_idx_t* & kIntermodalTarget_ptr,
                     short* & kMaxTravelTimeTicks_ptr);

void copy_to_device_destroy(
    gpu_clasz_mask_t*& allowed_claszes,
    std::uint16_t* & dist_to_end,
    std::uint32_t* & dist_to_end_size,
    gpu_day_idx_t* & base,
    bool* & is_dest,
    std::uint16_t* & lb,
    int* & n_days,
    std::uint16_t* & kUnreachable,
    gpu_location_idx_t* & kIntermodalTarget,
    short* & kMaxTravelTimeTicks);
}
void launch_kernel(void** args,
                          device_context const& device,
                          cudaStream_t s,
                          gpu_direction search_dir,
                          bool rt);

void* get_gpu_raptor_kernel(gpu_direction search_dir,bool rt);

void copy_to_gpu_args(gpu_unixtime_t const* start_time,
                      gpu_unixtime_t const* worst_time_at_dest,
                      gpu_profile_idx_t const* prf_idx,
                      gpu_unixtime_t*& start_time_ptr,
                      gpu_unixtime_t*& worst_time_at_dest_ptr,
                      gpu_profile_idx_t*& prf_idx_ptr);
void destroy_copy_to_gpu_args(gpu_unixtime_t* start_time_ptr,
                              gpu_unixtime_t* worst_time_at_dest_ptr,
                              gpu_profile_idx_t* prf_idx_ptr);
template<gpu_direction SearchDir>
__host__ __device__ static bool is_better(auto a, auto b) { return SearchDir==gpu_direction::kForward ? a < b : a > b; }
__host__ __device__ static auto get_smaller(auto a, auto b) { return a < b ? a : b ;}
template<gpu_direction SearchDir>
__host__ __device__ static bool is_better_or_eq(auto a, auto b) { return SearchDir==gpu_direction::kForward ? a <= b : a >= b; }

template<gpu_direction SearchDir>
__host__ __device__ static auto get_best(auto a, auto b) { return is_better<SearchDir>(a, b) ? a : b; }

template<gpu_direction SearchDir>
__host__ __device__ static auto get_best(auto x, auto... y) {
  ((x = get_best<SearchDir>(x, y)), ...);
  return x;
}

__host__ __device__ inline int as_int(gpu_location_idx_t d) { return static_cast<int>(d.v_); }
__host__ __device__ inline int as_int(gpu_day_idx_t d)  { return static_cast<int>(d.v_); }

__device__ inline gpu_sys_days base_of(gpu_day_idx_t* base,gpu_interval<gpu_sys_days> const* date_range_ptr) {
  return gpu_internal_interval_days(date_range_ptr).from_ + as_int(*base) * gpu_days{1};
}
__host__ inline gpu_sys_days cpu_base(gpu_timetable const* gtt, gpu_day_idx_t base) {
  return gtt->cpu_internal_interval_days().from_ + as_int(base) * gpu_days{1};
}
template<gpu_direction SearchDir>
__host__ __device__ static auto dir(auto a) { return (SearchDir==gpu_direction::kForward ? 1 : -1) * a; }

__host__ __device__ inline gpu_delta_t to_gpu_delta(gpu_day_idx_t const day, std::int16_t const mam,gpu_day_idx_t* base_) {
  return gpu_clamp((as_int(day) - as_int(*base_)) * 1440 + mam);
}

template <gpu_direction SearchDir, typename T>
__host__ __device__ auto gpu_get_begin_it(T const& t) {
  if constexpr (SearchDir == gpu_direction::kForward) {
    return t.begin();
  } else {
    return t.rbegin();
  }
}

template <gpu_direction SearchDir, typename T>
__host__ __device__ auto gpu_get_end_it(T const& t) {
  if constexpr ((SearchDir == gpu_direction::kForward)) {
    return t.end();
  } else {
    return t.rend();
  }
}

template <gpu_direction SearchDir, bool Rt>
struct gpu_raptor {
  using algo_stats_t = gpu_raptor_stats;
  static constexpr auto const kMaxTravelTimeTicks = gpu_kMaxTravelTime.count();
  static constexpr bool kUseLowerBounds = true;
  static constexpr auto const kFwd = (SearchDir == gpu_direction::kForward);
  static constexpr auto const kBwd = (SearchDir == gpu_direction::kBackward);
  static constexpr auto const kInvalid = kInvalidGpuDelta<SearchDir>;
  static constexpr auto const kUnreachable =
      std::numeric_limits<std::uint16_t>::max();

  gpu_raptor(gpu_timetable const* gtt,
             mem& mem,
         std::vector<uint8_t>& is_dest,
         std::vector<std::uint16_t>& dist_to_dest,
         std::vector<std::uint16_t>& lb,
         gpu_day_idx_t const& base,
         gpu_clasz_mask_t const& allowed_claszes,
             int const& n_days)
      : gtt_{gtt},
        mem_{mem},
        cpu_base_{base}
        {
    auto const kIntermodalTarget  =
        gpu_to_idx(get_gpu_special_station(gpu_special_station::kEnd));
    copy_to_devices(allowed_claszes,
                    dist_to_dest,
                    base,
                    is_dest,
                    lb,
                    n_days,
                    kUnreachable,
                    kMaxTravelTimeTicks,
                    kIntermodalTarget,
                    allowed_claszes_,
                    dist_to_end_,
                    dist_to_end_size_,
                    base_,
                    is_dest_,
                    lb_,
                    n_days_,
                    kUnreachable_,
                    kIntermodalTarget_,
                    kMaxTravelTimeTicks_);
  }
  ~gpu_raptor(){
      copy_to_device_destroy(allowed_claszes_,
                           dist_to_end_,
                           dist_to_end_size_,
                           base_,
                           is_dest_,
                           lb_,
                           n_days_,
                           kUnreachable_,
                           kIntermodalTarget_,
                           kMaxTravelTimeTicks_);
  }
  algo_stats_t get_stats() const {
    return stats_;
  }

  void reset_arrivals() {
    utl::fill(mem_.host_.round_times_,kInvalid);
    mem_.reset_arrivals_async();
  }

  void next_start_time() {
    utl::fill(mem_.host_.best_, kInvalid);
    utl::fill(mem_.host_.station_mark_, 0);
    mem_.next_start_time_async();
  }

  void add_start(gpu_location_idx_t const l, gpu_unixtime_t const t) {
    mem_.host_.best_[gpu_to_idx(l)] = unix_to_gpu_delta(cpu_base(gtt_, cpu_base_), t);
    mem_.host_.round_times_[0U * mem_.device_.column_count_round_times_ + gpu_to_idx(l)] = unix_to_gpu_delta(cpu_base(gtt_, cpu_base_), t);
    unsigned int const store_idx = (gpu_to_idx(l) >> 5);
    unsigned int const mask = 1 << (gpu_to_idx(l) % 32);
    mem_.host_.station_mark_[store_idx] |= mask;
    mem_.host_.synced = false;
  }

  void execute(gpu_unixtime_t const& start_time,
             uint8_t const& max_transfers,
             gpu_unixtime_t const& worst_time_at_dest,
             gpu_profile_idx_t const& prf_idx){
    if (!mem_.host_.synced){
      mem_.copy_host_to_device();
    }
    gpu_unixtime_t* start_time_ptr = nullptr;
    gpu_unixtime_t* worst_time_at_dest_ptr = nullptr;
    gpu_profile_idx_t* prf_idx_ptr = nullptr;
    copy_to_gpu_args(&start_time,
                     &worst_time_at_dest,
                     &prf_idx,
                     start_time_ptr,
                     worst_time_at_dest_ptr,
                     prf_idx_ptr);
    void* kernel_args[] = {(void*)&start_time_ptr,
                           (void*)&max_transfers,
                           (void*)&worst_time_at_dest_ptr,
                           (void*)&prf_idx_ptr,
                           (void*)&allowed_claszes_,
                           (void*)&dist_to_end_,
                           (void*)&dist_to_end_size_,
                           (void*)&base_,
                           (void*)&is_dest_,
                           (void*)&lb_,
                           (void*)&n_days_,
                           (void*)&kUnreachable_,
                           (void*)&kIntermodalTarget_,
                           (void*)&kMaxTravelTimeTicks_,
                           (void*)&(mem_.device_),
                           (void*)gtt_};
    launch_kernel(kernel_args, mem_.context_, mem_.context_.proc_stream_,SearchDir,Rt);
    mem_.copy_device_to_host();
    stats_ = mem_.host_.stats_[0];
    destroy_copy_to_gpu_args(start_time_ptr,worst_time_at_dest_ptr,prf_idx_ptr);
  }
  gpu_timetable const* gtt_{nullptr};
  mem& mem_;
  bool* is_dest_{nullptr};
  uint16_t* dist_to_end_{nullptr};
  uint32_t* dist_to_end_size_{nullptr};
  uint16_t* lb_{nullptr};
  gpu_day_idx_t* base_{nullptr};
  gpu_day_idx_t cpu_base_;
  int* n_days_{nullptr};
  gpu_raptor_stats stats_;
  gpu_clasz_mask_t* allowed_claszes_{nullptr};
  std::uint16_t* kUnreachable_{nullptr};
  gpu_location_idx_t* kIntermodalTarget_{nullptr};
  short* kMaxTravelTimeTicks_{nullptr};
};
