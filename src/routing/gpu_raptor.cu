#pragma once
#include "nigiri/routing/gpu_raptor.h"
#include <iostream>

#include <cooperative_groups.h>
using namespace cooperative_groups;

__device__ __forceinline__ unsigned int get_block_thread_id() {
  return threadIdx.x + (blockDim.x * threadIdx.y);
}

__device__ __forceinline__ unsigned int get_global_thread_id() {
  return get_block_thread_id() + (blockDim.x * blockDim.y * blockIdx.x);
}

__device__ __forceinline__ unsigned int get_block_stride() {
  return blockDim.x * blockDim.y;
}

__device__ __forceinline__ unsigned int get_global_stride() {
  return get_block_stride() * gridDim.x * gridDim.y;
}

__device__ void mark(unsigned int* store, unsigned int const idx) {
  unsigned int const store_idx = (idx >> 5);
  unsigned int const mask = 1 << (idx % 32);
  atomicOr(&store[store_idx], mask);
}

__device__ bool marked(unsigned int const* const store, unsigned int idx) {
  unsigned int const store_idx = (idx >> 5);
  unsigned int const val = store[store_idx];
  unsigned int const mask = 1 << (idx % 32);
  return (bool)(val & mask);
}

__device__ void reset_store(unsigned int* store, int const store_size) {
  auto const t_id = get_global_thread_id();
  auto const stride = get_global_stride();

  for (auto idx = t_id; idx < store_size; idx += stride) {
    store[idx] = 0x000;
  }
}
__device__ void swap_b_reset(unsigned int* store_a, unsigned int* store_b, int const store_size) {
  auto const t_id = get_global_thread_id();
  auto const stride = get_global_stride();
  for (auto idx = t_id; idx < store_size; idx += stride) {
    store_a[idx] = store_b[idx];
    store_b[idx] = 0;
  }
}

template <gpu_direction SearchDir>
__device__ bool update_arrival(gpu_delta_t* base,
                               const unsigned int l_idx, gpu_delta_t const val){
  gpu_delta_t* const arr_address = &base[l_idx];
  auto* base_address = (int*)((size_t)arr_address & ~2);
  int old_value, new_value;

  do {
    old_value = atomicCAS(base_address, *base_address, *base_address);

    if ((size_t)arr_address & 2) {
      int old_upper = (old_value >> 16) & 0xFFFF;
      old_upper = (old_upper << 16) >> 16;
      int new_upper = get_best<SearchDir>(old_upper, val);
      if (new_upper == old_upper) {
        return false;
      }
      new_value = (old_value & 0x0000FFFF) | (new_upper << 16);
    } else {
      int old_lower = old_value & 0xFFFF;
      old_lower = (old_lower << 16) >> 16;
      int new_lower = get_best<SearchDir>(old_lower, val);
      if (new_lower == old_lower){
        return false;
      }
      new_value = (old_value & 0xFFFF0000) | (new_lower & 0xFFFF);
    }
  } while (atomicCAS(base_address, old_value, new_value) != old_value);

  return true;
}

template <gpu_direction SearchDir, bool Rt>
__device__ void update_time_at_dest(unsigned const k, gpu_delta_t const t, gpu_delta_t * time_at_dest){
  for (auto i = k; i < gpu_kMaxTransfers+1; ++i) {
    update_arrival<SearchDir>(time_at_dest,i,t);
  }
}

template <gpu_direction SearchDir, bool Rt>
__device__ void convert_station_to_route_marks(unsigned int* station_marks,
                                               unsigned int* route_marks,
                                               int* any_station_marked,
                                               gpu_vecvec<gpu_location_idx_t ,
                                               gpu_route_idx_t> const* location_routes,
                                               std::uint32_t const n_locations) {
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();
  for (uint32_t idx = global_t_id; idx < n_locations; idx += global_stride) {
    if (marked(station_marks, idx)) {
      if (!*any_station_marked) {
        atomicOr(reinterpret_cast<int*>(any_station_marked),1);
      }
      for (auto r : (*location_routes)[gpu_location_idx_t{idx}]) {

        mark(route_marks, gpu_to_idx(r));
      }
    }
  }
}

template <gpu_direction SearchDir, bool Rt>
__device__ gpu_delta_t time_at_stop(gpu_route_idx_t const r, gpu_transport const t,
                                    gpu_stop_idx_t const stop_idx,
                                    gpu_event_type const ev_type,
                                    gpu_day_idx_t base,
                                    gpu_vector_map<gpu_route_idx_t,gpu_interval<std::uint32_t>> const* route_stop_time_ranges,
                                    gpu_vector_map<gpu_route_idx_t,gpu_interval<gpu_transport_idx_t >> const* route_transport_ranges,
                                    gpu_delta const* route_stop_times){
  auto const n_transports = static_cast<unsigned>((*route_transport_ranges)[r].size());
  auto const route_stop_begin = static_cast<unsigned>((*route_stop_time_ranges)[r].from_ + n_transports *
                                                      (stop_idx * 2 - (ev_type==gpu_event_type::kArr ? 1 : 0)));
  return gpu_clamp((as_int(t.day_) - as_int(base)) * 1440
                   + route_stop_times[route_stop_begin +
                   (gpu_to_idx(t.t_idx_) - gpu_to_idx((*route_transport_ranges)[r].from_))].count());
}

template <typename It, typename End, typename Key, typename Cmp>
__device__ It linear_lb(It from, End to, Key&& key, Cmp&& cmp) {
  for (auto it = from; it != to; ++it) {
    if (!cmp(*it, key)) {
      return it;
    }
  }
  return to;
}

template <gpu_direction SearchDir, bool Rt>
__device__ bool is_transport_active(gpu_transport_idx_t const t,
                                    std::size_t const day,
                                    gpu_vector_map<gpu_transport_idx_t,gpu_bitfield_idx_t> const* transport_traffic_days,
                                    gpu_vector_map<gpu_bitfield_idx_t, gpu_bitfield> const* bitfields)  {

  if (day >= (*bitfields)[(*transport_traffic_days)[t]].size()) {
    return false;
  }
  auto const block = (*bitfields)[(*transport_traffic_days)[t]].blocks_
                         [day / (*bitfields)[(*transport_traffic_days)[t]].bits_per_block];
  auto const bit = (day % (*bitfields)[(*transport_traffic_days)[t]].bits_per_block);
  return (block & (std::uint64_t{1U} << bit)) != 0U;
}

template <gpu_direction SearchDir>
__device__ bool valid(gpu_delta_t t) {
  if constexpr (SearchDir == gpu_direction::kForward) {
    return t != cuda::std::numeric_limits<gpu_delta_t>::max();
  } else {
    return t != cuda::std::numeric_limits<gpu_delta_t>::min();
  }
}

template <gpu_direction SearchDir, bool Rt>
__device__ gpu_transport get_earliest_transport(unsigned const k,
                                                gpu_route_idx_t const r,
                                                gpu_stop_idx_t const stop_idx,
                                                gpu_day_idx_t const day_at_stop,
                                                gpu_minutes_after_midnight_t const mam_at_stop,
                                                gpu_location_idx_t const l,
                                                gpu_raptor_stats* stats,
                                                uint16_t* lb,
                                                gpu_delta_t* time_at_dest,
                                                int n_days_, gpu_day_idx_t* base,gpu_timetable const& gtt) {
  atomicAdd(&stats[0].n_earliest_trip_calls_, 1);

  auto const n_days_to_iterate = get_smaller(
      gpu_kMaxTravelTime.count() / 1440 + 1,
      (SearchDir == gpu_direction::kForward) ? n_days_ - as_int(day_at_stop)
                                             : as_int(day_at_stop) + 1);

  auto const event_times = gpu_event_times_at_stop(
      r, stop_idx, (SearchDir == gpu_direction::kForward) ? gpu_event_type::kDep : gpu_event_type::kArr,
      gtt.route_stop_time_ranges_,gtt.route_transport_ranges_, gtt.route_stop_times_);


  auto const seek_first_day = [&]() {
    return linear_lb(gpu_get_begin_it<SearchDir>(event_times), gpu_get_end_it<SearchDir>(event_times),
                     mam_at_stop,
                     [&](gpu_delta const a, gpu_minutes_after_midnight_t const b) {
                       return is_better<SearchDir>(a.mam_, b.count());
                     });
  };

  for (auto i = gpu_day_idx_t::value_t{0U}; i != n_days_to_iterate; ++i) {
    auto const ev_time_range =
        gpu_it_range{i == 0U ? seek_first_day() : gpu_get_begin_it<SearchDir>(event_times),
                 gpu_get_end_it<SearchDir>(event_times)};

    if (ev_time_range.empty()) {
      continue;
    }

    auto const day = (SearchDir == gpu_direction::kForward) ?
                                          day_at_stop + i : day_at_stop - i;
    for (auto it = begin(ev_time_range); it != end(ev_time_range); ++it) {
      auto const t_offset =
          static_cast<std::size_t>(&*it - event_times.data());
      auto const ev = *it;
      auto const ev_mam = ev.mam_;

      if (is_better_or_eq<SearchDir>(time_at_dest[k],
                          to_gpu_delta(day, ev_mam, base) + dir<SearchDir>(lb[gpu_to_idx(l)]))) {
        return {gpu_transport_idx_t::invalid(), gpu_day_idx_t::invalid()};
      }
      auto const t = (*gtt.route_transport_ranges_)[r][t_offset];
      if (i == 0U && !is_better_or_eq<SearchDir>(mam_at_stop.count(), ev_mam)) {
        continue;
      }

      auto const ev_day_offset = ev.days_;
      auto const start_day =
          static_cast<std::size_t>(as_int(day) - ev_day_offset);

      if(!is_transport_active<SearchDir, Rt>(t, start_day, gtt.transport_traffic_days_, gtt.bitfields_)) {
        continue;
      }
      return {t, static_cast<gpu_day_idx_t>(as_int(day) - ev_day_offset)};
    }
  }
  return {};
}

template <gpu_direction SearchDir, bool Rt>
__device__ void update_route(unsigned const k, gpu_route_idx_t const r,
                             uint16_t* lb, unsigned short kUnreachable,
                             gpu_day_idx_t* base, int n_days,
                             gpu_timetable const& gtt, device_memory const& device_mem) {
  auto const stop_seq = (*gtt.route_location_seq_)[r];
  auto et = gpu_transport{};
  for (auto i = 0U; i != stop_seq.size(); ++i) {
    auto const stop_idx =
        static_cast<gpu_stop_idx_t>((SearchDir == gpu_direction::kForward) ? i : stop_seq.size() - i - 1U);
    auto const stp = gpu_stop{stop_seq[stop_idx]};
    auto const l_idx = gpu_to_idx(stp.gpu_location_idx());
    auto const is_last = i == stop_seq.size() - 1U;

    if (!et.is_valid() && !marked(device_mem.prev_station_mark_, l_idx)) {
      continue;
    }
    auto current_best = kInvalidGpuDelta<SearchDir>;

    if (et.is_valid() && ((SearchDir == gpu_direction::kForward) ? stp.out_allowed() : stp.in_allowed())) {
      auto const by_transport = time_at_stop<SearchDir, Rt>(r, et, stop_idx,
          (SearchDir == gpu_direction::kForward) ? gpu_event_type::kArr : gpu_event_type::kDep,
          *base, gtt.route_stop_time_ranges_, gtt.route_transport_ranges_, gtt.route_stop_times_);
      current_best = get_best<SearchDir>(device_mem.round_times_[(k - 1)*device_mem.column_count_round_times_ + l_idx],
                                         device_mem.tmp_[l_idx], device_mem.best_[l_idx]);

      if (is_better<SearchDir>(by_transport, current_best) &&
          is_better<SearchDir>(by_transport, device_mem.time_at_dest_[k]) &&
          lb[l_idx] != kUnreachable &&
          is_better<SearchDir>(by_transport + dir<SearchDir>(lb[l_idx]), device_mem.time_at_dest_[k])) {
        auto updated = update_arrival<SearchDir>(device_mem.tmp_,l_idx,get_best<SearchDir>(by_transport, device_mem.tmp_[l_idx]));
        if (updated){
          atomicAdd(&device_mem.stats_[0].n_earliest_arrival_updated_by_route_, 1);
          mark(device_mem.station_mark_, l_idx);
          current_best = by_transport;
          atomicOr(device_mem.any_station_marked_,1);
        }
      }
    }

    if (is_last || !((SearchDir == gpu_direction::kForward) ? stp.in_allowed() : stp.out_allowed()) ||
        !marked(device_mem.prev_station_mark_, l_idx)) {
      continue;
    }

    if (lb[l_idx] == kUnreachable) {
      break;
    }

    auto const et_time_at_stop = et.is_valid() ?
               time_at_stop<SearchDir, Rt>(r, et, stop_idx,
               (SearchDir == gpu_direction::kForward) ? gpu_event_type::kDep : gpu_event_type::kArr,
               *base, gtt.route_stop_time_ranges_, gtt.route_transport_ranges_, gtt.route_stop_times_)
            : kInvalidGpuDelta<SearchDir>;
    auto const prev_round_time = device_mem.round_times_[(k-1) * device_mem.column_count_round_times_ + l_idx];

    if (is_better_or_eq<SearchDir>(prev_round_time, et_time_at_stop)) {

      auto const [day, mam] =
          gpu_split_day_mam(*base, prev_round_time);

      auto const new_et = get_earliest_transport<SearchDir, Rt>(k, r, stop_idx,
                          day, mam, stp.gpu_location_idx(), device_mem.stats_, lb, device_mem.time_at_dest_,
                                                                n_days, base, gtt);
      current_best =
          get_best<SearchDir>(current_best, device_mem.best_[l_idx], device_mem.tmp_[l_idx]);
      if (new_et.is_valid() &&
          (current_best == kInvalidGpuDelta<SearchDir> ||
           is_better_or_eq<SearchDir>(
               time_at_stop<SearchDir, Rt>(r, new_et, stop_idx,
                            (SearchDir == gpu_direction::kForward) ? gpu_event_type::kDep : gpu_event_type::kArr,
                            *base, gtt.route_stop_time_ranges_, gtt.route_transport_ranges_, gtt.route_stop_times_), et_time_at_stop))) {
        et = new_et;
      }
    }
  }
}

template <gpu_direction SearchDir, bool Rt, bool WithClaszFilter>
__device__ void loop_routes(unsigned const k,gpu_clasz_mask_t const* allowed_claszes,
                            short const* kMaxTravelTimeTicks, uint16_t* lb,
                            gpu_day_idx_t* base, unsigned short kUnreachable,int n_days,
                            gpu_timetable const& gtt,device_memory const& device_mem){
  if(get_global_thread_id() == 0){
    atomicAnd(device_mem.any_station_marked_,0);
  }
  this_grid().sync();
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();
  for(auto r_idx = global_t_id;
       r_idx < gtt.n_routes_; r_idx += global_stride){
    auto const r = gpu_route_idx_t{r_idx};
    if(!marked(device_mem.route_mark_, r_idx)) {
      continue;
    }
    if constexpr (WithClaszFilter){
      auto const as_mask = static_cast<gpu_clasz_mask_t>
          (1U << static_cast<std::underlying_type_t<gpu_clasz>>((*gtt.route_clasz_)[r]));
      if(!((*allowed_claszes & as_mask)==as_mask)){
        continue;
      }
    }
    atomicAdd(&device_mem.stats_[0].n_routes_visited_, 1);
    update_route<SearchDir, Rt>(k, r, lb,  kUnreachable,
                                base, n_days, gtt, device_mem);
  }
}

template <gpu_direction SearchDir, bool Rt>
__device__ void update_transfers(unsigned const k, bool const * is_dest,
                                 uint16_t* dist_to_end, uint32_t dist_to_end_size,
                                 unsigned short kUnreachable, uint16_t* lb,
                                 std::uint32_t const n_locations,
                                 gpu_locations const& locations,device_memory const& device_mem){
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();
  for(auto l_idx = global_t_id;
       l_idx < n_locations; l_idx += global_stride){
    if(!marked(device_mem.prev_station_mark_, l_idx)){
      continue;
    }
    auto const dest = is_dest[l_idx];
    auto const tt = (dist_to_end_size==0 && dest)
        ? 0 : dir<SearchDir>((*locations.transfer_time_)[gpu_location_idx_t{l_idx}]).count();
    const auto fp_target_time =
        static_cast<gpu_delta_t>(device_mem.tmp_[l_idx] + tt);
    if(is_better<SearchDir>(fp_target_time, device_mem.best_[l_idx])
        && is_better<SearchDir>(fp_target_time, device_mem.time_at_dest_[k])){
      if(lb[l_idx] == kUnreachable
          || !is_better<SearchDir>(fp_target_time + dir<SearchDir>(lb[l_idx]), device_mem.time_at_dest_[k])){
        atomicAdd(&device_mem.stats_[0].fp_update_prevented_by_lower_bound_, 1);
        continue;
      }
      bool updated = update_arrival<SearchDir>(device_mem.best_, l_idx, fp_target_time);
      if(updated){
        update_arrival<SearchDir>(device_mem.round_times_, k * device_mem.column_count_round_times_ + l_idx, fp_target_time);
        atomicAdd(&device_mem.stats_[0].n_earliest_arrival_updated_by_footpath_, 1);
        mark(device_mem.station_mark_, l_idx);
        if(dest){
          update_time_at_dest<SearchDir, Rt>(k, fp_target_time, device_mem.time_at_dest_);
        }
      }
    }
  }
}

template <gpu_direction SearchDir, bool Rt>
__device__ void update_footpaths(unsigned const k, gpu_profile_idx_t const prf_idx,
                                 unsigned short kUnreachable, uint16_t const* lb,
                                 bool const* is_dest,
                                 std::uint32_t const n_locations,
                                 gpu_locations const& gpu_locations,device_memory const& device_mem){
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();
  for(auto idx = global_t_id;
       idx < n_locations; idx += global_stride){
    if(!marked(device_mem.prev_station_mark_, idx)){
      continue;
    }
    auto const l_idx = gpu_location_idx_t{idx};
    auto const& fps = (SearchDir == gpu_direction::kForward)
         ? gpu_locations.gpu_footpaths_out_[prf_idx][l_idx]
           : gpu_locations.gpu_footpaths_in_[prf_idx][l_idx];
    for(auto const& fp: fps){
      atomicAdd(&device_mem.stats_[0].n_footpaths_visited_, 1);
      auto const target = gpu_to_idx(gpu_location_idx_t{fp.target_});
      auto const fp_target_time =
          gpu_clamp(device_mem.tmp_[idx] + dir<SearchDir>(fp.duration()).count());

      if(is_better<SearchDir>(fp_target_time, device_mem.best_[target])
          && is_better<SearchDir>(fp_target_time, device_mem.time_at_dest_[k])){
        auto const lower_bound = lb[gpu_to_idx(gpu_location_idx_t{fp.target_})];
        if(lower_bound == kUnreachable
            || !is_better<SearchDir>(fp_target_time + dir<SearchDir>(lower_bound), device_mem.time_at_dest_[k])){
          atomicAdd(&device_mem.stats_[0].fp_update_prevented_by_lower_bound_, 1);
          continue;
        }
        bool updated = update_arrival<SearchDir>(device_mem.best_, gpu_to_idx(gpu_location_idx_t{fp.target_}), fp_target_time);
        if(updated){
          atomicAdd(&device_mem.stats_[0].n_earliest_arrival_updated_by_footpath_, 1);
          update_arrival<SearchDir>(device_mem.round_times_, k * device_mem.column_count_round_times_ +
                                                                 gpu_to_idx(gpu_location_idx_t{fp.target_}), fp_target_time);
          mark(device_mem.station_mark_, gpu_to_idx(gpu_location_idx_t{fp.target_}));
          if(is_dest[gpu_to_idx(gpu_location_idx_t{fp.target_})]){
            update_time_at_dest<SearchDir, Rt>(k, fp_target_time, device_mem.time_at_dest_);
          }
        }
      }
    }
  }

}

template <gpu_direction SearchDir, bool Rt>
__device__ void update_intermodal_footpaths(unsigned const k, std::uint32_t const n_locations,
                                            uint16_t* dist_to_end, uint32_t dist_to_end_size,
                                            unsigned short kUnreachable, gpu_location_idx_t* gpu_kIntermodalTarget,
                                            device_memory const& device_mem){
  if(dist_to_end_size==0){
    return;
  }
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();
  for(auto idx = global_t_id; idx < n_locations; idx += global_stride){
      if((marked(device_mem.prev_station_mark_, idx) || marked(device_mem.station_mark_, idx)) && dist_to_end[idx] != kUnreachable){
        auto const end_time = gpu_clamp(get_best<SearchDir>(device_mem.best_[idx], device_mem.tmp_[idx]) + dir<SearchDir>(dist_to_end[idx]));
        bool updated = update_arrival<SearchDir>(device_mem.best_,gpu_to_idx(*gpu_kIntermodalTarget),end_time);
        if (updated){
            update_arrival<SearchDir>(device_mem.round_times_,
                                      k * device_mem.column_count_round_times_ + gpu_kIntermodalTarget->v_,
                                      end_time);
            update_time_at_dest<SearchDir, Rt>(k, end_time, device_mem.time_at_dest_);
        }
      }
    }
}


template <gpu_direction SearchDir, bool Rt>
__device__ void raptor_round(unsigned const k, gpu_profile_idx_t const prf_idx,
                             gpu_day_idx_t* base,
                             gpu_clasz_mask_t allowed_claszes, uint16_t* dist_to_end,
                             uint32_t dist_to_end_size,
                             bool* is_dest, uint16_t* lb, int n_days,
                             unsigned short kUnreachable,
                             gpu_location_idx_t* gpu_kIntermodalTarget, short* kMaxTravelTimeTicks,
                             device_memory const& device_mem,gpu_timetable const& gtt){
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();
  for(auto idx = global_t_id; idx < gtt.n_locations_; idx += global_stride){
    auto value_1 = device_mem.round_times_[(k) * device_mem.column_count_round_times_ +idx];
    auto value_2 = device_mem.best_[idx];
    device_mem.best_[idx] =get_best<SearchDir>(value_1, value_2);
    if(is_dest[idx]){
      update_time_at_dest<SearchDir, Rt>(k, device_mem.best_[idx], device_mem.time_at_dest_);
    }
  }

  this_grid().sync();

  if(get_global_thread_id()==0){
    atomicAnd(device_mem.any_station_marked_,0);
  }
  this_grid().sync();
  convert_station_to_route_marks<SearchDir, Rt>(device_mem.station_mark_, device_mem.route_mark_, device_mem.any_station_marked_,
                                                gtt.location_routes_, gtt.n_locations_);

  this_grid().sync();

  if(!*device_mem.any_station_marked_){
    return;
  }

  swap_b_reset(device_mem.prev_station_mark_,device_mem.station_mark_,(gtt.n_locations_/32)+1);

  this_grid().sync();

  (allowed_claszes == 0xffff)? loop_routes<SearchDir, Rt, false>(k, &allowed_claszes, kMaxTravelTimeTicks,
                                                                  lb, base, kUnreachable,n_days,
                                                                  gtt,device_mem)
                           : loop_routes<SearchDir, Rt, true>(k, &allowed_claszes, kMaxTravelTimeTicks,
                                                                 lb, base, kUnreachable,n_days,
                                                                 gtt,device_mem);
  this_grid().sync();

  if(!*device_mem.any_station_marked_){
    return;
  }

  reset_store(device_mem.route_mark_,(gtt.n_routes_/32)+1);

  swap_b_reset(device_mem.prev_station_mark_,device_mem.station_mark_,(gtt.n_locations_/32)+1);

  this_grid().sync();

  update_transfers<SearchDir, Rt>(k, is_dest, dist_to_end, dist_to_end_size,
                                  kUnreachable, lb,
                                  gtt.n_locations_,gtt.locations_, device_mem);
  this_grid().sync();

  update_footpaths<SearchDir, Rt>(k, prf_idx, kUnreachable, lb,
                                  is_dest, gtt.n_locations_,
                                  gtt.locations_,device_mem);
  this_grid().sync();

  update_intermodal_footpaths<SearchDir, Rt>(k, gtt.n_locations_, dist_to_end, dist_to_end_size,  kUnreachable,
                                             gpu_kIntermodalTarget, device_mem);

}

template <gpu_direction SearchDir, bool Rt>
__device__ void init_arrivals(gpu_unixtime_t const worst_time_at_dest,
                              gpu_day_idx_t* base,
                              gpu_delta_t* time_at_dest,
                              gpu_delta const* route_stop_times,
                              gpu_vector_map<gpu_route_idx_t,gpu_interval<gpu_transport_idx_t >> const* route_transport_ranges,
                              gpu_interval<gpu_sys_days> const* date_range){
  auto const global_t_id = get_global_thread_id();
  auto const global_stride = get_global_stride();
  for(auto idx = global_t_id; idx <gpu_kMaxTransfers+1; idx += global_stride){
    time_at_dest[idx] = get_best<SearchDir>(unix_to_gpu_delta(base_of(base,date_range), worst_time_at_dest), time_at_dest[idx]);
  }

}

template <gpu_direction SearchDir, bool Rt>
__global__ void gpu_raptor_kernel(gpu_unixtime_t* start_time,
                                  uint8_t max_transfers,
                                  gpu_unixtime_t* worst_time_at_dest,
                                  gpu_profile_idx_t* prf_idx,
                                  gpu_clasz_mask_t* allowed_claszes,
                                  std::uint16_t* dist_to_end,
                                  std::uint32_t* dist_to_end_size,
                                  gpu_day_idx_t* base,
                                  bool* is_dest,
                                  std::uint16_t* lb,
                                  int* n_days,
                                  std::uint16_t* kUnreachable,
                                  gpu_location_idx_t* kIntermodalTarget,
                                  short* kMaxTravelTimeTicks,
                                  device_memory const device_mem,
                                  gpu_timetable const gtt
                                  ){
  auto const end_k =
      get_smaller(max_transfers, gpu_kMaxTransfers) + 1U;

  init_arrivals<SearchDir, Rt>(*worst_time_at_dest, base, device_mem.time_at_dest_,
                               gtt.route_stop_times_,gtt.route_transport_ranges_,gtt.date_range_);

  this_grid().sync();

  for (auto k = 1U; k != end_k; ++k) {
    if(k!= 1 && (!(*device_mem.any_station_marked_))){
      break;
    }
    raptor_round<SearchDir, Rt>(k, *prf_idx, base, *allowed_claszes,
                                dist_to_end, *dist_to_end_size, is_dest, lb,
                                *n_days, *kUnreachable,
                                kIntermodalTarget, kMaxTravelTimeTicks,
                                device_mem,gtt);
    this_grid().sync();
  }
  this_grid().sync();

}

#define XSTR(s) STR(s)
#define STR(s) #s

#define CUDA_CALL(call) \
    if ((code = (call)) != cudaSuccess) {                     \
      printf("CUDA error: %s at " XSTR(call) " %s:%d\n",     \
             cudaGetErrorString(code), __FILE__, __LINE__); \
      goto fail;                                            \
    }

#define CUDA_COPY_TO_DEVICE(type, target, source, size)                        \
    CUDA_CALL(cudaMalloc(&(target), (size) * sizeof(type)))                          \
    CUDA_CALL(                                                                   \
        cudaMemcpy(target, source, (size) * sizeof(type), cudaMemcpyHostToDevice))

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
                     short* & kMaxTravelTimeTicks_ptr){
  cudaError_t code;
  auto dist_to_end_size = dist_to_dest.size();

  allowed_claszes_ptr = nullptr;
  CUDA_COPY_TO_DEVICE(gpu_clasz_mask_t, allowed_claszes_ptr, &allowed_claszes, 1);
  dist_to_end_ptr = nullptr;
  CUDA_COPY_TO_DEVICE(std::uint16_t, dist_to_end_ptr, dist_to_dest.data(),
                      dist_to_dest.size());
  dist_to_end_size_ptr = nullptr;
  CUDA_COPY_TO_DEVICE(std::uint32_t, dist_to_end_size_ptr, &dist_to_end_size, 1);
  base_ptr = nullptr;
  CUDA_COPY_TO_DEVICE(gpu_day_idx_t, base_ptr, &base, 1);
  is_dest_ptr = nullptr;
  CUDA_COPY_TO_DEVICE(bool, is_dest_ptr, is_dest.data(), is_dest.size());
  lb_ptr = nullptr;
  CUDA_COPY_TO_DEVICE(std::uint16_t, lb_ptr, lb.data(), lb.size());
  n_days_ptr = nullptr;
  CUDA_COPY_TO_DEVICE(int, n_days_ptr, &n_days, 1);
  kUnreachable_ptr = nullptr;
  CUDA_COPY_TO_DEVICE(std::uint16_t, kUnreachable_ptr, &kUnreachable, 1);
  kIntermodalTarget_ptr = nullptr;
  CUDA_COPY_TO_DEVICE(gpu_location_idx_t, kIntermodalTarget_ptr,
                      &kIntermodalTarget, 1);
  kMaxTravelTimeTicks_ptr = nullptr;
  CUDA_COPY_TO_DEVICE(short, kMaxTravelTimeTicks_ptr, &kMaxTravelTimeTicks, 1);
  return;
fail:
  cudaFree(allowed_claszes_ptr);
  cudaFree(dist_to_end_ptr);
  cudaFree(dist_to_end_size_ptr);
  cudaFree(base_ptr);
  cudaFree(is_dest_ptr);
  cudaFree(lb_ptr);
  cudaFree(n_days_ptr);
  cudaFree(kUnreachable_ptr);
  cudaFree(kIntermodalTarget_ptr);
  cudaFree(kMaxTravelTimeTicks_ptr);
  return;
};
void copy_to_device_destroy(gpu_clasz_mask_t*& allowed_claszes,
                            std::uint16_t* & dist_to_end,
                            std::uint32_t* & dist_to_end_size,
                            gpu_day_idx_t* & base,
                            bool* & is_dest,
                            std::uint16_t* & lb,
                            int* & n_days,
                            std::uint16_t* & kUnreachable,
                            gpu_location_idx_t* & kIntermodalTarget,
                            short* & kMaxTravelTimeTicks){
  cudaFree(allowed_claszes);
  allowed_claszes = nullptr;
  cudaFree(dist_to_end);
  dist_to_end = nullptr;
  cudaFree(dist_to_end_size);
  dist_to_end_size = nullptr;
  cudaFree(base);
  base = nullptr;
  cudaFree(is_dest);
  is_dest = nullptr;
  cudaFree(lb);
  lb = nullptr;
  cudaFree(n_days);
  n_days = nullptr;
  cudaFree(kUnreachable);
  kUnreachable = nullptr;
  cudaFree(kIntermodalTarget);
  kIntermodalTarget = nullptr;
  cudaFree(kMaxTravelTimeTicks);
  kMaxTravelTimeTicks = nullptr;
  cuda_check();
};

void launch_kernel(void** args,
                          device_context const& device,
                          cudaStream_t s,
                          gpu_direction search_dir,
                          bool rt) {
  cudaSetDevice(device.id_);
  void* kernel_func = nullptr;
  if (search_dir == gpu_direction::kForward && rt == true) {
    kernel_func = (void*)gpu_raptor_kernel<gpu_direction::kForward, true>;
  } else if (search_dir == gpu_direction::kForward && rt == false) {
    kernel_func = (void*)gpu_raptor_kernel<gpu_direction::kForward, false>;
  } else if (search_dir == gpu_direction::kBackward && rt == true) {
    kernel_func = (void*)gpu_raptor_kernel<gpu_direction::kBackward, true>;
  } else if (search_dir == gpu_direction::kBackward && rt == false) {
    kernel_func = (void*)gpu_raptor_kernel<gpu_direction::kBackward, false>;
  }
  cudaLaunchCooperativeKernel(kernel_func, device.grid_, device.threads_per_block_, args, 0, s);
  cudaDeviceSynchronize();
  cuda_check();
}


void copy_to_gpu_args(gpu_unixtime_t const* start_time,
                      gpu_unixtime_t const* worst_time_at_dest,
                      gpu_profile_idx_t const* prf_idx,
                      gpu_unixtime_t*& start_time_ptr,
                      gpu_unixtime_t*& worst_time_at_dest_ptr,
                      gpu_profile_idx_t*& prf_idx_ptr){
  cudaError_t code;
  CUDA_COPY_TO_DEVICE(gpu_unixtime_t,start_time_ptr,start_time,1);
  CUDA_COPY_TO_DEVICE(gpu_unixtime_t,worst_time_at_dest_ptr,worst_time_at_dest,1);
  CUDA_COPY_TO_DEVICE(gpu_profile_idx_t ,prf_idx_ptr,prf_idx,1);
  return;
  fail:
    cudaFree(start_time_ptr);
    cudaFree(worst_time_at_dest_ptr);
    cudaFree(prf_idx_ptr);
    return;
}
void destroy_copy_to_gpu_args(gpu_unixtime_t* start_time_ptr,
                              gpu_unixtime_t* worst_time_at_dest_ptr,
                              gpu_profile_idx_t* prf_idx_ptr){
  cudaFree(start_time_ptr);
  start_time_ptr = nullptr;
  cudaFree(worst_time_at_dest_ptr);
  worst_time_at_dest_ptr = nullptr;
  cudaFree(prf_idx_ptr);
  prf_idx_ptr = nullptr;
  cuda_check();
}

void* get_gpu_raptor_kernel(gpu_direction search_dir,bool rt){
  void* kernel_func = nullptr;
  if (search_dir == gpu_direction::kForward && rt == true) {
    kernel_func = (void*)gpu_raptor_kernel<gpu_direction::kForward, true>;
  } else if (search_dir == gpu_direction::kForward && rt == false) {
    kernel_func = (void*)gpu_raptor_kernel<gpu_direction::kForward, false>;
  } else if (search_dir == gpu_direction::kBackward && rt == true) {
    kernel_func = (void*)gpu_raptor_kernel<gpu_direction::kBackward, true>;
  } else if (search_dir == gpu_direction::kBackward && rt == false) {
    kernel_func = (void*)gpu_raptor_kernel<gpu_direction::kBackward, false>;
  }
  return kernel_func;
}