#pragma once
#include "nigiri/routing/gpu_raptor_state.h"
#include <iostream>

#include <cuda_runtime.h>


std::pair<dim3, dim3> inline get_launch_paramters(
    cudaDeviceProp const& prop, int32_t const concurrency_per_device,void* kernel) {
    int32_t block_dim_x = 32;  // must always be 32!
    int32_t block_dim_y = 14;   // range [1, ..., 32]
    int32_t block_size = block_dim_x * block_dim_y;

    auto const mp_count = prop.multiProcessorCount / concurrency_per_device;

    cudaFuncAttributes attr;
    cudaError_t error = cudaFuncGetAttributes(&attr, kernel);


    int32_t max_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm,kernel,block_size,0);
    if (max_blocks_per_sm == 0){
      throw std::runtime_error("Error: max_blocks_per_sm is 0. Please check block_dim_y and lower it.\n");
    }
    int32_t total_blocks = mp_count * max_blocks_per_sm;
    dim3 threads_per_block(block_dim_x, block_dim_y, 1);
    dim3 grid(total_blocks, 1, 1);

    return {threads_per_block, grid};
}

device_context::device_context(device_id const device_id,void* kernel)
    : id_(device_id), kernel_{kernel} {
  cudaSetDevice(id_);
  cuda_check();

  cudaGetDeviceProperties(&props_, device_id);
  cuda_check();

  std::tie(threads_per_block_, grid_) =
      get_launch_paramters(props_, 1,kernel);

  cudaStreamCreate(&proc_stream_);
  cuda_check();
  cudaStreamCreate(&transfer_stream_);
  cuda_check();
}

void device_context::destroy() {
  if (proc_stream_ != nullptr) {
    cudaStreamDestroy(proc_stream_);
  }
  if (transfer_stream_ != nullptr) {
    cudaStreamDestroy(transfer_stream_);
  }
  cuda_check();
}

host_memory::host_memory(uint32_t n_locations,
                         uint32_t n_routes,
                         uint32_t row_count_round_times,
                         uint32_t column_count_round_times,
                         gpu_delta_t kInvalid):row_count_round_times_{row_count_round_times},
                             column_count_round_times_{column_count_round_times},
                             round_times_(row_count_round_times*column_count_round_times,kInvalid),
                             stats_(1),
                             tmp_(n_locations,kInvalid),
                             best_(n_locations,kInvalid),
                             station_mark_((n_locations / 32) + 1,0),
                             prev_station_mark_((n_locations / 32) + 1),
                             route_mark_((n_locations / 32) + 1),kInvalid_{kInvalid},synced{true}{}

device_memory::device_memory(uint32_t n_locations,
                             uint32_t n_routes,
                             uint32_t row_count_round_times,
                             uint32_t column_count_round_times,
                             gpu_delta_t kInvalid)
    : n_locations_{n_locations},
      n_routes_{n_routes},
      row_count_round_times_{row_count_round_times},
      column_count_round_times_{column_count_round_times}{
  tmp_ = nullptr;
  cudaMalloc(&tmp_, n_locations_ * sizeof(gpu_delta_t));
  cuda_check();
  time_at_dest_ = nullptr;
  cudaMalloc(&time_at_dest_, (gpu_kMaxTransfers+1) *sizeof(gpu_delta_t));
  cuda_check();
  best_ = nullptr;
  cudaMalloc(&best_, n_locations_ * sizeof(gpu_delta_t));
  cuda_check();
  round_times_ = nullptr;
  cudaMalloc(&round_times_, row_count_round_times_ * column_count_round_times_ *
                                sizeof(gpu_delta_t));
  cuda_check();
  station_mark_ = nullptr;
  cudaMalloc(&station_mark_, ((n_locations_/32)+1) * sizeof(uint32_t));
  cuda_check();
  prev_station_mark_ = nullptr;
  cudaMalloc(&prev_station_mark_, ((n_locations_/32)+1) * sizeof(uint32_t));
  cuda_check();
  route_mark_ = nullptr;
  cudaMalloc(&route_mark_, ((n_routes_/32)+1) * sizeof(uint32_t));
  cuda_check();
  any_station_marked_ = nullptr;
  cudaMalloc(&any_station_marked_, sizeof(int));
  cuda_check();
  stats_ = nullptr;
  cudaMalloc(&stats_,sizeof(gpu_raptor_stats));
  cuda_check();
  kInvalid_ = kInvalid;
  cudaDeviceSynchronize();
  this->reset_async(nullptr);
}

void device_memory::destroy() {
  cudaFree(time_at_dest_);
  time_at_dest_ = nullptr;
  cudaFree(tmp_);
  tmp_ = nullptr;
  cudaFree(best_);
  best_ = nullptr;
  cudaFree(round_times_);
  round_times_ = nullptr;
  cudaFree(station_mark_);
  station_mark_ = nullptr;
  cudaFree(prev_station_mark_);
  prev_station_mark_ = nullptr;
  cudaFree(route_mark_);
  route_mark_ = nullptr;
  cudaFree(any_station_marked_);
  any_station_marked_ = nullptr;
  cudaFree(stats_);
  stats_ = nullptr;
  cuda_check();
  cudaDeviceSynchronize();
}

void device_memory::reset_async(cudaStream_t s) {
  std::vector<gpu_delta_t> invalid_time_at_dest((gpu_kMaxTransfers+1), kInvalid_);
  cudaMemcpyAsync(time_at_dest_, invalid_time_at_dest.data(), (gpu_kMaxTransfers+1) * sizeof(gpu_delta_t), cudaMemcpyHostToDevice, s);
  std::vector<gpu_delta_t> invalid_n_locations(n_locations_, kInvalid_);
  cudaMemcpyAsync(tmp_,invalid_n_locations.data(), n_locations_ * sizeof(gpu_delta_t), cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(best_,invalid_n_locations.data(), n_locations_ * sizeof(gpu_delta_t), cudaMemcpyHostToDevice, s);
  std::vector<gpu_delta_t> invalid_round_times(column_count_round_times_*row_count_round_times_, kInvalid_);
  cudaMemcpyAsync(round_times_,invalid_round_times.data(),column_count_round_times_*row_count_round_times_ * sizeof(gpu_delta_t), cudaMemcpyHostToDevice, s);
  cudaMemsetAsync(station_mark_, 0000, ((n_locations_/32)+1)*sizeof(uint32_t), s);
  cudaMemsetAsync(prev_station_mark_, 0000, ((n_locations_/32)+1)*sizeof(uint32_t), s);
  cudaMemsetAsync(route_mark_, 0000, ((n_routes_/32)+1)*sizeof(uint32_t), s);
  cudaMemsetAsync(any_station_marked_, 0000, sizeof(int), s);
  gpu_raptor_stats init_value = {};
  cudaMemcpyAsync(stats_, &init_value, sizeof(gpu_raptor_stats), cudaMemcpyHostToDevice, s);
}
void device_memory::next_start_time_async(cudaStream_t s) {
  std::vector<gpu_delta_t> invalid_n_locations(n_locations_, kInvalid_);
  cudaMemcpyAsync(tmp_,invalid_n_locations.data(), n_locations_ * sizeof(gpu_delta_t), cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(best_,invalid_n_locations.data(), n_locations_ * sizeof(gpu_delta_t), cudaMemcpyHostToDevice, s);
  cudaMemsetAsync(station_mark_, 0000, ((n_locations_/32)+1)*sizeof(uint32_t), s);
  cudaMemsetAsync(prev_station_mark_, 0000, ((n_locations_/32)+1)*sizeof(uint32_t), s);
  cudaMemsetAsync(route_mark_, 0000, ((n_routes_/32)+1)*sizeof(uint32_t), s);
}
void device_memory::reset_arrivals_async(cudaStream_t s) {
  std::vector<gpu_delta_t> invalid_time_at_dest((gpu_kMaxTransfers+1), kInvalid_);

  cudaMemcpyAsync(time_at_dest_, invalid_time_at_dest.data(), (gpu_kMaxTransfers+1) * sizeof(gpu_delta_t), cudaMemcpyHostToDevice, s);

  std::vector<gpu_delta_t> invalid_round_times(column_count_round_times_*row_count_round_times_, kInvalid_);
  cudaMemcpyAsync(round_times_,invalid_round_times.data(),column_count_round_times_*row_count_round_times_ * sizeof(gpu_delta_t), cudaMemcpyHostToDevice, s);
}
mem::mem(uint32_t n_locations,
         uint32_t n_routes,
         uint32_t row_count_round_times_,
         uint32_t column_count_round_times_,
         gpu_delta_t kInvalid,
         device_id const device_id,
         void* kernel)
    : host_{n_locations,n_routes, row_count_round_times_, column_count_round_times_,kInvalid},
      device_{n_locations, n_routes, row_count_round_times_, column_count_round_times_, kInvalid},
      context_{device_id,kernel} {}

mem::~mem() {
  device_.destroy();
  context_.destroy();
}

void mem::reset_arrivals_async(){
  device_.reset_arrivals_async(context_.proc_stream_);
  cuda_sync_stream(context_.proc_stream_);
}

void mem::next_start_time_async(){
  device_.next_start_time_async(context_.proc_stream_);
  cuda_sync_stream(context_.proc_stream_);
}

void mem::copy_host_to_device() {
  cudaMemcpy(device_.best_, host_.best_.data(), host_.best_.size() * sizeof(gpu_delta_t), cudaMemcpyHostToDevice);
  cudaMemcpy(device_.round_times_, host_.round_times_.data(), host_.round_times_.size() * sizeof(gpu_delta_t), cudaMemcpyHostToDevice);
  cudaMemcpy(device_.station_mark_, host_.station_mark_.data(), host_.station_mark_.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
  host_.synced = true;
  cuda_check();
}

void mem::fetch_arrivals_async(){
  auto s = context_.transfer_stream_;
  cudaMemcpyAsync(
      host_.round_times_.data(), device_.round_times_,
      sizeof(gpu_delta_t)*host_.row_count_round_times_*host_.column_count_round_times_, cudaMemcpyDeviceToHost, s);
  cudaMemcpyAsync(
      host_.stats_.data(), device_.stats_,
      sizeof(gpu_raptor_stats), cudaMemcpyDeviceToHost, s);
  cudaMemcpyAsync(
      host_.tmp_.data(), device_.tmp_,
      sizeof(gpu_delta_t)*device_.n_locations_, cudaMemcpyDeviceToHost, s);
  cudaMemcpyAsync(
      host_.best_.data(), device_.best_,
      sizeof(gpu_delta_t)*device_.n_locations_, cudaMemcpyDeviceToHost, s);
  cudaMemcpyAsync(
      host_.station_mark_.data(), device_.station_mark_,
      sizeof(uint32_t)*((device_.n_locations_/32)+1), cudaMemcpyDeviceToHost, s);
  cudaMemcpyAsync(
      host_.prev_station_mark_.data(), device_.prev_station_mark_,
      sizeof(uint32_t)*((device_.n_locations_/32)+1), cudaMemcpyDeviceToHost, s);
  cudaMemcpyAsync(
      host_.route_mark_.data(), device_.route_mark_,
      sizeof(uint32_t)*((device_.n_routes_/32)+1), cudaMemcpyDeviceToHost, s);
  cuda_check();
}

void mem::copy_device_to_host() {
  cuda_sync_stream(context_.proc_stream_);
  cuda_check();
  fetch_arrivals_async();
  cuda_check();
  cuda_sync_stream(context_.transfer_stream_);
  cuda_check();
}
