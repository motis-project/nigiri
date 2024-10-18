#pragma once

#include <cuda_runtime.h>
#include "nigiri/routing/gpu_timetable.h"
#include <iostream>
#include <cstdio>

#define XSTR(s) STR(s)
#define STR(s) #s

#define CUDA_CALL(call)                                   \
    if ((code = call) != cudaSuccess) {                     \
      printf("CUDA error: %s at " XSTR(call) " %s:%d\n",     \
             cudaGetErrorString(code), __FILE__, __LINE__); \
      goto fail;                                            \
    }

#define CUDA_COPY_TO_DEVICE(type, target, source, size)                        \
    CUDA_CALL(cudaMalloc(&target, size * sizeof(type)))                          \
    CUDA_CALL(                                                                   \
        cudaMemcpy(target, source, size * sizeof(type), cudaMemcpyHostToDevice)) \
    device_bytes += size * sizeof(type);

template <typename KeyType, typename ValueType>
void copy_gpu_vecvec_to_device(const gpu_vecvec<KeyType, ValueType>* h_vecvec,gpu_vecvec<KeyType, ValueType>*& d_vecvec, size_t& device_bytes, cudaError_t& code) {

  d_vecvec = nullptr;
  gpu_vector<ValueType>* d_data = nullptr;
  gpu_vector<gpu_base_t<KeyType>>* d_bucket_starts = nullptr;
  CUDA_CALL(cudaMalloc(&d_vecvec, sizeof(gpu_vecvec<KeyType, ValueType>)));

  CUDA_CALL(cudaMalloc(&d_bucket_starts, h_vecvec->bucket_starts_.size() * sizeof(gpu_base_t<KeyType>)));
  CUDA_CALL(cudaMemcpy(d_bucket_starts, h_vecvec->bucket_starts_.data(),
                        h_vecvec->bucket_starts_.size() * sizeof(gpu_base_t<KeyType>), cudaMemcpyHostToDevice));

  CUDA_CALL(cudaMalloc(&d_data, h_vecvec->data_.size() * sizeof(ValueType)));
  CUDA_CALL(cudaMemcpy(d_data, h_vecvec->data_.data(),
                        h_vecvec->data_.size() * sizeof(ValueType), cudaMemcpyHostToDevice));

  CUDA_CALL(cudaMemcpy(&(d_vecvec->bucket_starts_), &d_bucket_starts, sizeof(gpu_vector<gpu_base_t<KeyType>>), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(&(d_vecvec->data_), &d_data, sizeof(gpu_vector<ValueType>), cudaMemcpyHostToDevice));

  CUDA_CALL(cudaMemcpy(&(d_vecvec->bucket_starts_.used_size_), &h_vecvec->bucket_starts_.used_size_, sizeof(h_vecvec->bucket_starts_.used_size_), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(&(d_vecvec->bucket_starts_.allocated_size_), &h_vecvec->bucket_starts_.allocated_size_, sizeof(h_vecvec->bucket_starts_.allocated_size_), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(&(d_vecvec->bucket_starts_.self_allocated_), &h_vecvec->bucket_starts_.self_allocated_, sizeof(h_vecvec->bucket_starts_.self_allocated_), cudaMemcpyHostToDevice));

  CUDA_CALL(cudaMemcpy(&(d_vecvec->data_.used_size_), &h_vecvec->data_.used_size_, sizeof(h_vecvec->data_.used_size_), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(&(d_vecvec->data_.allocated_size_), &h_vecvec->data_.allocated_size_, sizeof(h_vecvec->data_.allocated_size_), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(&(d_vecvec->data_.self_allocated_), &h_vecvec->data_.self_allocated_, sizeof(h_vecvec->data_.self_allocated_), cudaMemcpyHostToDevice));

  device_bytes += sizeof(gpu_vecvec<KeyType, ValueType>);
  return;
fail:
  std::cerr << "ERROR VECVEC" << std::endl;
  if (d_bucket_starts) cudaFree(d_bucket_starts);
  if (d_data) cudaFree(d_data);
  if (d_vecvec) cudaFree(d_vecvec);
  d_vecvec = nullptr;
  return;
}

template <typename KeyType, typename ValueType>
void copy_gpu_vector_map_to_device(const gpu_vector_map<KeyType, ValueType>* h_map,
                                   gpu_vector_map<KeyType, ValueType>*& d_map,
                                   size_t& device_bytes, cudaError_t& code) {

  d_map = nullptr;
  gpu_vector<ValueType>* d_data = nullptr;
  CUDA_CALL(cudaMalloc(&d_map, sizeof(gpu_vector_map<KeyType, ValueType>)));
  CUDA_CALL(cudaMalloc(&d_data, h_map->size() * sizeof(ValueType)));
  CUDA_CALL(cudaMemcpy(d_data, h_map->data(), h_map->size() * sizeof(ValueType), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(&(d_map->el_), &d_data, sizeof(gpu_vector<ValueType>), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(&(d_map->used_size_), &h_map->used_size_, sizeof(h_map->used_size_), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(&(d_map->allocated_size_), &h_map->allocated_size_, sizeof(h_map->allocated_size_), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(&(d_map->self_allocated_), &h_map->self_allocated_, sizeof(h_map->self_allocated_), cudaMemcpyHostToDevice));

  device_bytes += sizeof(gpu_vector_map<KeyType, ValueType>);
  return;
fail:
  std::cerr << "ERROR VECMAP" << std::endl;
    if (d_data) cudaFree(d_data);
    if (d_map) cudaFree(d_map);
    d_map = nullptr;
    return;
}
void copy_gpu_locations_to_device(const gpu_locations* h_locations, gpu_locations& d_locations, size_t& device_bytes, cudaError_t& code) {

  d_locations = gpu_locations{nullptr, nullptr, nullptr};

  copy_gpu_vector_map_to_device(h_locations->transfer_time_, d_locations.transfer_time_, device_bytes, code);
  copy_gpu_vecvec_to_device(h_locations->gpu_footpaths_in_, d_locations.gpu_footpaths_in_, device_bytes, code);
  copy_gpu_vecvec_to_device(h_locations->gpu_footpaths_out_, d_locations.gpu_footpaths_out_, device_bytes, code);

  device_bytes += sizeof(gpu_locations);

  return;
}
template <typename KeyType, typename ValueType>
void free_gpu_vecvec(gpu_vecvec<KeyType, ValueType>* d_vecvec) {
  if (!d_vecvec) return;

  cudaError_t code;
  gpu_base_t<KeyType>* d_bucket_starts = nullptr;
  ValueType* d_data = nullptr;

  cudaMemcpy(&d_bucket_starts, &(d_vecvec->bucket_starts_), sizeof(gpu_base_t<KeyType>*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&d_data, &(d_vecvec->data_), sizeof(ValueType*), cudaMemcpyDeviceToHost);

  if (d_bucket_starts) {
    code = cudaFree(d_bucket_starts);
    if (code != cudaSuccess) {
      std::cerr << "Error freeing d_bucket_starts: " << cudaGetErrorString(code) << std::endl;
    }
  }
  if (d_data) {
    code = cudaFree(d_data);
    if (code != cudaSuccess) {
      std::cerr << "Error freeing d_data: " << cudaGetErrorString(code) << std::endl;
    }
  }

  code = cudaFree(d_vecvec);
  if (code != cudaSuccess) {
    std::cerr << "Error freeing gpu_vecvec: " << cudaGetErrorString(code) << std::endl;
  }
}

template <typename KeyType, typename ValueType>
void free_gpu_vector_map(gpu_vector_map<KeyType, ValueType>* d_map) {
  if (!d_map) return;

  cudaError_t code;
  ValueType* d_data = nullptr;

  cudaMemcpy(&d_data, &(d_map->el_), sizeof(ValueType*), cudaMemcpyDeviceToHost);

  if (d_data) {
    code = cudaFree(d_data);
    if (code != cudaSuccess) {
      std::cerr << "Error freeing d_data: " << cudaGetErrorString(code) << std::endl;
    }
  }

  code = cudaFree(d_map);
  if (code != cudaSuccess) {
    std::cerr << "Error freeing gpu_vector_map: " << cudaGetErrorString(code) << std::endl;
  }
}

void free_gpu_locations(gpu_locations& d_locations) {
  if (d_locations.transfer_time_) free_gpu_vector_map(d_locations.transfer_time_);
  if (d_locations.gpu_footpaths_in_) free_gpu_vecvec(d_locations.gpu_footpaths_in_);
  if (d_locations.gpu_footpaths_out_) free_gpu_vecvec(d_locations.gpu_footpaths_out_);
}

struct gpu_timetable* create_gpu_timetable(gpu_delta const* route_stop_times,
                                           std::uint32_t  n_route_stop_times,
                                           gpu_vecvec<gpu_route_idx_t,gpu_value_type> const* route_location_seq,
                                           gpu_vecvec<gpu_location_idx_t , gpu_route_idx_t> const* location_routes,
                                           std::uint32_t const n_locations,
                                           std::uint32_t const n_routes,
                                           gpu_vector_map<gpu_route_idx_t,gpu_interval<std::uint32_t>> const* route_stop_time_ranges,
                                           gpu_vector_map<gpu_route_idx_t,gpu_interval<gpu_transport_idx_t >> const* route_transport_ranges,
                                           gpu_vector_map<gpu_bitfield_idx_t, gpu_bitfield> const* bitfields,
                                           gpu_vector_map<gpu_transport_idx_t,gpu_bitfield_idx_t> const* transport_traffic_days,
                                           gpu_interval<gpu_sys_days> const* date_range,
                                           gpu_locations const* locations,
                                           gpu_vector_map<gpu_route_idx_t, gpu_clasz> const* route_clasz) {
  size_t device_bytes = 0U;
  cudaError_t code;
  gpu_timetable* gtt =
      static_cast<gpu_timetable*>(malloc(sizeof(gpu_timetable)));
  if (gtt == nullptr) {
    return nullptr;
  }
  gtt->route_stop_times_ = nullptr;
  CUDA_COPY_TO_DEVICE(gpu_delta, gtt->route_stop_times_, route_stop_times,
                      n_route_stop_times);
   copy_gpu_vecvec_to_device(route_location_seq,gtt->route_location_seq_,device_bytes,code);

   copy_gpu_vecvec_to_device(location_routes,gtt->location_routes_, device_bytes, code);
  gtt->n_locations_ = n_locations;
  gtt->n_routes_ = n_routes;
  copy_gpu_vector_map_to_device(route_stop_time_ranges,gtt->route_stop_time_ranges_,device_bytes,code);
  copy_gpu_vector_map_to_device(route_transport_ranges,gtt->route_transport_ranges_,device_bytes,code);
  copy_gpu_vector_map_to_device(bitfields,gtt->bitfields_,device_bytes,code);
  copy_gpu_vector_map_to_device(transport_traffic_days,gtt->transport_traffic_days_,device_bytes,code);
  gtt->date_range_ = nullptr;
  using gpu_date_range = gpu_interval<gpu_sys_days>;
  CUDA_COPY_TO_DEVICE(gpu_date_range , gtt->date_range_, date_range,1);
  copy_gpu_locations_to_device(locations,gtt->locations_,device_bytes,code);
  copy_gpu_vector_map_to_device(route_clasz,gtt->route_clasz_,device_bytes,code);

  cudaDeviceSynchronize();
  if(!gtt->route_stop_times_||!gtt->route_location_seq_ || !gtt->location_routes_||!gtt->n_locations_||!gtt->n_routes_||!gtt->route_stop_time_ranges_||!gtt->route_transport_ranges_||!gtt->bitfields_||
      !gtt->transport_traffic_days_||! gtt->date_range_||!gtt->route_clasz_){
    std::cerr << "something went wrong, one attribute ist nullptr" << std::endl;
    goto fail;
  }
  gtt->cpu_date_range_ = date_range;

  device_bytes += sizeof(gpu_interval<gpu_sys_days> const*);
  return gtt;


fail:
  destroy_gpu_timetable(gtt);
  return nullptr;
}
void destroy_gpu_timetable(gpu_timetable* gtt) {
  if (!gtt) return;

  cudaError_t code;

  if (gtt->route_stop_times_) {
    code = cudaFree(gtt->route_stop_times_);
    if (code != cudaSuccess) {
      std::cerr << "Error freeing route_stop_times_: " << cudaGetErrorString(code) << std::endl;
    }
    gtt->route_stop_times_ = nullptr;
  }

  if (gtt->route_location_seq_) {
    free_gpu_vecvec(gtt->route_location_seq_);
    gtt->route_location_seq_ = nullptr;
  }

  if (gtt->location_routes_) {
    free_gpu_vecvec(gtt->location_routes_);
    gtt->location_routes_ = nullptr;
  }
  if (gtt->route_stop_time_ranges_) {
    free_gpu_vector_map(gtt->route_stop_time_ranges_);
    gtt->route_stop_time_ranges_ = nullptr;
  }

  if (gtt->route_transport_ranges_) {
    free_gpu_vector_map(gtt->route_transport_ranges_);
    gtt->route_transport_ranges_ = nullptr;
  }

  if (gtt->bitfields_) {
    free_gpu_vector_map(gtt->bitfields_);
    gtt->bitfields_ = nullptr;
  }

  if (gtt->transport_traffic_days_) {
    free_gpu_vector_map(gtt->transport_traffic_days_);
    gtt->transport_traffic_days_ = nullptr;
  }

  if (gtt->date_range_) {
    code = cudaFree(gtt->date_range_);
    if (code != cudaSuccess) {
      std::cerr << "Error freeing date_range_: " << cudaGetErrorString(code) << std::endl;
    }
    gtt->date_range_ = nullptr;
  }

  free_gpu_locations(gtt->locations_);

  if (gtt->route_clasz_) {
    free_gpu_vector_map(gtt->route_clasz_);
    gtt->route_clasz_ = nullptr;
  }

  free(gtt);
  cudaDeviceSynchronize();
  auto const last_error = cudaGetLastError();
  if (last_error != cudaSuccess) {
    printf("CUDA error: %s at " STR(last_error) " %s:%d\n",
           cudaGetErrorString(last_error), __FILE__, __LINE__);
  }
}