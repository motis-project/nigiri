#include "nigiri/routing/gpu.h"

#include <cstdio>

extern "C" {

#define XSTR(s) STR(s)
#define STR(s) #s

#define CUDA_CALL(call)                                   \
  if ((code = call) != cudaSuccess) {                     \
    printf("CUDA error: %s at " STR(call) " %s:%d\n",     \
           cudaGetErrorString(code), __FILE__, __LINE__); \
    goto fail;                                            \
  }

#define CUDA_COPY_TO_DEVICE(type, target, source, size)                        \
  CUDA_CALL(cudaMalloc(&target, size * sizeof(type)))                          \
  CUDA_CALL(                                                                   \
      cudaMemcpy(target, source, size * sizeof(type), cudaMemcpyHostToDevice)) \
  device_bytes += size * sizeof(type);

struct gpu_timetable {
  gpu_delta* route_stop_times_{nullptr};
};

struct gpu_timetable* create_gpu_timetable(gpu_delta const* route_stop_times,
                                           std::uint32_t n_route_stop_times) {
  size_t device_bytes = 0U;

  cudaError_t code;
  gpu_timetable* gtt =
      static_cast<gpu_timetable*>(malloc(sizeof(gpu_timetable)));
  if (gtt == nullptr) {
    printf("nigiri gpu raptor: malloc for gpu_timetable failed\n");
    return nullptr;
  }

  gtt->route_stop_times_ = nullptr;

  CUDA_COPY_TO_DEVICE(gpu_delta, gtt->route_stop_times_, route_stop_times,
                      n_route_stop_times);

  return gtt;

fail:
  cudaFree(gtt->route_stop_times_);
  free(gtt);
  return nullptr;
}

}  // extern "C"