#include "nigiri/routing/gpu/raptor.h"

#include <cstdio>
#include <iostream>

#include "nigiri/timetable.h"

namespace nigiri::routing::gpu {

__global__ void raptor(timetable const& tt) {}

void say_hello() {
  std::cout << "RUNNING CUDA CODE\n";
  raptor<<<1, 1>>>();
  cudaDeviceSynchronize();
}

}  // namespace nigiri::routing::gpu