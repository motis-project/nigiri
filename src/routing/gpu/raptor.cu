#include "nigiri/routing/gpu/raptor.h"

#include <cstdio>
#include <iostream>

#include "thrust/device_vector.h"

#include "nigiri/timetable.h"

namespace nigiri::routing::gpu {

raptor_state& raptor_state::resize(unsigned n_locations,
                                   unsigned n_routes,
                                   unsigned n_rt_transports) {
  n_locations_ = n_locations;
  tmp_storage_.resize(n_locations * (kMaxVias + 1));
  best_storage_.resize(n_locations * (kMaxVias + 1));
  round_times_storage_.resize(n_locations * (kMaxVias + 1) *
                              (kMaxTransfers + 1));
  station_mark_.resize(n_locations);
  prev_station_mark_.resize(n_locations);
  route_mark_.resize(n_routes);
  rt_transport_mark_.resize(n_rt_transports);
  end_reachable_.resize(n_locations);
  return *this;
}

__global__ void run_raptor() { printf("Hello World from device!\n"); }

thrust::device_vector<std::uint8_t> copy_timetable(timetable const& tt) {
  auto const buf = cista::serialize<cista::mode::CAST>(tt);
  return {begin(buf), end(buf)};
}

}  // namespace nigiri::routing::gpu