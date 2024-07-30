#pragma once

#include <array>
#include <vector>

#include "date/date.h"

#include "cista/containers/bitvec.h"
#include "cista/containers/flat_matrix.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/limits.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::routing {

struct raptor_state {
  raptor_state() = default;
  raptor_state(raptor_state const&) = delete;
  raptor_state& operator=(raptor_state const&) = delete;
  raptor_state(raptor_state&&) = default;
  raptor_state& operator=(raptor_state&&) = default;
  ~raptor_state() = default;

  void resize(unsigned n_locations,
              unsigned n_routes,
              unsigned n_rt_transports);

  void print(timetable const& tt, date::sys_days, delta_t invalid);

  std::vector<std::array<delta_t, kMaxVias + 1>> tmp_;
  std::vector<std::array<delta_t, kMaxVias + 1>> best_;
  cista::raw::flat_matrix<std::array<delta_t, kMaxVias + 1>> round_times_;
  bitvec station_mark_;
  bitvec prev_station_mark_;
  bitvec route_mark_;
  bitvec rt_transport_mark_;
  bitvec end_reachable_;
};

}  // namespace nigiri::routing
