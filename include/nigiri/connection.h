#pragma once

#include "nigiri/stop.h"
#include "nigiri/types.h"

namespace nigiri {

struct connection {
  using trip_con_idx_t = uint16_t;

  stop::value_type dep_stop_;
  stop::value_type arr_stop_;
  delta dep_time_ = delta{0};
  delta arr_time_ = delta{0};
  transport_idx_t transport_idx_;
  trip_con_idx_t trip_con_idx_;  // n´th connection of this transport(trip),
                                 // starting with 0
};

}  // namespace nigiri