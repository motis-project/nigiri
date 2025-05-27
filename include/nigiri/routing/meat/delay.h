#pragma once

#include "nigiri/common/delta_t.h"

namespace nigiri::routing::meat {

/*
 * This code is from the original version by Ben Strasser
 */
inline double delay_prob(delta_t x, delta_t change_time, double max_delay) {
  if (x <= 0)
    return 0.0;
  else if (x <= change_time)
    return static_cast<double>(2 * x) /
           static_cast<double>(6 * change_time - 3 * x);
  else if (x <= max_delay + change_time) {
    x -= change_time;
    return 0.99999 * static_cast<double>(31 * x + 2 * max_delay) /
           static_cast<double>(30 * x + 3 * max_delay);
  } else
    return 1.0;
}

}  // namespace nigiri::routing::meat