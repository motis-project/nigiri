#pragma once

#include "cista/reflection/printable.h"

#include "nigiri/types.h"

namespace nigiri {

struct footpath {
  CISTA_PRINTABLE(footpath, "target", "duration")
  location_idx_t target_;
  duration_t duration_;
};

}  // namespace nigiri