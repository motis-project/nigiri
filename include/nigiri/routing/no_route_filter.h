#pragma once

#include "nigiri/types.h"

namespace nigiri::routing {

struct no_route_filter {
  bool is_filtered(route_idx_t) const { return false; }
};

}  // namespace nigiri::routing