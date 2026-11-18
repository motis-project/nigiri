#pragma once

#include "nigiri/routing/clasz_mask.h"

namespace nigiri::routing {

// What McRAPTOR's criteria updates (mc_criteria.h) need per query; the search
// fills it from the query. Plain RAPTOR takes and ignores it.
struct mc_context {
  clasz_mask_t avoided_{to_mask(clasz::kAir)};  // mode_filter_dim
};

}  // namespace nigiri::routing
