#pragma once

#include "nigiri/routing/ch/ch_data.h"
#include "nigiri/routing/query.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}  // namespace nigiri

namespace nigiri::routing {

void obtain_relevant_stops(timetable const& tt,
                           routing::query const& q,
                           profile_idx_t const prf_idx,
                           bitvec& relevant_stops);

}  // namespace nigiri::routing
