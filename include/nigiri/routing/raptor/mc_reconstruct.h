#pragma once

#include <span>

#include "nigiri/routing/mc_criteria.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
struct rt_timetable;
}  // namespace nigiri

namespace nigiri::routing {

struct query;
struct journey;

// Shared by the CPU and the GPU McRAPTOR.

// The transit legs and footpaths behind a destination label; `chain`: its
// breadcrumbs back to the start. The offset legs follow in
// mc_reconstruct_offsets().
template <direction SearchDir>
void mc_add_legs(timetable const&,
                 rt_timetable const*,
                 day_idx_t base_day,
                 bool is_intermodal_dest,
                 std::span<mc_breadcrumb const> chain,
                 journey&);

// offset legs and the start footpath around the transit legs
template <direction SearchDir>
void mc_reconstruct_offsets(timetable const&,
                            rt_timetable const*,
                            query const&,
                            journey&);

}  // namespace nigiri::routing
