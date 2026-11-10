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

// Journey reconstruction shared by the CPU and the GPU McRAPTOR.

// The transit legs, with the footpaths between them, behind a destination
// label: chain holds its breadcrumbs from the label back to the start (as
// parent_ links them), times relative to base_day. A ride into the intermodal
// target ends the journey at its alighting stop (the offset leg follows in
// mc_reconstruct_offsets). Legs end up chronological.
template <direction SearchDir>
void mc_add_legs(timetable const&,
                 rt_timetable const*,
                 day_idx_t base_day,
                 bool is_intermodal_dest,
                 std::span<mc_breadcrumb const> chain,
                 journey&);

// Offset legs and the start footpath around the transit legs (as gpu raptor,
// without optimize_footpaths), td footpaths cut to their walk.
template <direction SearchDir>
void mc_reconstruct_offsets(timetable const&,
                            rt_timetable const*,
                            query const&,
                            journey&);

}  // namespace nigiri::routing
