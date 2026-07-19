#pragma once

#include <cinttypes>
#include <utility>
#include <vector>

#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::routing {

// Footpath-component graph: a time-independent relaxation of the timetable
// used to compute per-transfer lower bounds (travel time + number of
// boardings to the destination).
//
// - Everything connected by a footpath (any profile) or a parent/child
//   station relation collapses into one component (zero cost, zero
//   transfers inside a component -> footpaths/transfers vanish entirely).
// - Every route's stop sequence becomes a component sequence (consecutive
//   duplicates compressed). Routes with identical component sequences are
//   merged; each segment stores the duration of the fastest trip on this
//   segment over all merged routes (dwell times dropped).
// - in/out_allowed, traffic days, transfer times and mode restrictions are
//   ignored.
//
// All simplifications are relaxations -> distances in this graph are
// admissible lower bounds for any query on the real timetable WITHOUT
// realtime data (rt transports may be faster than the static minimum).
using component_idx_t = cista::strong<std::uint32_t, struct _component_idx>;
using comp_route_idx_t = cista::strong<std::uint32_t, struct _comp_route_idx>;

struct component_graph {
  std::uint32_t n_components_{0U};
  vector_map<location_idx_t, component_idx_t> location_component_;
  vecvec<comp_route_idx_t, component_idx_t> seqs_;  // compressed comp seqs
  vecvec<comp_route_idx_t, std::uint16_t> durations_;  // per segment, minutes
  vecvec<component_idx_t, comp_route_idx_t> comp_routes_;  // inverted index
};

component_graph build_component_graph(timetable const&);

// Result of the time-independent component RAPTOR:
//   tt_[c] = lower bound on the remaining travel time from component c to
//            the seeds (minutes), kUnreachableTt if none found
//   ic_[c] = lower bound on the remaining number of boardings (0 for seeds)
// A label with arrival t in round k at a location of component c cannot
// produce a destination arrival better than t + tt_[c] in any round < k +
// ic_[c] -> compare against time_at_dest[k + ic_[c]].
struct component_lb {
  static constexpr auto const kUnreachableTt =
      std::numeric_limits<std::uint16_t>::max();
  static constexpr auto const kUnreachableIc =
      std::numeric_limits<std::uint8_t>::max();
  std::vector<std::uint16_t> tt_;
  std::vector<std::uint8_t> ic_;
};

// CPU reference implementation of the GPU kernels (tests/documentation).
// dir = search direction of the raptor that will consume the bounds:
//   kForward  -> bounds measure distance TO the seeds (reverse scan)
//   kBackward -> bounds measure distance FROM the seeds (forward scan)
// seeds = (component, initial duration), e.g. destination components with 0
// or intermodal egress components with their offset duration.
component_lb compute_component_lb(
    component_graph const&,
    direction,
    std::vector<std::pair<component_idx_t, std::uint16_t>> const& seeds,
    unsigned max_rounds);

}  // namespace nigiri::routing
