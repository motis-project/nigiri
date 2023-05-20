#pragma once

#include "nigiri/types.h"

namespace nigiri {
struct timetable;
struct footpath;
}  // namespace nigiri

namespace nigiri::routing {

struct query;

void dijkstra(timetable const&,
              query const&,
              vecvec<location_idx_t, footpath> const& lb_graph,
              std::vector<std::uint16_t>& dists);

}  // namespace nigiri::routing
