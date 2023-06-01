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

void dijkstra(timetable const&,
              query const&,
              vecvec<component_idx_t, component_idx_t> const& lb_graph,
              std::vector<std::uint8_t>& dists);

}  // namespace nigiri::routing
