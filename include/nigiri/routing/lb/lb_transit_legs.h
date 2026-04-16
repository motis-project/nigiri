#pragma once

#include "nigiri/query_generator/generator.h"
#include "nigiri/routing/query.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

struct lb_transit_legs_state {
  bitvec station_mark_;
  bitvec prev_station_mark_;
  bitvec route_mark_;
  vector_map<location_idx_t, std::uint8_t> lb_;
};

// SearchDir refers to the direction of the main routing query
// fwd: finds the minimum number of transit legs backward from the destination
// bwd: finds the minimum number of transit legs forward from the destination
template <direction SearchDir>
void lb_transit_legs(timetable const&, query const&, lb_transit_legs_state&);

struct lb_transit_legs {
  lb_transit_legs(timetable const&, query const&,
                  lb_transit_legs_state&);

  timetable const& tt_;
  query const& q_;
  lb_transit_legs_state& state_;
  std::uint8_t k_;
};

}  // namespace nigiri::routing