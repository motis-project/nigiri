#pragma once

#include "cista/reflection/comparable.h"

#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::loader {

struct transfer_pair {
  CISTA_COMPARABLE()
  location_idx_t from_{location_idx_t::invalid()};
  location_idx_t to_{location_idx_t::invalid()};
};

transfer_rule_specificity_t get_specificity(
    timetable const&,
    location_idx_t from_stop,
    location_idx_t to_stop,
    route_id_idx_t from_route = route_id_idx_t::invalid(),
    route_id_idx_t to_route = route_id_idx_t::invalid(),
    trip_idx_t from_trip = trip_idx_t::invalid(),
    trip_idx_t to_trip = trip_idx_t::invalid());

void store_rule_lookups(timetable&, interval<transfer_rule_idx_t> rules);

}  // namespace nigiri::loader
