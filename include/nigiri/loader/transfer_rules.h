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

void store_rule_lookups(timetable&, interval<transfer_rule_idx_t> rules);

}  // namespace nigiri::loader
