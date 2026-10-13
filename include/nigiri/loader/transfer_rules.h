#pragma once

#include <cassert>
#include <cstdint>
#include <algorithm>
#include <functional>
#include <ranges>
#include <vector>

#include "cista/reflection/comparable.h"

#include "utl/helpers/algorithm.h"

#include "nigiri/footpath.h"
#include "nigiri/timetable.h"
#include "nigiri/transfer_rule_sides.h"
#include "nigiri/types.h"

namespace nigiri::loader {

struct transfer_pair {
  CISTA_COMPARABLE()
  location_idx_t from_{location_idx_t::invalid()};
  location_idx_t to_{location_idx_t::invalid()};
};

hash_map<transfer_pair, transfer_rule_idx_t> get_most_specific(
    timetable const&, interval<transfer_rule_idx_t> rules);

// One footpath per stop pair a rule wins, with the rule's duration.
void add_rule_footpaths(
    timetable const&,
    hash_map<transfer_pair, transfer_rule_idx_t> const& most_specific,
    mutable_fws_multimap<location_idx_t, footpath>& footpaths);

void store_rule_lookups(timetable&, interval<transfer_rule_idx_t> rules);

location_idx_t get_or_create_virt(
    timetable&,
    hash_map<pair<location_idx_t, std::vector<transfer_rule_side_idx>>,
             location_idx_t>& virts,
    location_idx_t base,
    std::vector<transfer_rule_side_idx> const& sig);

}  // namespace nigiri::loader
