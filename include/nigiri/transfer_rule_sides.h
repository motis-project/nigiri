#pragma once

#include <algorithm>
#include <optional>
#include <span>
#include <tuple>
#include <vector>

#include "cista/reflection/comparable.h"

#include "nigiri/for_each_meta.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri {

inline transfer_rule_idx_t get_more_specific(timetable const& tt,
                                             transfer_rule_idx_t const a,
                                             transfer_rule_idx_t const b) {
  // Compare with rule index as tie breaker
  // -> deterministic ordering accross imports
  // -> loader and real-time updates choose the same rule
  auto const& rules = tt.transfer_rules_.rules_;
  return std::tie(rules[a].specificity_, a) < std::tie(rules[b].specificity_, b)
             ? b
             : a;
}

void add_trip_rules(timetable const&,
                    std::span<trip_idx_t const> trips,
                    std::vector<transfer_rule_side_idx>&);

void get_signature(timetable const&,
                   std::span<transfer_rule_side_idx const> arriving,
                   std::span<transfer_rule_side_idx const> departing,
                   bool is_handover,
                   location_idx_t base,
                   std::vector<transfer_rule_side_idx>& sig);

// nullopt: none of the trips has rules.
std::optional<std::vector<transfer_rule_side_idx>> get_change_signature(
    timetable const&,
    std::span<trip_idx_t const> arriving,
    std::span<trip_idx_t const> departing,
    bool is_handover,
    location_idx_t base);

inline bool is_applicable(timetable const& tt,
                          transfer_rule_side_idx const side,
                          std::span<transfer_rule_side_idx const> sig,
                          location_idx_t const base) {
  auto const& r = tt.transfer_rules_.rules_[side.rule()];
  return r.is_qualified(side.is_from())
             ? std::ranges::binary_search(sig, side)
             : tt.locations_.is_self_or_parent(r.stop(side.is_from()), base);
}

std::vector<transfer_rule_side> get_transfer_rule_sides(
    timetable const&,
    std::span<transfer_rule_side_idx const>,
    location_idx_t base);

virt_key get_virt_key(timetable const&,
                      std::span<transfer_rule_side_idx const> sig,
                      location_idx_t base);

template <typename Fn>
void for_each_side_location(timetable const& tt,
                            transfer_rule_side_idx const side,
                            Fn&& fn) {
  auto const& r = tt.transfer_rules_.rules_[side.rule()];
  if (r.is_qualified(side.is_from())) {
    // Qualified rules map to virt locations.
    for (auto const l : values_of(tt.transfer_rules_.rule_virts_, side)) {
      fn(l);
    }
  } else {
    // Unqualified rules map to the stop and all descendants.
    routing::for_each_meta(tt, routing::location_match_mode::kOnlyChildren,
                           r.stop(side.is_from()), fn);
  }
}

}  // namespace nigiri
