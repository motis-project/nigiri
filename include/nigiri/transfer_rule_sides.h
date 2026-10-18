#pragma once

#include <algorithm>
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

bool has_transfer_rules(timetable const&, transport_idx_t);

// the rule stop covers location l: l itself or its parent
inline bool is_rule_stop(timetable const& tt,
                         location_idx_t const rule_stop,
                         location_idx_t const l) {
  return tt.locations_.is_self_or_parent(rule_stop, l);
}

void get_stop_signature(timetable const&,
                        transport_idx_t,
                        stop_idx_t,
                        location_idx_t base,
                        std::vector<transfer_rule_side_idx>& sig);

// The sides of the qualified rules of the trips of a section (merged trips).
void get_section_sides(timetable const&,
                       merged_trips_idx_t,
                       std::vector<transfer_rule_side_idx>& sides);

// Appends the sides of the event type whose rule stop covers the base.
void add_stop_sides(timetable const&,
                    std::span<transfer_rule_side_idx const> section_sides,
                    event_type,
                    location_idx_t base,
                    std::vector<transfer_rule_side_idx>& sig);

// Appends the sides of the event type of the provider's rules at the base's
// station.
void add_provider_sides(timetable const&,
                        provider_idx_t,
                        event_type,
                        location_idx_t base,
                        std::vector<transfer_rule_side_idx>& sig);

inline bool is_applicable(timetable const& tt,
                          transfer_rule_side_idx const side,
                          std::span<transfer_rule_side_idx const> sig,
                          location_idx_t const base) {
  auto const& r = tt.transfer_rules_.rules_[side.rule()];
  return r.is_qualified(side.ev_type())
             ? std::ranges::binary_search(sig, side)
             : tt.locations_.is_self_or_parent(r.stop(side.ev_type()), base);
}

// Transfer time within a location carrying the sides sig: the most specific
// rule applying to both of its sides, else the base's transfer time.
duration_t get_transfer_time(timetable const&,
                             std::span<transfer_rule_side_idx const> sig,
                             location_idx_t base);

transfer_rule_idx_t get_change_rule(
    timetable const&,
    std::span<transfer_rule_side_idx const> from,
    std::span<transfer_rule_side_idx const> to,
    location_idx_t from_base,
    location_idx_t to_base);

template <typename Fn>
void for_each_side_location(timetable const& tt,
                            transfer_rule_side_idx const side,
                            Fn&& fn) {
  auto const& r = tt.transfer_rules_.rules_[side.rule()];
  if (!r.is_qualified(side.ev_type())) {
    // Unqualified rules map to the stop and all descendants.
    routing::for_each_meta(tt, routing::location_match_mode::kOnlyChildren,
                           r.stop(side.ev_type()), fn);
  }
}

}  // namespace nigiri
