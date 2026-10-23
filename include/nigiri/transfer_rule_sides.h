#pragma once

#include <algorithm>
#include <span>
#include <tuple>
#include <vector>

#include "cista/reflection/comparable.h"

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

void get_stop_signature(timetable const&,
                        transport_idx_t,
                        stop_idx_t,
                        location_idx_t base,
                        std::vector<transfer_rule_side_idx>& sig);

struct section_rules {
  bool empty() const { return trips_.empty() && routes_.empty(); }

  std::vector<std::span<qualified_side<trip_idx_t> const>> trips_;
  std::vector<
      std::span<qualified_side<pair<source_idx_t, route_id_idx_t>> const>>
      routes_;
};

void get_section_rules(timetable const&, merged_trips_idx_t, section_rules&);

// Appends the sides of the event type whose rule stop covers the base.
void add_stop_rules(timetable const&,
                    section_rules const&,
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

transfer_rule_idx_t get_change_rule(
    timetable const&,
    std::span<transfer_rule_side_idx const> from,
    std::span<transfer_rule_side_idx const> to,
    location_idx_t from_base,
    location_idx_t to_base);

}  // namespace nigiri
