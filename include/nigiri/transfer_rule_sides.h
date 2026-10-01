#pragma once

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

void add_trip_rules(timetable const&,
                    std::span<trip_idx_t const> trips,
                    std::vector<transfer_rule_side_idx>&);

void get_signature(timetable const&,
                   std::span<transfer_rule_side_idx const> arriving,
                   std::span<transfer_rule_side_idx const> departing,
                   bool is_handover,
                   location_idx_t base,
                   std::vector<transfer_rule_side_idx>& sig);

struct transfer_rule_side {
  CISTA_COMPARABLE()
  bool is_from_;
  location_idx_t rule_stop_;
  location_idx_t other_stop_;
  source_idx_t src_;
  route_id_idx_t other_route_;
  trip_idx_t other_trip_;
  duration_t duration_;
  transfer_rule_specificity_t specificity_;
};

std::vector<transfer_rule_side> get_transfer_rule_sides(
    timetable const&, std::span<transfer_rule_side_idx const>);

struct virt_key {
  CISTA_COMPARABLE()
  location_idx_t base_;
  u8_minutes transfer_time_;
  std::vector<transfer_rule_side> sides_;
};

virt_key get_virt_key(timetable const&,
                      std::span<transfer_rule_side_idx const> sig,
                      location_idx_t base);

template <typename Fn>
void for_each_side_location(timetable const& tt,
                            transfer_rule_side_idx const side,
                            Fn&& fn) {
  auto const& r = tt.transfer_rules_.rules_[side.rule()];
  auto const is_qualified =
      side.is_from() ? r.from_qualified() : r.to_qualified();
  if (is_qualified) {
    // Qualified rules map to virt locations.
    for (auto const l : values_of(tt.transfer_rules_.rule_virts_, side)) {
      fn(l);
    }
  } else {
    // Unqualified rules map to all descendant stops.
    auto const stop = side.is_from() ? r.from_stop_ : r.to_stop_;
    fn(stop);
    for (auto const c : tt.locations_.children_[stop]) {
      fn(c);
      for (auto const cc : tt.locations_.children_[c]) {
        fn(cc);
      }
    }
  }
}

}  // namespace nigiri
