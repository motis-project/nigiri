#include "nigiri/loader/transfer_rules.h"

#include <vector>

#include "nigiri/common/merge_sorted.h"
#include "nigiri/timetable.h"

namespace nigiri::loader {

enum class specificity : std::uint8_t {
  kStopsOnly,
  kOneRoute,
  kBothRoutes,
  kOneTrip,
  kTripAndRoute,
  kBothTrips
};

transfer_rule_specificity_t get_specificity(timetable const& tt,
                                            location_idx_t const from_stop,
                                            location_idx_t const to_stop,
                                            route_id_idx_t const from_route,
                                            route_id_idx_t const to_route,
                                            trip_idx_t const from_trip,
                                            trip_idx_t const to_trip) {
  auto const has_from_trip = from_trip != trip_idx_t::invalid();
  auto const has_to_trip = to_trip != trip_idx_t::invalid();
  auto const has_from_route = from_route != route_id_idx_t::invalid();
  auto const has_to_route = to_route != route_id_idx_t::invalid();
  auto const level =
      has_from_trip && has_to_trip ? specificity::kBothTrips
      : (has_from_trip && has_to_route) || (has_from_route && has_to_trip)
          ? specificity::kTripAndRoute
      : has_from_trip || has_to_trip   ? specificity::kOneTrip
      : has_from_route && has_to_route ? specificity::kBothRoutes
      : has_from_route || has_to_route ? specificity::kOneRoute
                                       : specificity::kStopsOnly;
  auto const has_parent = [&](location_idx_t const stop) {
    return tt.locations_.parents_[stop] != location_idx_t::invalid() ? 1U : 0U;
  };
  return static_cast<transfer_rule_specificity_t>(
      (static_cast<unsigned>(level) << 2U) |
      (has_parent(from_stop) + has_parent(to_stop)));
}

void store_rule_lookups(timetable& tt,
                        interval<transfer_rule_idx_t> const rules) {
  auto& tr = tt.transfer_rules_;
  auto trip_rules = qualified_sides_t<trip_idx_t>{};
  auto route_rules = qualified_sides_t<pair<source_idx_t, route_id_idx_t>>{};
  auto stop_rules = std::vector<pair<location_idx_t, transfer_rule_side_idx>>{};
  for (auto const rule : rules) {
    auto const& r = tr.rules_[rule];
    for (auto const ev : {event_type::kArr, event_type::kDep}) {
      if (r.is_listed(ev)) {
        continue;  // listed by the loader
      }
      auto const s = transfer_rule_side_idx{rule, ev};
      auto const root = tt.locations_.get_root_idx(r.stop(ev));
      if (r.trip(ev) != trip_idx_t::invalid()) {
        trip_rules.push_back({r.trip(ev), root, s});
      } else if (r.route(ev) != route_id_idx_t::invalid()) {
        route_rules.push_back({{r.src_, r.route(ev)}, root, s});
      } else {
        stop_rules.push_back({r.stop(ev), s});
      }
    }
  }

  merge_sorted(tr.trip_rules_, trip_rules);
  merge_sorted(tr.route_rules_, route_rules);
  merge_sorted(tr.stop_rules_, stop_rules);
}

}  // namespace nigiri::loader
