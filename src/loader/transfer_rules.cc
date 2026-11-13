#include "nigiri/loader/transfer_rules.h"

#include <vector>

#include "nigiri/common/merge_sorted.h"
#include "nigiri/timetable.h"

namespace nigiri::loader {

void store_rule_lookups(timetable& tt,
                        interval<transfer_rule_idx_t> const rules) {
  auto& tr = tt.transfer_rules_;
  auto trip_rules = qualified_sides_t<trip_idx_t>{};
  auto route_rules = qualified_sides_t<pair<source_idx_t, route_id_idx_t>>{};
  auto stop_rules = std::vector<pair<location_idx_t, transfer_rule_side_idx>>{};
  for (auto const rule : rules) {
    auto const& r = tr.rules_[rule];
    for (auto const ev : {event_type::kArr, event_type::kDep}) {
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
