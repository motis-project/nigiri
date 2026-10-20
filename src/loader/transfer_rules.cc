#include "nigiri/loader/transfer_rules.h"

#include <cassert>
#include <algorithm>
#include <vector>

#include "utl/erase_duplicates.h"
#include "utl/helpers/algorithm.h"
#include "utl/lookup.h"

#include "nigiri/common/merge_sorted.h"
#include "nigiri/for_each_meta.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"

namespace nigiri::loader {

void add_rule_footpaths(
    timetable const& tt,
    hash_map<transfer_pair, transfer_rule_idx_t> const& most_specific,
    mutable_fws_multimap<location_idx_t, footpath>& footpaths) {
  auto const& rules = tt.transfer_rules_.rules_;
  for (auto const& [from_to, rule] : most_specific) {
    // Same base location, its transfer time: the walks of virtual locations.
    auto const d = rules[rule].duration_;
    auto const base = tt.base(from_to.from_);
    if (base == tt.base(from_to.to_) &&
        d == to_fp_duration(tt.locations_.transfer_time_[base])) {
      continue;
    }
    footpaths[from_to.from_].emplace_back(from_to.to_, d);
  }
}

void store_rule_lookups(timetable& tt,
                        interval<transfer_rule_idx_t> const rules) {
  auto& tr = tt.transfer_rules_;
  auto trip_rules = qualified_sides_t<trip_idx_t>{};
  auto route_rules = qualified_sides_t<route_id_idx_t>{};
  auto stop_rules = std::vector<pair<location_idx_t, transfer_rule_side_idx>>{};
  for (auto const rule : rules) {
    auto const& r = tr.rules_[rule];
    for (auto const ev : {event_type::kArr, event_type::kDep}) {
      auto const s = transfer_rule_side_idx{rule, ev};
      if ((r.listed_ & (ev == event_type::kArr ? 1U : 2U)) != 0U) {
        continue;  // listed by the loader
      }
      auto const root = tt.locations_.get_root_idx(r.stop(ev));
      if (r.trip(ev) != trip_idx_t::invalid()) {
        trip_rules.push_back({r.trip(ev), root, s});
      } else if (r.route(ev) != route_id_idx_t::invalid()) {
        route_rules.push_back({r.route(ev), root, s});
      } else {
        stop_rules.push_back({r.stop(ev), s});
      }
    }
  }

  merge_sorted(tr.trip_rules_, trip_rules);
  merge_sorted(tr.route_rules_, route_rules);
  merge_sorted(tr.stop_rules_, stop_rules);
}

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

hash_map<transfer_pair, transfer_rule_idx_t> get_most_specific(
    timetable const& tt, interval<transfer_rule_idx_t> const rules) {
  auto most_specific = hash_map<transfer_pair, transfer_rule_idx_t>{};
  for (auto const rule : rules) {
    auto const from_side = transfer_rule_side_idx{rule, event_type::kArr};
    auto const to_side = transfer_rule_side_idx{rule, event_type::kDep};
    auto const add = [&](location_idx_t const from, location_idx_t const to) {
      if (from == to) {
        return;
      }
      auto const [it, is_new] =
          most_specific.emplace(transfer_pair{from, to}, rule);
      if (!is_new) {
        it->second = get_more_specific(tt, it->second, rule);
      }
    };
    for_each_side_location(tt, from_side, [&](location_idx_t const from) {
      for_each_side_location(tt, to_side,
                             [&](location_idx_t const to) { add(from, to); });
    });
  }
  return most_specific;
}

}  // namespace nigiri::loader
