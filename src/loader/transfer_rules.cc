#include "nigiri/loader/transfer_rules.h"

#include <cassert>
#include <algorithm>
#include <vector>

#include "utl/erase_duplicates.h"
#include "utl/get_or_create.h"
#include "utl/helpers/algorithm.h"
#include "utl/lookup.h"

#include "nigiri/loader/register.h"
#include "nigiri/common/merge_sorted.h"
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

location_idx_t get_or_create_virt(
    timetable& tt,
    hash_map<pair<location_idx_t, std::vector<transfer_rule_side_idx>>,
             location_idx_t>& virts,
    location_idx_t const base,
    std::vector<transfer_rule_side_idx> const& sig) {
  auto const v = utl::get_or_create(virts, pair{base, sig}, [&]() {
    auto l = location{};
    l.src_ = tt.locations_.src_[base];
    l.pos_ = tt.locations_.coordinates_[base];
    l.type_ = location_type::kVirt;
    l.parent_ = base;
    l.transfer_time_ =
        to_fp_duration(to_transfer_time(get_transfer_time(tt, sig, base)));

    auto const idx = register_location(tt, l);
    tt.locations_.children_[base].push_back(idx);
    return idx;
  });
  for (auto const side : sig) {
    tt.transfer_rules_.rule_virts_.push_back({side, v});
    tt.transfer_rules_.virt_rules_.push_back({v, side});
  }
  return v;
}

void store_rule_lookups(timetable& tt,
                        interval<transfer_rule_idx_t> const rules) {
  auto& tr = tt.transfer_rules_;
  auto trip_rules = std::vector<pair<trip_idx_t, transfer_rule_side_idx>>{};
  auto route_rules =
      std::vector<pair<route_id_idx_t, transfer_rule_side_idx>>{};
  auto stop_rules = std::vector<pair<location_idx_t, transfer_rule_side_idx>>{};
  for (auto const rule : rules) {
    auto const& r = tr.rules_[rule];
    for (auto const ev : {event_type::kArr, event_type::kDep}) {
      auto const s = transfer_rule_side_idx{rule, ev};
      if (r.trip(ev) != trip_idx_t::invalid()) {
        trip_rules.push_back({r.trip(ev), s});
      } else if (r.route(ev) != route_id_idx_t::invalid()) {
        route_rules.push_back({r.route(ev), s});
      } else {
        stop_rules.push_back({r.stop(ev), s});
      }
    }
  }

  merge_sorted(tr.trip_rules_, trip_rules);
  merge_sorted(tr.route_rules_, route_rules);
  merge_sorted(tr.stop_rules_, stop_rules);
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
    if (tt.transfer_rules_.rules_[rule].is_global()) {
      // both ends at the same station
      auto const root = [&](location_idx_t const l) {
        return tt.locations_.get_root_idx(l);
      };
      auto to_by_station =
          hash_map<location_idx_t, std::vector<location_idx_t>>{};
      for_each_side_location(tt, to_side, [&](location_idx_t const to) {
        to_by_station[root(to)].push_back(to);
      });
      for_each_side_location(tt, from_side, [&](location_idx_t const from) {
        if (auto const it = to_by_station.find(root(from));
            it != end(to_by_station)) {
          for (auto const to : it->second) {
            add(from, to);
          }
        }
      });
      continue;
    }
    for_each_side_location(tt, from_side, [&](location_idx_t const from) {
      for_each_side_location(tt, to_side,
                             [&](location_idx_t const to) { add(from, to); });
    });
  }
  return most_specific;
}

}  // namespace nigiri::loader
