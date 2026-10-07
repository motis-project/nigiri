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

void index_hubs(timetable& tt) {
  constexpr auto const p = kDefaultProfile;
  auto& loc = tt.locations_;
  auto const index = [&](auto const& hub_locations, auto& by_location) {
    auto tmp = mutable_fws_multimap<location_idx_t, hub_idx_t>{};
    for (auto h = hub_idx_t{0U}; h != hub_idx_t{hub_locations.size()}; ++h) {
      for (auto const l : hub_locations[h]) {
        tmp[l].push_back(h);
      }
    }
    by_location.clear();
    for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
      by_location.emplace_back(tmp[l]);
    }
  };
  index(loc.hub_in_[p], loc.hub_in_by_loc_[p]);
  index(loc.hub_out_[p], loc.hub_out_by_loc_[p]);
}

void add_rule_hubs(
    timetable& tt,
    hash_map<transfer_pair, transfer_rule_idx_t> const& most_specific,
    mutable_fws_multimap<location_idx_t, footpath>& footpaths) {
  auto const& rules = tt.transfer_rules_.rules_;

  // Per rule: the from and to locations of the pairs it wins between
  // different base locations.
  struct cross_product {
    hash_set<location_idx_t> from_, to_;
  };
  auto cross_products = hash_map<transfer_rule_idx_t, cross_product>{};
  for (auto const& [from_to, rule] : most_specific) {
    // Same base location, different transfer time
    // -> not covered by rule hub => build footpath
    auto const base = tt.base(from_to.from_);
    if (base == tt.base(from_to.to_)) {
      auto const d = rules[rule].duration_;
      if (d != to_fp_duration(tt.locations_.transfer_time_[base])) {
        footpaths[from_to.from_].emplace_back(from_to.to_, d);
      }
      continue;
    }

    // Forbidden transfer -> not covered by rule hub => build footpath
    if (rules[rule].duration_ == footpath::kMaxDuration) {
      footpaths[from_to.from_].emplace_back(from_to.to_,
                                            footpath::kMaxDuration);
      continue;
    }

    auto& product = cross_products[rule];
    product.from_.insert(from_to.from_);
    product.to_.insert(from_to.to_);
  }

  for (auto const& [rule_idx, x] : cross_products) {
    auto const& [from_locations, to_locations] = x;
    auto const d = rules[rule_idx].duration_;

    // Mark slow from+to so they will be excluded from the hub.
    auto coverage = hub_coverage{};
    for (auto const from : from_locations) {
      for (auto const to : to_locations) {
        auto const winner = utl::lookup(most_specific, transfer_pair{from, to});
        assert((from == to || winner.has_value()) &&
               "no winner for a pair of its cross product");
        auto const is_slower =
            from == to ? to_fp_duration(tt.locations_.transfer_time_[from]) > d
                       : !winner.has_value() || (*winner != rule_idx &&
                                                 rules[*winner].duration_ > d);
        if (is_slower) {
          coverage.mark_slow(from, to);
        }
      }
    }

    // Pairs won by another rule are written by that rule.
    add_hubs_or_footpaths(
        from_locations, to_locations, d, coverage,
        [&](location_idx_t const from, location_idx_t const to) {
          return utl::lookup(most_specific, transfer_pair{from, to}) ==
                 rule_idx;
        },
        tt, footpaths);
  }
}

location_idx_t get_or_create_virt(
    timetable& tt,
    hash_map<virt_key, location_idx_t>& virts,
    location_idx_t const base,
    std::vector<transfer_rule_side_idx> const& sig) {
  auto key = get_virt_key(tt, sig, base);
  auto const transfer_time = to_fp_duration(key.transfer_time_);
  auto const v = utl::get_or_create(virts, std::move(key), [&]() {
    auto l = location{};
    l.src_ = tt.locations_.src_[base];
    l.pos_ = tt.locations_.coordinates_[base];
    l.type_ = location_type::kVirt;
    l.parent_ = base;
    l.transfer_time_ = transfer_time;

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
    for (auto const is_from : {true, false}) {
      auto const s = transfer_rule_side_idx{rule, is_from};
      if (r.trip(is_from) != trip_idx_t::invalid()) {
        trip_rules.push_back({r.trip(is_from), s});
      } else if (r.route(is_from) != route_id_idx_t::invalid()) {
        route_rules.push_back({r.route(is_from), s});
      } else {
        stop_rules.push_back({r.stop(is_from), s});
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
    auto const from_side = transfer_rule_side_idx{rule, true};
    auto const to_side = transfer_rule_side_idx{rule, false};
    for_each_side_location(tt, from_side, [&](location_idx_t const from) {
      for_each_side_location(tt, to_side, [&](location_idx_t const to) {
        if (from == to) {
          return;
        }

        auto const [it, is_new] =
            most_specific.emplace(transfer_pair{from, to}, rule);
        if (!is_new) {
          it->second = get_more_specific(tt, it->second, rule);
        }
      });
    });
  }
  return most_specific;
}

}  // namespace nigiri::loader
