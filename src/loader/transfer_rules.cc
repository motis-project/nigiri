#include "nigiri/loader/transfer_rules.h"

#include <cassert>
#include <algorithm>
#include <vector>

#include "utl/equal_ranges_linear.h"
#include "utl/erase_duplicates.h"
#include "utl/get_or_create.h"
#include "utl/helpers/algorithm.h"
#include "utl/lookup.h"

#include "nigiri/loader/register.h"
#include "nigiri/common/merge_sorted.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"

namespace nigiri::loader {

void write_hubs(timetable& tt, hub_lists const& hubs) {
  constexpr auto const p = kDefaultProfile;
  auto& loc = tt.locations_;
  for (auto h = hub_idx_t{0U}; h != hub_idx_t{hubs.time_.size()}; ++h) {
    loc.hub_in_[p].emplace_back(hubs.in_[h]);
    loc.hub_out_[p].emplace_back(hubs.out_[h]);
    loc.hub_time_[p].push_back(hubs.time_[h]);
    utl::sort(loc.hub_in_[p].back());
    utl::sort(loc.hub_out_[p].back());

    // l in in and out: routing relaxes l -> l, must not beat l's own change.
    assert(utl::all_of(loc.hub_in_[p].back(), [&](location_idx_t const l) {
      auto const out = loc.hub_out_[p].back();
      return !std::binary_search(begin(out), end(out), l) ||
             to_fp_duration(loc.transfer_time_[l]) <= hubs.time_[h];
    }));
  }

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
    timetable const& tt,
    hash_map<transfer_pair, transfer_rule_idx_t> const& most_specific,
    rule_transfers& transfers) {
  auto const& rules = tt.transfer_rules_.rules_;

  // Per rule: the from and to locations of the pairs it wins between
  // different base locations.
  struct cross_product {
    hash_set<location_idx_t> from_, to_;
  };
  auto cross_products = hash_map<transfer_rule_idx_t, cross_product>{};
  for (auto const& [from_to, rule] : most_specific) {
    if (tt.base(from_to.from_) != tt.base(from_to.to_)) {
      auto& product = cross_products[rule];
      product.from_.insert(from_to.from_);
      product.to_.insert(from_to.to_);
    }
  }

  for (auto const& [rule_idx, x] : cross_products) {
    auto const& [from_locations, to_locations] = x;

    auto const d = rules[rule_idx].duration_;

    // This rule bans the transfer -> add_rule_footpaths.
    if (d == footpath::kMaxDuration) {
      continue;
    }

    // Mark slow from+to so they will be excluded from the hub.
    auto coverage = hub_coverage{};
    for (auto const from : from_locations) {
      for (auto const to : to_locations) {
        assert(
            d == duration_t{0} || tt.base(from) == tt.base(to) ||
            most_specific.contains(transfer_pair{tt.base(from), tt.base(to)}));

        auto is_slower = false;
        if (from == to) {
          is_slower = to_fp_duration(tt.locations_.transfer_time_[from]) > d;
        } else {
          auto const winner =
              utl::lookup(most_specific, transfer_pair{from, to});
          if (!winner.has_value()) {
            assert(false && "no winner for a pair of its cross product");
            coverage.mark_slow(from, to);
            continue;
          }
          is_slower = *winner != rule_idx && rules[*winner].duration_ > d;
        }
        if (is_slower) {
          coverage.mark_slow(from, to);
        }
      }
    }

    // Pairs won by another rule are written by that rule.
    add_hubs_or_footpaths(
        from_locations, to_locations, d, coverage,
        [&](location_idx_t const from, location_idx_t const to) {
          return utl::lookup(most_specific, transfer_pair{from, to}) !=
                 rule_idx;
        },
        transfers);
  }
}

void add_stop_hubs(
    timetable const& tt,
    hash_map<transfer_pair, transfer_rule_idx_t> const& most_specific,
    std::span<location_idx_t const> virts,
    rule_transfers& transfers) {
  auto const& loc = tt.locations_;
  auto const& rules = tt.transfer_rules_.rules_;

  // For every most specific rule:
  // mark all from+to within the same base (including base<->virt) that are
  // slower than the base transfer time itself. Those from+to cannot be
  // connected via hubs.
  auto coverage = hub_coverage{};
  for (auto const& [from_to, rule] : most_specific) {
    auto const from_base = tt.base(from_to.from_);
    auto const to_base = tt.base(from_to.to_);
    if (from_base != to_base) {
      continue;
    }

    auto const rule_duration = rules[rule].duration_;
    auto const self_transfer_duration =
        to_fp_duration(loc.transfer_time_[from_base]);

    if (rule_duration > self_transfer_duration) {
      coverage.mark_slow(from_to.from_, from_to.to_);
    }
  }

  auto bases = std::vector<location_idx_t>{};
  for (auto const virt : virts) {
    bases.push_back(tt.base(virt));
  }
  utl::erase_duplicates(bases);

  auto members = std::vector<location_idx_t>{};
  for (auto const base : bases) {
    if (loc.transfer_time_[base] == kNoTransferAllowed) {
      continue;
    }
    auto const d = to_fp_duration(loc.transfer_time_[base]);

    // Changes slower than the base: must not reach itself through the hub.
    members.assign({base});
    loc.for_each_virt(base, [&](location_idx_t const v) {
      members.push_back(v);
      if (to_fp_duration(loc.transfer_time_[v]) > d) {
        coverage.mark_slow(v, v);
      }
    });

    // Pairs a rule states with another duration -> add_rule_footpaths.
    auto const is_owned = [&](location_idx_t const from,
                              location_idx_t const to) {
      auto const winner = utl::lookup(most_specific, transfer_pair{from, to});
      return winner.has_value() && rules[*winner].duration_ != d;
    };
    add_hubs_or_footpaths(members, members, d, coverage, is_owned, transfers);
  }
}

// Rules no hub speaks for: within a base with another duration than the
// base's transfer time, and forbidden transfers between bases.
void add_rule_footpaths(
    timetable const& tt,
    hash_map<transfer_pair, transfer_rule_idx_t> const& most_specific,
    rule_transfers& transfers) {
  for (auto const& [from_to, rule] : most_specific) {
    auto const d = tt.transfer_rules_.rules_[rule].duration_;
    auto const from_base = tt.base(from_to.from_);
    auto const to_base = tt.base(from_to.to_);
    auto const is_hub_duration =
        from_base == to_base
            ? d == to_fp_duration(tt.locations_.transfer_time_[from_base])
            : d != footpath::kMaxDuration;
    if (!is_hub_duration) {
      transfers.add_footpath(from_to.from_, from_to.to_, d);
    }
  }
}

location_idx_t get_or_create_virt(
    timetable& tt,
    hash_map<virt_key, virt>& virts,
    location_idx_t const base,
    std::vector<transfer_rule_side_idx> const& sig) {
  auto key = get_virt_key(tt, sig, base);
  auto const transfer_time = to_fp_duration(key.transfer_time_);
  auto& entry = utl::get_or_create(virts, std::move(key), [&]() {
    auto l = location{};
    l.src_ = tt.locations_.src_[base];
    l.pos_ = tt.locations_.coordinates_[base];
    l.type_ = location_type::kVirt;
    l.parent_ = base;
    l.transfer_time_ = transfer_time;

    auto const v = register_location(tt, l);
    tt.locations_.children_[base].push_back(v);
    return virt{.location_ = v};
  });
  entry.rules_.insert(end(entry.rules_), begin(sig), end(sig));
  return entry.location_;
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

void store_virt_lookups_to_tt(timetable& tt,
                              hash_map<virt_key, virt> const& virts) {
  auto rule_virts = std::vector<pair<transfer_rule_side_idx, location_idx_t>>{};
  auto virt_rules = std::vector<pair<location_idx_t, transfer_rule_side_idx>>{};
  for (auto const& [_, v] : virts) {
    for (auto const rule : v.rules_) {
      rule_virts.push_back({rule, v.location_});
      virt_rules.push_back({v.location_, rule});
    }
  }
  merge_sorted(tt.transfer_rules_.rule_virts_, rule_virts);
  merge_sorted(tt.transfer_rules_.virt_rules_, virt_rules);
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

rule_transfers get_rule_transfers(timetable const& tt) {
  auto const& loc = tt.locations_;

  auto virts = vector_map<source_idx_t, std::vector<location_idx_t>>{};
  virts.resize(tt.n_sources());
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    if (loc.is_virt(l)) {
      virts[loc.src_[l]].push_back(l);
    }
  }

  auto transfers = rule_transfers{};

  auto const& rules = tt.transfer_rules_.rules_;
  utl::equal_ranges_linear(
      rules,
      [](transfer_rule const& a, transfer_rule const& b) {
        return a.src_ == b.src_;
      },
      [&](auto const from, auto const to) {
        auto const feed_rules = interval{
            transfer_rule_idx_t{static_cast<std::size_t>(from - begin(rules))},
            transfer_rule_idx_t{static_cast<std::size_t>(to - begin(rules))}};
        auto const& feed_virts = virts[from->src_];

        auto const most_specific = get_most_specific(tt, feed_rules);
        add_rule_hubs(tt, most_specific, transfers);
        add_stop_hubs(tt, most_specific, feed_virts, transfers);
        add_rule_footpaths(tt, most_specific, transfers);
      });

  while (transfers.footpaths_.size() < tt.n_locations()) {
    transfers.footpaths_.emplace_back();
  }

  log(log_lvl::info, "loader.transfer_rules", "{} rule and stop hubs",
      transfers.hubs_.time_.size());

  return transfers;
}

}  // namespace nigiri::loader
