#include "nigiri/loader/transfer_rules.h"

#include <cassert>
#include <algorithm>
#include <vector>

#include "utl/concat.h"
#include "utl/equal_ranges_linear.h"
#include "utl/erase_duplicates.h"
#include "utl/get_or_create.h"
#include "utl/helpers/algorithm.h"

#include "nigiri/loader/register.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"

namespace nigiri::loader {

bool in_stop_hub(timetable const& tt,
                 hub_coverage const& stop_coverage,
                 location_idx_t const from,
                 location_idx_t const to) {
  auto const base = tt.locations_.get_base_idx(from);
  assert(base == tt.locations_.get_base_idx(to));
  auto const is_slow = from != base && tt.locations_.transfer_time_[from] >
                                           tt.locations_.transfer_time_[base];
  return !is_slow && stop_coverage.in_hub(from, to);
}

void write_hubs(timetable& tt, hub_lists const& hubs) {
  auto& loc = tt.locations_;
  for (auto h = hub_idx_t{0U}; h != hub_idx_t{hubs.time_.size()}; ++h) {
    loc.hub_in_[kDefaultProfile].emplace_back(hubs.in_[h]);
    loc.hub_out_[kDefaultProfile].emplace_back(hubs.out_[h]);
    loc.hub_time_[kDefaultProfile].push_back(hubs.time_[h]);
    utl::sort(loc.hub_in_[kDefaultProfile].back());
    utl::sort(loc.hub_out_[kDefaultProfile].back());
  }
}

rule_hubs get_rule_hubs(
    timetable const& tt,
    hash_map<transfer_pair, transfer_rule_idx_t> const& most_specific) {
  auto const& rules = tt.transfer_rules_.rules_;

  // Per rule: the from and to locations of the pairs it wins between
  // different base locations.
  struct cross_product {
    hash_set<location_idx_t> from_, to_;
  };
  auto cross_products = hash_map<transfer_rule_idx_t, cross_product>{};
  for (auto const& [from_to, rule] : most_specific) {
    if (tt.locations_.get_base_idx(from_to.from_) !=
        tt.locations_.get_base_idx(from_to.to_)) {
      auto& product = cross_products[rule];
      product.from_.insert(from_to.from_);
      product.to_.insert(from_to.to_);
    }
  }

  auto hubs = rule_hubs{};
  for (auto const& [rule_idx, x] : cross_products) {
    auto const& [from_locations, to_locations] = x;

    auto const d = rules[rule_idx].duration_;

    // This rule bans the transfer -> no hub creation.
    if (d == footpath::kMaxDuration) {
      continue;
    }

    // Having footpaths instead is cheaper -> no hub creation.
    if (from_locations.size() * to_locations.size() <=
        from_locations.size() + to_locations.size()) {
      continue;
    }

    // Mark slow from+to so they will be excluded from the hub.
    auto coverage = hub_coverage{};
    for (auto const from : from_locations) {
      for (auto const to : to_locations) {
        assert(d == duration_t{0} ||
               tt.locations_.get_base_idx(from) ==
                   tt.locations_.get_base_idx(to) ||
               most_specific.contains(
                   transfer_pair{tt.locations_.get_base_idx(from),
                                 tt.locations_.get_base_idx(to)}));

        auto is_slower = false;
        if (from == to) {
          is_slower = to_fp_duration(tt.locations_.transfer_time_[from]) > d;
        } else {
          auto const it = most_specific.find(transfer_pair{from, to});
          if (it == end(most_specific)) {
            assert(false && "no winner for a pair of its cross product");
            coverage.mark_slow(from, to);
            continue;
          }
          is_slower = it->second != rule_idx && rules[it->second].duration_ > d;
        }
        if (is_slower) {
          coverage.mark_slow(from, to);
        }
      }
    }

    // Create hubs and store coverage for this rule.
    for_each_hub(from_locations, to_locations, coverage,
                 [&](std::vector<location_idx_t> const& in,
                     std::vector<location_idx_t> const& out) {
                   hubs.hubs_.add(in, out, d);
                 });
    hubs.coverage_.emplace(rule_idx, std::move(coverage));
  }
  return hubs;
}

stop_hubs get_stop_hubs(
    timetable const& tt,
    hash_map<transfer_pair, transfer_rule_idx_t> const& most_specific,
    std::span<location_idx_t const> virts) {
  auto const& loc = tt.locations_;

  // For every most specific rule:
  // mark all from+to within the same base (including base<->virt) that are
  // slower than the base transfer time itself. Those from+to cannot be
  // connected via hubs and will get direct footpaths.
  auto hubs = stop_hubs{};
  for (auto const& [from_to, rule] : most_specific) {
    auto const from_base = loc.get_base_idx(from_to.from_);
    auto const to_base = loc.get_base_idx(from_to.to_);
    if (from_base != to_base) {
      continue;
    }

    auto const rule_duration = tt.transfer_rules_.rules_[rule].duration_;
    auto const self_transfer_duration =
        to_fp_duration(loc.transfer_time_[from_base]);

    if (rule_duration > self_transfer_duration) {
      hubs.coverage_.mark_slow(from_to.from_, from_to.to_);
    }
  }

  auto bases = std::vector<location_idx_t>{};
  for (auto const virt : virts) {
    bases.push_back(loc.get_base_idx(virt));
  }
  utl::erase_duplicates(bases);

  auto members = std::vector<location_idx_t>{};
  auto sources = std::vector<location_idx_t>{};
  for (auto const base : bases) {
    auto const d = loc.transfer_time_[base];
    if (d == kNoTransferAllowed) {
      continue;
    }
    members.assign({base});
    sources.assign({base});
    loc.for_each_virt(base, [&](location_idx_t const v) {
      members.push_back(v);
      if (loc.transfer_time_[v] <= d) {
        sources.push_back(v);
      }
    });
    for_each_hub(sources, members, hubs.coverage_,
                 [&](std::vector<location_idx_t> const& in,
                     std::vector<location_idx_t> const& out) {
                   hubs.hubs_.add(in, out, duration_t{d});
                 });
  }
  return hubs;
}

void add_rule_footpaths(
    timetable const& tt,
    hash_map<transfer_pair, transfer_rule_idx_t> const& most_specific,
    hub_coverage const& stop_coverage,
    hash_map<transfer_rule_idx_t, hub_coverage> const& rule_coverage,
    mutable_fws_multimap<location_idx_t, footpath>& footpaths) {
  for (auto const& [from_to, rule] : most_specific) {
    auto const d = tt.transfer_rules_.rules_[rule].duration_;
    auto const from_base = tt.locations_.get_base_idx(from_to.from_);
    auto const to_base = tt.locations_.get_base_idx(from_to.to_);
    if (from_base == to_base) {
      if (d == to_fp_duration(tt.locations_.transfer_time_[from_base]) &&
          in_stop_hub(tt, stop_coverage, from_to.from_, from_to.to_)) {
        continue;
      }
    } else if (auto const it = rule_coverage.find(rule);
               it != end(rule_coverage) &&
               it->second.in_hub(from_to.from_, from_to.to_)) {
      // Footpath is already covered by a hub.
      continue;
    }
    footpaths[from_to.from_].emplace_back(from_to.to_, d);
  }
}

void add_virt_footpaths(
    timetable const& tt,
    hash_map<transfer_pair, transfer_rule_idx_t> const& most_specific,
    hub_coverage const& stop_coverage,
    std::span<location_idx_t const> virts,
    mutable_fws_multimap<location_idx_t, footpath>& footpaths) {
  for (auto const virt : virts) {
    auto const base = tt.locations_.get_base_idx(virt);
    auto const d = to_fp_duration(tt.locations_.transfer_time_[base]);
    auto const add_rule_footpath = [&](location_idx_t const from,
                                       location_idx_t const to) {
      if (!most_specific.contains({from, to}) &&
          !in_stop_hub(tt, stop_coverage, from, to)) {
        footpaths[from].emplace_back(to, d);
      }
    };

    add_rule_footpath(virt, base);
    add_rule_footpath(base, virt);
    for (auto const sibling : tt.locations_.children_[base]) {
      if (sibling != virt && tt.locations_.is_virt(sibling)) {
        add_rule_footpath(virt, sibling);
      }
    }
  }
}

template <typename T>
void merge_sorted(vector<T>& v, std::vector<T> added) {
  utl::erase_duplicates(added);
  auto const old_size = static_cast<std::ptrdiff_t>(v.size());
  utl::concat(v, added);
  std::inplace_merge(begin(v), begin(v) + old_size, end(v));
}

location_idx_t get_or_create_virt(
    timetable& tt,
    hash_map<virt_key, new_virt>& virts,
    location_idx_t const base,
    std::vector<transfer_rule_side_idx> const& sig) {
  auto key = get_virt_key(tt, sig, base);
  auto const transfer_time = to_fp_duration(key.transfer_time_);
  auto& virt = utl::get_or_create(virts, std::move(key), [&]() {
    auto l = location{};
    l.src_ = tt.locations_.src_[base];
    l.pos_ = tt.locations_.coordinates_[base];
    l.type_ = location_type::kVirt;
    l.parent_ = base;
    l.transfer_time_ = transfer_time;

    auto const v = register_location(tt, l);
    tt.locations_.children_[base].push_back(v);
    return new_virt{.location_ = v};
  });
  virt.rules_.insert(end(virt.rules_), begin(sig), end(sig));
  return virt.location_;
}

void store_rule_lookups(timetable& tt,
                        interval<transfer_rule_idx_t> const rules) {
  auto& tr = tt.transfer_rules_;
  auto trip_rules = std::vector<pair<trip_idx_t, transfer_rule_side_idx>>{};
  auto route_rules =
      std::vector<pair<route_id_idx_t, transfer_rule_side_idx>>{};
  auto stop_rules = std::vector<pair<location_idx_t, transfer_rule_side_idx>>{};
  for (auto const rule : rules) {
    auto const r = tr.rules_[rule];
    auto const add_side = [&](location_idx_t const stop,
                              route_id_idx_t const route, trip_idx_t const trip,
                              bool const is_from) {
      auto const s = transfer_rule_side_idx{rule, is_from};
      if (trip != trip_idx_t::invalid()) {
        trip_rules.push_back({trip, s});
      } else if (route != route_id_idx_t::invalid()) {
        route_rules.push_back({route, s});
      } else {
        stop_rules.push_back({stop, s});
      }
    };
    add_side(r.from_stop_, r.from_route_, r.from_trip_, true);
    add_side(r.to_stop_, r.to_route_, r.to_trip_, false);
  }

  merge_sorted(tr.trip_rules_, std::move(trip_rules));
  merge_sorted(tr.route_rules_, std::move(route_rules));
  merge_sorted(tr.stop_rules_, std::move(stop_rules));
}

void store_virt_lookups_to_tt(timetable& tt,
                              hash_map<virt_key, new_virt> const& virts) {
  auto rule_virts = std::vector<pair<transfer_rule_side_idx, location_idx_t>>{};
  auto virt_rules = std::vector<pair<location_idx_t, transfer_rule_side_idx>>{};
  for (auto const& [_, virt] : virts) {
    for (auto const rule : virt.rules_) {
      rule_virts.push_back({rule, virt.location_});
      virt_rules.push_back({virt.location_, rule});
    }
  }
  merge_sorted(tt.transfer_rules_.rule_virts_, std::move(rule_virts));
  merge_sorted(tt.transfer_rules_.virt_rules_, std::move(virt_rules));
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
  auto const add_hubs = [&](hub_lists const& hubs) {
    for (auto h = hub_idx_t{0U}; h != hub_idx_t{hubs.time_.size()}; ++h) {
      transfers.hubs_.add(hubs.in_[h], hubs.out_[h], hubs.time_[h]);
    }
  };

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
        auto const [rule_hubs, rule_coverage] =
            get_rule_hubs(tt, most_specific);
        auto const [stop_hubs, stop_coverage] =
            get_stop_hubs(tt, most_specific, feed_virts);
        add_hubs(rule_hubs);
        add_hubs(stop_hubs);
        add_rule_footpaths(tt, most_specific, stop_coverage, rule_coverage,
                           transfers.footpaths_);
        add_virt_footpaths(tt, most_specific, stop_coverage, feed_virts,
                           transfers.footpaths_);
      });

  while (transfers.footpaths_.size() < tt.n_locations()) {
    transfers.footpaths_.emplace_back();
  }

  log(log_lvl::info, "loader.transfer_rules", "{} rule and stop hubs",
      transfers.hubs_.time_.size());

  return transfers;
}

}  // namespace nigiri::loader
