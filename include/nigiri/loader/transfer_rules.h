#pragma once

#include <cassert>
#include <cstdint>
#include <algorithm>
#include <functional>
#include <ranges>
#include <vector>

#include "cista/reflection/comparable.h"

#include "utl/helpers/algorithm.h"

#include "nigiri/footpath.h"
#include "nigiri/timetable.h"
#include "nigiri/transfer_rule_sides.h"
#include "nigiri/types.h"

namespace nigiri::loader {

struct transfer_pair {
  CISTA_COMPARABLE()
  location_idx_t from_{location_idx_t::invalid()};
  location_idx_t to_{location_idx_t::invalid()};
};

struct hub_coverage {
  void mark_slow(location_idx_t const from, location_idx_t const to) {
    slow_from_.insert(from);
    slow_to_.insert(to);
  }
  hash_set<location_idx_t> slow_from_, slow_to_;
};

template <typename In, typename Out>
void add_hub(timetable& tt, In&& in, Out&& out, duration_t const d) {
  if (std::ranges::empty(in) || std::ranges::empty(out)) {
    return;
  }

  auto& loc = tt.locations_;
  loc.hub_in_[kDefaultProfile].emplace_back(in);
  loc.hub_out_[kDefaultProfile].emplace_back(out);
  loc.hub_time_[kDefaultProfile].push_back(d);
  utl::sort(loc.hub_in_[kDefaultProfile].back());
  utl::sort(loc.hub_out_[kDefaultProfile].back());

  // l in in and out: routing relaxes l -> l, must not beat l's own change.
  assert(utl::all_of(
      loc.hub_in_[kDefaultProfile].back(), [&](location_idx_t const l) {
        auto const hub_out = loc.hub_out_[kDefaultProfile].back();
        return !std::binary_search(begin(hub_out), end(hub_out), l) ||
               to_fp_duration(loc.transfer_time_[l]) <= d;
      }));
}

void index_hubs(timetable&);

// Connects every pair of from x to at duration d:
//
// - unrestricted hub: !slow_from -> all
// - restricted hub: slow_from -> !slow_to
// - footpaths: slow_from -> slow_to IF is_footpath_allowed(slow_from, slow_to)
//
// A hub with X sources and Y targets is written as only footpaths instead if:
// X*Y (minus self-pairs) <= X+Y.
template <typename From, typename To, typename IsFootpathAllowed>
void add_hubs_or_footpaths(
    From const& from,
    To const& to,
    duration_t const d,
    hub_coverage const& coverage,
    IsFootpathAllowed&& is_footpath_allowed,
    timetable& tt,
    mutable_fws_multimap<location_idx_t, footpath>& footpaths) {
  auto const add_footpaths = [&](auto&& in, auto&& targets) {
    for (auto const f : in) {
      for (auto const t : targets) {
        if (f != t && is_footpath_allowed(f, t)) {
          footpaths[f].emplace_back(t, d);
        }
      }
    }
  };

  auto const add_hub_or_footpaths = [&](auto&& in, auto&& targets) {
    auto const n_in = static_cast<std::size_t>(std::ranges::distance(in));
    auto const n_targets =
        static_cast<std::size_t>(std::ranges::distance(targets));
    auto const n_entries = n_in + n_targets;
    auto n_pairs = n_in * n_targets;

    // Don't count self-pairs.
    if (n_pairs > n_entries &&
        n_pairs <= n_entries + std::min(n_in, n_targets)) {
      n_pairs -= static_cast<std::size_t>(
          std::ranges::count_if(in, [&](location_idx_t const l) {
            return std::ranges::find(targets, l) != std::ranges::end(targets);
          }));
    }

    if (n_pairs <= n_entries) {
      add_footpaths(in, targets);
    } else {
      add_hub(tt, in, targets, d);
    }
  };

  auto const is_slow_from = [&](location_idx_t const l) {
    return coverage.slow_from_.contains(l);
  };
  auto const is_slow_to = [&](location_idx_t const l) {
    return coverage.slow_to_.contains(l);
  };
  auto from_slow = from | std::views::filter(is_slow_from);
  auto from_other = from | std::views::filter(std::not_fn(is_slow_from));
  auto to_slow = to | std::views::filter(is_slow_to);
  auto to_other = to | std::views::filter(std::not_fn(is_slow_to));

  add_hub_or_footpaths(from_other, to);
  add_hub_or_footpaths(from_slow, to_other);
  add_footpaths(from_slow, to_slow);
}

hash_map<transfer_pair, transfer_rule_idx_t> get_most_specific(
    timetable const&, interval<transfer_rule_idx_t> rules);

void add_rule_hubs(
    timetable&,
    hash_map<transfer_pair, transfer_rule_idx_t> const& most_specific,
    mutable_fws_multimap<location_idx_t, footpath>& footpaths);

void add_stop_hubs(
    timetable&,
    interval<transfer_rule_idx_t> rules,
    hash_map<transfer_pair, transfer_rule_idx_t> const& most_specific,
    mutable_fws_multimap<location_idx_t, footpath>& footpaths);

void store_rule_lookups(timetable&, interval<transfer_rule_idx_t> rules);

location_idx_t get_or_create_virt(
    timetable&,
    hash_map<virt_key, location_idx_t>& virts,
    location_idx_t base,
    std::vector<transfer_rule_side_idx> const& sig);

}  // namespace nigiri::loader
