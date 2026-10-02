#pragma once

#include <cstdint>
#include <algorithm>
#include <functional>
#include <ranges>
#include <span>
#include <vector>

#include "cista/reflection/comparable.h"

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

struct virt {
  location_idx_t location_;
  std::vector<transfer_rule_side_idx> rules_{};
};

struct hub_coverage {
  void mark_slow(location_idx_t const from, location_idx_t const to) {
    slow_from_.insert(from);
    slow_to_.insert(to);
  }
  hash_set<location_idx_t> slow_from_, slow_to_;
};

struct hub_lists {
  template <typename In, typename Out>
  void add(In&& in, Out&& out, duration_t const d) {
    if (std::ranges::empty(in) || std::ranges::empty(out)) {
      return;
    }
    in_.emplace_back(in);
    out_.emplace_back(out);
    time_.push_back(d);
  }

  vecvec<hub_idx_t, location_idx_t> in_, out_;
  vector_map<hub_idx_t, duration_t> time_;
};

void write_hubs(timetable&, hub_lists const&);

struct rule_transfers {
  void add_footpath(location_idx_t const from,
                    location_idx_t const to,
                    duration_t const d) {
    footpaths_[from].emplace_back(to, d);
  }

  hub_lists hubs_;
  mutable_fws_multimap<location_idx_t, footpath> footpaths_;
};

// Connects every pair of from x to at duration d, except pairs is_owned leaves
// to someone else:
//
// - unrestricted hub: !slow_from -> all
// - restricted hub: slow_from -> !slow_to
// - footpaths: slow_from -> slow_to
//
// A hub with X sources and Y targets is written as only footpaths instead if:
// X*Y (minus self-pairs) <= X+Y.
template <typename From, typename To, typename IsOwned, typename Out>
void add_hubs_or_footpaths(From const& from,
                           To const& to,
                           duration_t const d,
                           hub_coverage const& coverage,
                           IsOwned&& is_owned,
                           Out& out) {
  auto const add_footpaths = [&](auto&& in, auto&& targets) {
    for (auto const f : in) {
      for (auto const t : targets) {
        if (f != t && !is_owned(f, t)) {
          out.add_footpath(f, t, d);
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
      out.hubs_.add(in, targets, d);
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

void add_rule_hubs(
    timetable const&,
    hash_map<transfer_pair, transfer_rule_idx_t> const& most_specific,
    rule_transfers&);

void store_rule_lookups(timetable&, interval<transfer_rule_idx_t> rules);

location_idx_t get_or_create_virt(
    timetable&,
    hash_map<virt_key, virt>& virts,
    location_idx_t base,
    std::vector<transfer_rule_side_idx> const& sig);

void store_virt_lookups_to_tt(timetable&,
                              hash_map<virt_key, virt> const& virts);

rule_transfers get_rule_transfers(timetable const&);

}  // namespace nigiri::loader
