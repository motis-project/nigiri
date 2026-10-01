#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "cista/reflection/comparable.h"

#include "utl/to_vec.h"

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

struct new_virt {
  location_idx_t location_;
  std::vector<transfer_rule_side_idx> rules_{};
};

struct hub_coverage {
  void mark_slow(location_idx_t const from, location_idx_t const to) {
    slow_from_.insert(from);
    slow_to_.insert(to);
  }
  bool in_hub(location_idx_t const from, location_idx_t const to) const {
    return !slow_from_.contains(from) || !slow_to_.contains(to);
  }
  hash_set<location_idx_t> slow_from_, slow_to_;
};

struct hub_lists {
  template <typename In, typename Out>
  void add(In const& in, Out const& out, duration_t const d) {
    if (in.empty() || out.empty()) {
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

template <typename From, typename To, typename Fn>
void for_each_hub(From const& from,
                  To const& to,
                  hub_coverage const& coverage,
                  Fn&& fn) {
  // unrestricted hub:
  // !slow -> all
  auto in = std::vector<location_idx_t>{};
  for (auto const m : from) {
    if (!coverage.slow_from_.contains(m)) {
      in.push_back(m);
    }
  }
  auto out = utl::to_vec(to);
  fn(in, out);

  // In case no slow from locations exist,
  // skip building the restricted hub (would have zero ingress).
  if (in.size() == from.size()) {
    return;
  }

  in.clear();
  out.clear();

  // restricted hub:
  // slow -> !slow
  for (auto const m : from) {
    if (coverage.slow_from_.contains(m)) {
      in.push_back(m);
    }
  }
  for (auto const t : to) {
    if (!coverage.slow_to_.contains(t)) {
      out.push_back(t);
    }
  }
  fn(in, out);
}

struct rule_hubs {
  hub_lists hubs_;
  hash_map<transfer_rule_idx_t, hub_coverage> coverage_;
};

rule_hubs get_rule_hubs(
    timetable const&,
    hash_map<transfer_pair, transfer_rule_idx_t> const& most_specific);

struct stop_hubs {
  hub_lists hubs_;
  hub_coverage coverage_;
};

stop_hubs get_stop_hubs(
    timetable const&,
    hash_map<transfer_pair, transfer_rule_idx_t> const& most_specific,
    std::span<location_idx_t const> virts);

void store_rule_lookups(timetable&, interval<transfer_rule_idx_t> rules);

location_idx_t get_or_create_virt(
    timetable&,
    hash_map<virt_key, new_virt>& virts,
    location_idx_t base,
    std::vector<transfer_rule_side_idx> const& sig);

void store_virt_lookups_to_tt(timetable&,
                              hash_map<virt_key, new_virt> const& virts);

struct rule_transfers {
  hub_lists hubs_;
  mutable_fws_multimap<location_idx_t, footpath> footpaths_;
};

rule_transfers get_rule_transfers(timetable const&);

}  // namespace nigiri::loader
