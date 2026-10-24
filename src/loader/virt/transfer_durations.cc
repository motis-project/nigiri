#include "nigiri/loader/virt/transfer_durations.h"

#include <algorithm>

#include "utl/get_or_create.h"
#include "utl/zip.h"

#include "nigiri/loader/virt/footpaths.h"
#include "nigiri/loader/virt/signatures.h"
#include "nigiri/footpath.h"
#include "nigiri/stop.h"
#include "nigiri/timetable.h"

namespace nigiri::loader {

duration_t get_walk(timetable const& tt,
                    mutable_fws_multimap<location_idx_t, footpath> const& walks,
                    rule_index const& stop_rules,
                    location_idx_t const from,
                    location_idx_t const to) {
  if (from == to) {
    auto const t = tt.locations_.transfer_time_[from];
    return t == kNoTransferAllowed ? kNever : duration_t{t};
  }
  if (auto const rule = stop_rules.get(from, to); rule.has_value()) {
    return *rule == footpath::kBan ? kNever : *rule;
  }
  auto walk = kNever;
  for (auto const fp : walks[from]) {
    if (fp.target() == to) {
      walk = std::min(walk, fp.duration());
    }
  }
  return walk;
}

hash_map<transfer_pair, transfer_durations> get_transfer_durations(
    timetable const& tt,
    mutable_fws_multimap<location_idx_t, footpath> const& walks,
    rule_index const& stop_rules,
    signatures const& sigs) {
  auto const& locs = tt.locations_;

  // Mark stops with qualified transfer rules.
  auto rule_stops = bitvec_map<location_idx_t>{};
  rule_stops.resize(tt.n_locations());
  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    if (!sigs.has_rules(r)) {
      continue;
    }
    for (auto const [s, stop_sigs] :
         utl::zip(tt.route_location_seq_[r], sigs.route_stop_sigs_[r])) {
      if (!stop_sigs.empty()) {
        rule_stops.set(stop{s}.location_idx());
      }
    }
  }

  // For each from/to stop pair where either stop has qualified rules:
  // Build convex hull interval of
  // - the fallback duration (unqualified rule / routed / beelined)
  // - and all qualified transfer rule durations (forbidden => max=kNever)
  auto durations_of = hash_map<transfer_pair, transfer_durations>{};
  auto const for_each_stop = [&](location_idx_t const l, auto&& fn) {
    fn(l);
    for (auto const c : locs.children_[l]) {
      fn(c);
    }
  };
  for (auto const& r : tt.transfer_rules_.rules_) {
    if (!r.is_qualified()) {
      continue;
    }
    auto const m = r.is_forbidden() ? kNever : r.duration_;
    for_each_stop(r.from_stop_, [&](location_idx_t const from) {
      for_each_stop(r.to_stop_, [&](location_idx_t const to) {
        if (!rule_stops.test(from) && !rule_stops.test(to)) {
          return;
        }
        auto& d =
            utl::get_or_create(durations_of, transfer_pair{from, to}, [&]() {
              auto const fallback = get_walk(tt, walks, stop_rules, from, to);
              return transfer_durations{
                  fallback, window_t{fallback, fallback + duration_t{1}}};
            });
        d.hull_ = d.hull_.extend({m, m + duration_t{1}});
      });
    });
  }
  return durations_of;
}

}  // namespace nigiri::loader
