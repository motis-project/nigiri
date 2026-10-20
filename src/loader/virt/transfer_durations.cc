#include "nigiri/loader/virt/transfer_durations.h"

#include "utl/get_or_create.h"
#include "utl/zip.h"

#include "nigiri/loader/virt/signatures.h"
#include "nigiri/footpath.h"
#include "nigiri/routing/transfers.h"
#include "nigiri/stop.h"
#include "nigiri/timetable.h"

namespace nigiri::loader {

duration_t get_walk(timetable const& tt,
                    location_idx_t const from,
                    location_idx_t const to) {
  return routing::shortest_transfer(tt, nullptr, kDefaultProfile, from, to)
      .value_or(kNever);
}

hash_map<transfer_pair, transfer_durations> get_transfer_durations(
    timetable const& tt, signatures const& sigs) {
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
    if (!r.is_qualified(event_type::kArr) &&
        !r.is_qualified(event_type::kDep)) {
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
              auto const fallback = get_walk(tt, from, to);
              return transfer_durations{
                  fallback, window_t{fallback, fallback + duration_t{1}}};
            });
        d.window_ = d.window_.extend({m, m + duration_t{1}});
      });
    });
  }
  return durations_of;
}

}  // namespace nigiri::loader
