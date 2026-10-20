#include "nigiri/loader/virt/location_route_stops.h"

#include <vector>

#include "nigiri/stop.h"
#include "nigiri/timetable.h"

namespace nigiri::loader {

bool has_event(timetable const& tt,
               route_idx_t const r,
               stop_idx_t const i,
               event_type const ev) {
  auto const seq = tt.route_location_seq_[r];
  auto const stp = stop{seq[i]};
  return ev == event_type::kArr ? i != 0U && stp.out_allowed()
                                : i + 1U != seq.size() && stp.in_allowed();
}

location_route_stops_t get_location_route_stops(
    timetable const& tt,
    signatures const& sigs,
    hash_map<transfer_pair, transfer_durations> const& durations_of) {
  // Mark any stop that's involved in a transfer rule.
  auto is_needed = bitvec_map<location_idx_t>{};
  is_needed.resize(tt.n_locations());
  for (auto const& [stops, _] : durations_of) {
    is_needed.set(stops.from_);
    is_needed.set(stops.to_);
  }

  // For each location: collect all route stops with their signatures.
  auto route_stops = location_route_stops_t{};
  auto v = std::vector<location_route_stop>{};
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    v.clear();
    if (is_needed.test(l)) {
      for (auto const r : tt.location_routes_[l]) {
        auto const seq = tt.route_location_seq_[r];
        for (auto i = stop_idx_t{0U}; i != seq.size(); ++i) {
          if (stop{seq[i]}.location_idx() != l) {
            continue;
          }
          v.push_back(location_route_stop{
              .route_ = r,
              .idx_ = i,
              .sigs_ = sigs.get(r, i),
              .has_arr_ = has_event(tt, r, i, event_type::kArr),
              .has_dep_ = has_event(tt, r, i, event_type::kDep)});
        }
      }
    }
    route_stops.emplace_back(v);
  }
  return route_stops;
}

}  // namespace nigiri::loader
