#pragma once

#include "utl/pairwise.h"

#include "nigiri/timetable.h"

namespace nigiri::loader {

void build_lb_graph(timetable& tt) {
  std::map<location_idx_t, duration_t> weights;

  auto const update_weight = [&](location_idx_t const target,
                                 duration_t const d) {
    if (auto const it = weights.find(target); it != end(weights)) {
      it->second = std::min(it->second, d);
    } else {
      weights.emplace_hint(it, target, d);
    }
  };

  auto const add_edges = [&](location_idx_t const l) {
    auto const parent_l = tt.locations_.parents_[l] == location_idx_t::invalid()
                              ? l
                              : tt.locations_.parents_[l];

    for (auto const& fp : tt.locations_.footpaths_in_[l]) {
      auto const parent = tt.locations_.parents_[fp.target_];
      auto const target =
          parent == location_idx_t::invalid() ? fp.target_ : parent;
      if (target != parent_l) {
        update_weight(target, fp.duration_);
      }
    }

    for (auto const& r : tt.location_routes_[l]) {
      auto const location_seq = tt.route_location_seq_[r];
      for (auto const [from, to] :
           utl::pairwise(interval{0U, location_seq.size()})) {
        auto const from_l = timetable::stop{location_seq[from]}.location_idx();
        auto const to_l = timetable::stop{location_seq[to]}.location_idx();
        if (to_l != l) {
          // We only collect incoming edges.
          // Inverted graph => incoming edges become outgoing edges.
          continue;
        }

        auto const target_parent = tt.locations_.parents_[from_l];
        auto const target =
            target_parent == location_idx_t::invalid() ? from_l : target_parent;
        if (target == parent_l) {
          continue;
        }

        auto min = duration_t{std::numeric_limits<duration_t::rep>::max()};
        for (auto const t : tt.route_transport_ranges_[r]) {
          auto const from_time = tt.event_mam(t, from, event_type::kDep);
          auto const to_time = tt.event_mam(t, to, event_type::kArr);
          min = std::min(to_time - from_time, min);
        }
        update_weight(target, min);
      }
    }
  };

  std::vector<footpath> footpaths;
  for (auto i = location_idx_t{0U}; i != tt.locations_.ids_.size(); ++i) {
    if (tt.locations_.parents_[i] != location_idx_t::invalid()) {
      tt.lb_graph_.emplace_back(std::vector<footpath>{});
      continue;
    }

    for (auto const& c : tt.locations_.children_[i]) {
      add_edges(c);
    }
    add_edges(i);

    for (auto const& [target, duration] : weights) {
      footpaths.emplace_back(
          footpath{.target_ = target, .duration_ = duration});
    }

    tt.lb_graph_.emplace_back(footpaths);

    footpaths.clear();
    weights.clear();
  }
}

}  // namespace nigiri::loader
