#pragma once

#include <limits>
#include <utility>

#include "utl/pairwise.h"

#include "nigiri/logging.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::loader {

template <direction SearchDir>
void build_lb_graph(timetable& tt, profile_idx_t const prf_idx = 0) {
  hash_map<location_idx_t,
           std::pair<duration_t /*footpath*/, duration_t /*trip*/>>
      weights;

  auto const update_footpath_weight = [&](location_idx_t const target,
                                          duration_t const d) {
    if (auto const it = weights.find(target); it != end(weights)) {
      it->second.first = std::min(it->second.first, d);
    } else {
      weights.emplace_hint(it, target, std::pair{d, lb_entry::kUnreachable});
    }
  };

  auto const update_trip_weight = [&](location_idx_t const target,
                                      duration_t const d) {
    if (auto const it = weights.find(target); it != end(weights)) {
      it->second.second = std::min(it->second.second, d);
    } else {
      weights.emplace_hint(it, target, std::pair{lb_entry::kUnreachable, d});
    }
  };

  auto const add_edges = [&](location_idx_t const l) {
    auto const parent_l = tt.locations_.parents_[l] == location_idx_t::invalid()
                              ? l
                              : tt.locations_.parents_[l];

    auto const& footpaths = SearchDir == direction::kForward
                                ? tt.locations_.footpaths_in_[prf_idx][l]
                                : tt.locations_.footpaths_out_[prf_idx][l];
    for (auto const& fp : footpaths) {
      auto const parent = tt.locations_.parents_[fp.target()];
      auto const target =
          parent == location_idx_t::invalid() ? fp.target() : parent;
      if (target != parent_l) {
        update_footpath_weight(target, fp.duration());
      }
    }

    for (auto const& r : tt.location_routes_[l]) {
      auto const location_seq = tt.route_location_seq_[r];
      for (auto const [from, to] : utl::pairwise(interval{
               stop_idx_t{0U}, static_cast<stop_idx_t>(location_seq.size())})) {
        auto const from_l = stop{location_seq[from]}.location_idx();
        auto const to_l = stop{location_seq[to]}.location_idx();

        if ((SearchDir == direction::kForward ? to_l : from_l) != l) {
          continue;
        }

        auto const target_l =
            (SearchDir == direction::kForward ? from_l : to_l);
        auto const target_parent = tt.locations_.parents_[target_l];
        auto const target = target_parent == location_idx_t::invalid()
                                ? target_l
                                : target_parent;
        if (target == parent_l) {
          continue;
        }

        auto min = duration_t{std::numeric_limits<duration_t::rep>::max()};
        for (auto const t : tt.route_transport_ranges_[r]) {
          auto const from_time = tt.event_mam(t, from, event_type::kDep);
          auto const to_time = tt.event_mam(t, to, event_type::kArr);
          min = std::min((to_time - from_time).as_duration(), min);
        }
        update_trip_weight(target, min);
      }
    }
  };

  auto const timer = scoped_timer{"nigiri.loader.lb"};
  std::vector<lb_entry> entries;
  auto& lb_graph = SearchDir == direction::kForward ? tt.fwd_search_lb_graph_
                                                    : tt.bwd_search_lb_graph_;
  for (auto i = location_idx_t{0U}; i != tt.locations_.ids_.size(); ++i) {
    if (tt.locations_.parents_[i] != location_idx_t::invalid()) {
      lb_graph.emplace_back(std::vector<lb_entry>{});
      continue;
    }

    for (auto const& c : tt.locations_.children_[i]) {
      add_edges(c);
    }
    add_edges(i);

    for (auto const& [target, duration] : weights) {
      entries.emplace_back(target, duration.first, duration.second);
    }

    lb_graph.emplace_back(entries);

    entries.clear();
    weights.clear();
  }
}

}  // namespace nigiri::loader
