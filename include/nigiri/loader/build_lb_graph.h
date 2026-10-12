#pragma once

#include "utl/pairwise.h"
#include "utl/parallel_for.h"

#include "nigiri/logging.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::loader {

template <direction SearchDir>
void build_lb_graph(timetable& tt, profile_idx_t const prf_idx) {
  using weights_t = hash_map<location_idx_t, duration_t>;

  auto const update_weight = [](weights_t& weights, location_idx_t const target,
                                duration_t const d) {
    if (auto const it = weights.find(target); it != end(weights)) {
      it->second = std::min(it->second, d);
    } else {
      weights.emplace_hint(it, target, d);
    }
  };

  auto const add_edges = [&](weights_t& weights, location_idx_t const l) {
    auto const parent_l = tt.locations_.get_root_idx(l);

    auto const& footpaths = SearchDir == direction::kForward
                                ? tt.locations_.footpaths_in_[prf_idx][l]
                                : tt.locations_.footpaths_out_[prf_idx][l];
    for (auto const& fp : footpaths) {
      auto const target = tt.locations_.get_root_idx(fp.target());
      if (target != parent_l) {
        update_weight(weights, target, fp.duration());
      }
    }

    for (auto const& r : tt.location_routes_[l]) {
      if ((prf_idx == kCarProfile && !tt.is_flag_set(kCarsAllowed, r)) ||
          (prf_idx == kBikeProfile && !tt.is_flag_set(kBikesAllowed, r))) {
        continue;
      }

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
        auto const target = tt.locations_.get_root_idx(target_l);
        if (target == parent_l) {
          continue;
        }

        auto min = duration_t{std::numeric_limits<duration_t::rep>::max()};
        for (auto const t : tt.route_transport_ranges_[r]) {
          auto const from_time = tt.event_mam(t, from, event_type::kDep);
          auto const to_time = tt.event_mam(t, to, event_type::kArr);
          min = std::min((to_time - from_time).as_duration(), min);
        }
        update_weight(weights, target, min);
      }
    }
  };

  auto const timer = scoped_timer{"nigiri.loader.lb"};

  // A location's edges only read the timetable: collect them in parallel,
  // then append them in location order.
  auto edges = std::vector<std::vector<footpath>>(tt.locations_.ids_.size());
  utl::parallel_for_run_threadlocal<weights_t>(
      edges.size(), [&](weights_t& weights, std::size_t const idx) {
        auto const i = location_idx_t{idx};
        if (tt.locations_.parents_[i] != location_idx_t::invalid()) {
          return;
        }

        for (auto const& c : tt.locations_.children_[i]) {
          add_edges(weights, c);
          for (auto const& cc : tt.locations_.children_[c]) {
            add_edges(weights, cc);
          }
        }
        add_edges(weights, i);

        for (auto const& [target, duration] : weights) {
          edges[idx].emplace_back(footpath{target, duration});
        }
        weights.clear();
      });

  auto& lb_graph = SearchDir == direction::kForward
                       ? tt.fwd_search_lb_graph_[prf_idx]
                       : tt.bwd_search_lb_graph_[prf_idx];
  for (auto const& e : edges) {
    lb_graph.emplace_back(e);
  }
}

}  // namespace nigiri::loader
