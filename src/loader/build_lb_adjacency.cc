#include "nigiri/loader/build_lb_adjacency.h"

#include <ranges>

#include "utl/enumerate.h"
#include "utl/get_or_create.h"
#include "utl/parallel_for.h"
#include "utl/progress_tracker.h"

namespace nigiri::loader {

void build_lb_adjacency(timetable& tt, profile_idx_t const prf_idx) {
  auto const timer = scoped_timer{"loader.build_lb_adjacency"};

  struct adjacencies {
    hash_map<location_idx_t, std::uint16_t> in_;
    hash_map<location_idx_t, std::uint16_t> out_;
  };

  struct lb_neighbors {
    std::vector<lb_neighbor> in_;
    std::vector<lb_neighbor> out_;
  };

  auto const explore_routes = [&](location_idx_t const l, adjacencies& a) {
    for (auto const r : tt.location_routes_[l]) {
      if ((prf_idx == kCarProfile && !tt.has_car_transport(r)) ||
          (prf_idx == kBikeProfile && !tt.has_bike_transport(r))) {
        continue;
      }

      auto const location_seq = tt.route_location_seq_[r];
      for (auto const [i, x] : utl::enumerate(location_seq)) {
        auto const i_stop = stop{x};
        auto const i_stop_idx = static_cast<stop_idx_t>(i);
        if (l != i_stop.location_idx()) {
          continue;
        }

        for (auto const [j, y] : utl::enumerate(location_seq)) {
          auto const j_stop = stop{y};
          auto const j_stop_idx = static_cast<stop_idx_t>(j);
          for (auto const t : tt.route_transport_ranges_[r]) {
            auto const min = [&](auto& hm, auto const n, auto const d) {
              auto& v = utl::get_or_create(hm, n, [] {
                return std::numeric_limits<std::uint16_t>::max();
              });
              v = std::min(v, d);
            };

            if (j < i && j_stop.in_allowed() &&
                i_stop.out_allowed()) {  // TODO wheelchair profile
              min(a.in_, j_stop.location_idx(),
                  static_cast<std::uint16_t>(
                      (tt.event_mam(t, i_stop_idx, event_type::kArr) -
                       tt.event_mam(t, j_stop_idx, event_type::kDep))
                          .count()));
            }

            if (i < j && i_stop.in_allowed() &&
                j_stop.out_allowed()) {  // TODO wheelchair profile
              min(a.out_, j_stop.location_idx(),
                  static_cast<std::uint16_t>(
                      (tt.event_mam(t, j_stop_idx, event_type::kArr) -
                       tt.event_mam(t, i_stop_idx, event_type::kDep))
                          .count()));
            }
          }
        }
      }
    }
  };

  auto const pt = utl::get_active_progress_tracker();
  pt->status("Compute lower bound adjacencies").in_high(tt.n_locations());
  utl::parallel_ordered_collect_threadlocal<adjacencies>(
      tt.n_locations(),
      // parallel
      [&](adjacencies& a, std::size_t const i) {
        auto const l = location_idx_t{i};
        a.in_.clear();
        a.out_.clear();
        auto ns = lb_neighbors{};

        explore_routes(l, a);

        for (auto const& [n, dist] : a.in_) {
          ns.in_.emplace_back(n, dist);
        }
        for (auto const& [n, dist] : a.out_) {
          ns.out_.emplace_back(n, dist);
        }

        return ns;
      },
      // ordered
      [&](std::size_t, lb_neighbors&& ns) {
        tt.fwd_lb_adjacency_[prf_idx].emplace_back(ns.in_);
        tt.bwd_lb_adjacency_[prf_idx].emplace_back(ns.out_);
      },
      pt->update_fn());
}

}  // namespace nigiri::loader