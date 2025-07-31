#include "nigiri/loader/reduce_footpaths.h"

#include "utl/erase_duplicates.h"
#include "utl/insert_sorted.h"
#include "utl/parallel_for.h"
#include "utl/timer.h"

#include "nigiri/timetable.h"

namespace nigiri::loader {

constexpr auto const kN = 2U;

vecvec<location_idx_t, footpath> reduce_footpaths(
    timetable& tt, vecvec<location_idx_t, footpath> const& fps) {
  auto const timer = utl::scoped_timer{"reduce-footpaths"};

  auto const init = []() {
    auto x = std::array<footpath, kN>{};
    x.fill(footpath{footpath::kMaxTarget, footpath::kMaxDuration});
    return x;
  }();

  auto reduced =
      vector_map<location_idx_t, std::vector<footpath>>(tt.n_locations());

  struct state {
    vector_map<route_idx_t, std::array<footpath, kN>> route_fps_;
    bitvec_map<route_idx_t> route_is_reachable_;
  };

  utl::parallel_for_run_threadlocal<state>(
      tt.n_locations(), [&](state& s, std::size_t const l_idx) {
        auto const l = location_idx_t{l_idx};

        if (s.route_fps_.size() != tt.n_locations()) {
          s.route_fps_ = vector_map<route_idx_t, std::array<footpath, kN>>{
              tt.n_locations(), init};
          s.route_is_reachable_.resize(tt.n_locations());
        }

        // Group by route (duplicates footpaths).
        // This eliminates footpaths to locations w/o scheduled traffic.
        auto const l_routes = tt.location_routes_[l];
        for (auto const& fp : fps[l]) {
          for (auto const& r : tt.location_routes_[fp.target()]) {
            if (utl::find(l_routes, r) != end(l_routes)) {
              continue;  // Skip routes that are already available at this stop.
            }

            s.route_is_reachable_.set(r, true);

            auto& r_reachable = s.route_fps_[r];

            auto insert = fp;
            for (auto i = 0U; i != r_reachable.size(); ++i) {
              if (insert.duration() < r_reachable[i].duration()) {
                std::swap(insert, r_reachable[i]);
              }
            }
          }
        }

        // Join fastest N footpaths per route.
        s.route_is_reachable_.for_each_set_bit([&](route_idx_t const r) {
          auto& route_fps = s.route_fps_[r];
          for (auto j = 0U; j != route_fps.size(); ++j) {
            if (route_fps[j].target() != footpath::kMaxTarget) {
              reduced[l].push_back(route_fps[j]);
            }
          }
          route_fps = init;
        });
        s.route_is_reachable_.zero_out();

        // Deduplicate and sort by duration.
        utl::erase_duplicates(reduced[l],
                              [](footpath const& a, footpath const& b) {
                                return std::tuple{a.duration(), a.target()} <
                                       std::tuple{b.duration(), b.target()};
                              });
      });

  // Copy to vecvec.
  auto compact = vecvec<location_idx_t, footpath>{};
  for (auto const& r : reduced) {
    compact.emplace_back(r);
  }

  // Count.
  auto n_full = 0U;
  for (auto const x : fps) {
    n_full += x.size();
  }
  auto n_reduced = 0U;
  for (auto const x : compact) {
    n_reduced += x.size();
  }

  log(log_lvl::info, "nigiri.loader.reduce_footpaths",
      "reduce footpaths: #full={}, #reduced={} ({}%)", n_full, n_reduced,
      100.0 * n_reduced / n_full);

  return compact;
}

}  // namespace nigiri::loader