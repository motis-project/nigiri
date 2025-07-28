#include "nigiri/loader/reduce_footpaths.h"

#include "nigiri/timetable.h"

#include "utl/erase_duplicates.h"

namespace nigiri::loader {

vecvec<location_idx_t, footpath> reduce_footpaths(
    timetable& tt,
    vecvec<location_idx_t, footpath> const& fps,
    std::size_t const n) {
  using diff_t = std::vector<footpath>::iterator::difference_type;
  auto reduced =
      vector_map<location_idx_t, std::vector<footpath>>(tt.n_locations());
  auto reachable = hash_map<route_idx_t, std::vector<footpath>>{};
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    for (auto& x : reachable) {
      x.second.clear();
    }

    // Group by route (duplicates footpaths).
    // Skips routes that are already available at this stop.
    // This also eliminates footpaths to locations without scheduled traffic.
    auto const l_routes = tt.location_routes_[l];
    for (auto const& fp : fps[l]) {
      for (auto const& r : tt.location_routes_[fp.target()]) {
        if (utl::find(l_routes, r) == end(l_routes)) {
          reachable[r].emplace_back(fp);
        }
      }
    }

    // Sort per-route footpath by duration.
    for (auto& [_, x] : reachable) {
      utl::sort(x, [](footpath const& a, footpath const& b) {
        return a.duration() < b.duration();
      });
    }

    // Join fastest N footpaths per route.
    auto& reduced_l_fps = reduced[l];
    for (auto const& [r, x] : reachable) {
      reduced_l_fps.insert(
          end(reduced[l]), begin(x),
          begin(x) + static_cast<diff_t>(std::min(n, x.size())));
    }

    // Deduplicate and sort by duration.
    utl::erase_duplicates(reduced_l_fps,
                          [](footpath const& a, footpath const& b) {
                            return std::tuple{a.duration(), a.target()} <
                                   std::tuple{b.duration(), b.target()};
                          });
  }

  // Copy to vecvec.
  auto compact = vecvec<location_idx_t, footpath>{};
  for (auto const& r : reduced) {
    compact.emplace_back(r);
  }
  return compact;
}

}  // namespace nigiri::loader