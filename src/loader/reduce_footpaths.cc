#include "nigiri/loader/reduce_footpaths.h"

#include "nigiri/timetable.h"

#include "utl/erase_duplicates.h"
#include "utl/insert_sorted.h"

namespace nigiri::loader {

constexpr auto const N = 2U;

vecvec<location_idx_t, footpath> reduce_footpaths(
    timetable& tt, vecvec<location_idx_t, footpath> const& fps) {
  auto reduced =
      vector_map<location_idx_t, std::vector<footpath>>(tt.n_locations());

  auto init = std::array<footpath, N>{};
  init.fill(footpath{footpath::kMaxTarget, footpath::kMaxDuration});
  auto reachable =
      vector_map<route_idx_t, std::array<footpath, N>>{tt.n_routes(), init};

  auto reachable_bits = bitvec_map<route_idx_t>{tt.n_routes()};
  reachable_bits.one_out();  // trigger reset

  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    reachable_bits.zero_out();

    // Group by route (duplicates footpaths).
    // Skips routes that are already available at this stop.
    // This also eliminates footpaths to locations without scheduled traffic.
    auto const l_routes = tt.location_routes_[l];
    for (auto const& fp : fps[l]) {
      for (auto const& r : tt.location_routes_[fp.target()]) {
        if (utl::find(l_routes, r) != end(l_routes)) {
          continue;
        }

        reachable_bits.set(r, true);

        auto& r_reachable = reachable[r];

        auto insert = fp;
        for (auto i = 0U; i != r_reachable.size(); ++i) {
          if (insert.duration() < r_reachable[i].duration()) {
            std::swap(insert, r_reachable[i]);
          }
        }
      }
    }

    // Join fastest N footpaths per route.
    reachable_bits.for_each_set_bit([&](route_idx_t const r) {
      auto& r_reachable = reachable[r];
      for (auto j = 0U; j != r_reachable.size(); ++j) {
        if (r_reachable[j].target() != footpath::kMaxTarget) {
          reduced[l].push_back(r_reachable[j]);
        }
      }
      r_reachable = init;
    });

    // Deduplicate and sort by duration.
    utl::erase_duplicates(reduced[l], [](footpath const& a, footpath const& b) {
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