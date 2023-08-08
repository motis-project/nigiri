#pragma once

#include <vector>

#include "nigiri/routing/query.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

struct arcflag_filter {
  arcflag_filter(timetable const& tt, query const& q) {
    auto dest_partitions = hash_set<partition_idx_t>{};
    for (auto const offset : q.destination_) {
      for (auto const p :
           tt.component_partitions_[tt.locations_
                                        .components_[offset.target()]]) {
        dest_partitions.emplace(p);
      }
    }

    route_filtered_.resize(tt.n_routes(), true);
    for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
      auto const p = tt.route_partitions_[r];
      if (dest_partitions.contains(p)) {
        route_filtered_[to_idx(r)] = false;
        break;
      }
      for (auto const dp : dest_partitions) {
        if (tt.arc_flags_[r].test(to_idx(dp))) {
          route_filtered_[to_idx(r)] = false;
          break;
        }
      }
    }
  }

  bool is_filtered(route_idx_t const r) const {
    return route_filtered_[to_idx(r)];
  }

  std::vector<bool> route_filtered_;
};

}  // namespace nigiri::routing