#pragma once

#include <vector>

#include "nigiri/routing/query.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

struct arcflag_filter {
  void init(timetable const& tt, query const& q) {
    dest_partition_.clear();
    route_filtered_.resize(tt.n_routes(), true);

    for (auto const offset : q.destination_) {
      for (auto const p :
           tt.component_partitions_[tt.locations_
                                        .components_[offset.target()]]) {
        dest_partition_.emplace(p);
      }
    }

    //    std::cout << "dest partitions: ";
    //    for (auto const d : dest_partition_) {
    //      std::cout << static_cast<int>(to_idx(d)) << " ";
    //    }
    //    std::cout << "\n";

    for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
      auto const p = tt.route_partitions_[r];
      if (dest_partition_.contains(p)) {
        //        std::cout << tt.transport_name(
        //                         tt.route_transport_ranges_[route_idx_t{r}].from_)
        //                  << " in partition " << static_cast<int>(to_idx(p))
        //                  << " is dest partition -> not filtered\n";
        route_filtered_[to_idx(r)] = false;
        continue;
      }
      for (auto const dp : dest_partition_) {
        //        std::cout << "dest_partition=" << static_cast<int>(to_idx(dp))
        //                  << ", route=" << r << ": "
        //                  << tt.transport_name(
        //                         tt.route_transport_ranges_[route_idx_t{r}].from_)
        //                  << ": " << tt.arc_flags_[r].test(to_idx(dp));
        if (tt.arc_flags_[r].test(to_idx(dp))) {
          route_filtered_[to_idx(r)] = false;
          //          std::cout << " -> NOT FILTERED\n";
          break;
        } else {
          //          std::cout << " - > filtered\n";
        }
      }
    }
  }

  bool is_filtered(route_idx_t const r) const {
    return route_filtered_[to_idx(r)];
  }

  std::vector<bool> route_filtered_;
  hash_set<partition_idx_t> dest_partition_;
};

}  // namespace nigiri::routing