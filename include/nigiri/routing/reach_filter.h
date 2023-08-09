#pragma once

#include <vector>

#include "geo/box.h"

#include "utl/enumerate.h"

#include "nigiri/routing/query.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

struct reach_filter {
  void init(timetable const& tt, query const& q) {
    route_filtered_.resize(tt.n_routes(), true);

    // Computes bounding box around all stations.
    // -> returns (center of bounding box, half diagonal as buffer)
    auto const get_start_end = [&](std::vector<offset> const& offsets) {
      auto bbox = geo::box{};
      for (auto const offset : offsets) {
        bbox.extend(tt.locations_.coordinates_[offset.target()]);
      }
      return std::pair{geo::midpoint(bbox.min_, bbox.max_),
                       geo::distance(bbox.min_, bbox.max_) / 2.0};
    };

    auto const [start_pos, start_buffer] = get_start_end(q.start_);
    auto const [end_pos, end_buffer] = get_start_end(q.destination_);

    for (auto const [r, stop_seq] : utl::enumerate(tt.route_location_seq_)) {
      for (auto const s : stop_seq) {
        auto const sp = tt.locations_.coordinates_[stop{s}.location_idx()];
        auto const dist = std::min(geo::distance(start_pos, sp) - start_buffer,
                                   geo::distance(sp, end_pos) - end_buffer);
        if (tt.route_reachs_[route_idx_t{r}] > dist) {
          route_filtered_[r] = false;  // reach > dist -> not filtered!
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