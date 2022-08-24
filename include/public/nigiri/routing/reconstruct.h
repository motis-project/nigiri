#pragma once

#include "nigiri/routing/limits.h"
#include "nigiri/routing/search_state.h"

namespace nigiri::routing {

template <direction SearchDir>
void reconstruct(timetable const& tt,
                 query const& q,
                 search_state const& s,
                 journey& j) {
  constexpr auto const kIsFwd = SearchDir == direction::kForward;

  auto curr_location = location_idx_t{j.dest_};
  auto curr_time = j.dest_time_;

  (void)q;
  (void)s;
  (void)curr_time;

  // auto const get_transport = [&]() { return journey::leg{}; };
  auto const get_legs =
      [&](unsigned const) -> std::pair<journey::leg, journey::leg> {
    auto const& fps = kIsFwd ? tt.locations_.footpaths_in_[curr_location]
                             : tt.locations_.footpaths_out_[curr_location];

    for (auto const& fp : fps) {
      (void)fp;
    }

    return {};
  };

  for (auto k = 0U; k != kMaxTransfers + 1; ++k) {
    auto [fp_leg, transport_leg] = get_legs(k);
    j.add(std::move(fp_leg));
    j.add(std::move(transport_leg));
  }
}

template <direction SearchDir>
void reconstruct(timetable const& tt, query const& q, search_state& s) {
  auto const starts_in_interval = [&](journey const& j) {
    return q.interval_.contains(j.start_time_);
  };

  for (auto it = begin(s.results_); it != end(s.results_);) {
    if (starts_in_interval(*it)) {
      reconstruct<SearchDir>(tt, q, s, *it);
      ++it;
    } else {
      it = s.results_.erase(it);
    }
  }
}

}  // namespace nigiri::routing