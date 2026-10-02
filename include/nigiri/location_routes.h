#pragma once

#include <span>
#include <vector>

#include "utl/erase_duplicates.h"
#include "utl/helpers/algorithm.h"

#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

namespace nigiri {

inline std::span<route_idx_t const> static_routes(timetable const& tt,
                                                  rt_timetable const* rtt,
                                                  location_idx_t const l) {
  if (rtt != nullptr && rtt->is_rt_location(l)) {
    return {};
  }
  return tt.location_routes_[l];
}

template <typename Fn>
void for_each_route_at_stop(timetable const& tt,
                            location_idx_t const l,
                            Fn&& fn) {
  auto routes = std::vector<route_idx_t>{begin(tt.location_routes_[l]),
                                         end(tt.location_routes_[l])};
  tt.locations_.for_each_virt(l, [&](location_idx_t const v) {
    routes.insert(end(routes), begin(tt.location_routes_[v]),
                  end(tt.location_routes_[v]));
  });
  utl::erase_duplicates(routes);
  for (auto const r : routes) {
    fn(r);
  }
}

template <typename Fn>
void for_each_route_at(timetable const& tt,
                       profile_idx_t const prf,
                       location_idx_t const l,
                       Fn&& fn) {
  if (is_projected(prf)) {
    for_each_route_at_stop(tt, l, fn);
  } else {
    for (auto const r : tt.location_routes_[l]) {
      fn(r);
    }
  }
}

template <typename Pred>
bool any_route_at(timetable const& tt, location_idx_t const l, Pred&& pred) {
  auto has_match = utl::any_of(tt.location_routes_[l], pred);
  tt.locations_.for_each_virt(l, [&](location_idx_t const v) {
    has_match = has_match || utl::any_of(tt.location_routes_[v], pred);
  });
  return has_match;
}

inline bool has_routes(timetable const& tt, location_idx_t const l) {
  return any_route_at(tt, l, [](route_idx_t) { return true; });
}

}  // namespace nigiri
