#pragma once

#include <span>

#include "utl/helpers/algorithm.h"

#include "nigiri/for_each_meta.h"
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
  routing::for_each_meta(tt, routing::location_match_mode::kExact, l,
                         [&](location_idx_t const c) {
                           for (auto const r : tt.location_routes_[c]) {
                             fn(c, r);
                           }
                         });
}

template <typename Pred = decltype([](route_idx_t) { return true; })>
bool any_route_at(timetable const& tt,
                  location_idx_t const l,
                  Pred&& pred = {}) {
  return utl::any_of(tt.location_routes_[l], pred) ||
         utl::any_of(tt.locations_.children_[l], [&](location_idx_t const c) {
           return tt.locations_.is_virt(c) &&
                  utl::any_of(tt.location_routes_[c], pred);
         });
}

}  // namespace nigiri
