#pragma once

#include "nigiri/routing/limits.h"
#include "nigiri/routing/query.h"

namespace nigiri::routing {

inline void sanitize_query(query& q) {
  // Ensure upper bound required by algorithm is a valid duration
  constexpr auto const kAdditionalRequiredTime = decltype(kMaxTravelTime){1};
  static_assert(kMaxTravelTime + kAdditionalRequiredTime <=
                decltype(kMaxTravelTime)::max());
  if (q.max_travel_time_.count() < 0 || q.max_travel_time_ > kMaxTravelTime) {
    q.max_travel_time_ = kMaxTravelTime;
  }
}

}  // namespace nigiri::routing