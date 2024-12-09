#pragma once

#include "nigiri/routing/limits.h"
#include "nigiri/routing/query.h"

namespace nigiri::routing {

inline void sanitize_query(query& q) {
  if (q.max_travel_time_.count() < 0 || q.max_travel_time_ > kMaxTravelTime) {
    q.max_travel_time_ = kMaxTravelTime;
  }
}

}  // namespace nigiri::routing