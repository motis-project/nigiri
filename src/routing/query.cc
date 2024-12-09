#include "nigiri/routing/query.h"

#include "nigiri/for_each_meta.h"

namespace nigiri::routing {
inline void sanitize_query(query& q) {
  if (q.max_travel_time_.count() < 0 || q.max_travel_time_ > kMaxTravelTime) {
    q.max_travel_time_ = kMaxTravelTime;
  }
}

inline void sanitize_via_stops(timetable const& tt, query& q) {
  while (q.via_stops_.size() >= 2) {
    auto updated = false;
    for (auto i = 0U; i < q.via_stops_.size() - 1; ++i) {
      auto& a = q.via_stops_[i];
      auto& b = q.via_stops_[i + 1];
      if (matches(tt, location_match_mode::kEquivalent, a.location_,
                  b.location_)) {
        a.stay_ += b.stay_;
        q.via_stops_.erase(q.via_stops_.begin() + i + 1);
        updated = true;
        break;
      }
    }
    if (!updated) {
      break;
    }
  }
}

void query::sanitize(timetable const& tt) {
  sanitize_query(*this);
  sanitize_via_stops(tt, *this);
}

}  // namespace nigiri::routing