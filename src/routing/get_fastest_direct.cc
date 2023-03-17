#include "nigiri/routing/get_fastest_direct.h"

namespace nigiri::routing {

duration_t get_fastest_direct_with_foot(timetable const& tt,
                                        query const& q,
                                        direction const dir) {
  auto min = duration_t{std::numeric_limits<duration_t::rep>::max()};
  for (auto const& start : q.start_) {
    auto const& footpaths =
        (dir == direction::kForward ? tt.locations_.footpaths_out_
                                    : tt.locations_.footpaths_in_);
    for (auto const& fp : footpaths[start.target_]) {
      for (auto const& dest : q.destinations_.front()) {
        if (dest.target_ == fp.target_) {
          min = std::min(min, start.duration_ + fp.duration_ + dest.duration_);
        }
      }
    }
  }
  return min;
}

duration_t get_fastest_start_dest_overlap(query const& q) {
  utl::verify(!q.destinations_.empty(), "no destination");
  auto min = duration_t{std::numeric_limits<duration_t::rep>::max()};
  for (auto const& start : q.start_) {
    for (auto const& dest : q.destinations_.front()) {
      if (start.target_ == dest.target_) {
        min = std::min(min, start.duration_ + dest.duration_);
      }
    }
  }
  return min;
}

duration_t get_fastest_direct(timetable const& tt,
                              query const& q,
                              direction const dir) {
  return std::min(get_fastest_direct_with_foot(tt, q, dir),
                  get_fastest_start_dest_overlap(q));
}

}  // namespace nigiri::routing
