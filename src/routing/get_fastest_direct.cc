#include "nigiri/routing/get_fastest_direct.h"
#include "nigiri/routing/for_each_meta.h"

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
      for (auto const& dest : q.destination_) {
        if (dest.target_ == fp.target()) {
          min = std::min(min, start.duration_ + fp.duration() + dest.duration_);
        }
      }
    }
  }
  return min;
}

duration_t get_fastest_start_dest_overlap(timetable const& tt, query const& q) {
  auto min = duration_t{std::numeric_limits<duration_t::rep>::max()};
  for (auto const& s : q.start_) {
    for_each_meta(tt, q.start_match_mode_, s.target_,
                  [&](location_idx_t const start) {
                    for (auto const& dest : q.destination_) {
                      if (start == dest.target_) {
                        min = std::min(min, s.duration_ + dest.duration_);
                      }
                    }
                  });
  }
  return min;
}

duration_t get_fastest_direct(timetable const& tt,
                              query const& q,
                              direction const dir) {
  return std::min(get_fastest_direct_with_foot(tt, q, dir),
                  get_fastest_start_dest_overlap(tt, q));
}

}  // namespace nigiri::routing
