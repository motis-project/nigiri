#include "nigiri/routing/search_interval.h"

#include "utl/visit.h"

namespace nigiri::routing {

interval<unixtime_t> start_time_interval(
    std::variant<unixtime_t, interval<unixtime_t>> const& start_time) {
  return utl::visit(
      start_time,
      [](unixtime_t const t) { return interval{t, t + i32_minutes{1}}; },
      [](interval<unixtime_t> const i) { return i; });
}

interval<unixtime_t> reachable_events(direction const search_dir,
                                      interval<unixtime_t> const max_search_iv,
                                      duration_t const max_reach) {
  return search_dir == direction::kForward
             ? interval{max_search_iv.from_, max_search_iv.to_ + max_reach}
             : interval{max_search_iv.from_ - max_reach, max_search_iv.to_};
}

}  // namespace nigiri::routing
