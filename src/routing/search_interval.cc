#include "nigiri/routing/search_interval.h"

#include <cstdint>
#include <algorithm>
#include <limits>

#include "utl/overloaded.h"

namespace nigiri::routing {

interval<unixtime_t> start_time_interval(
    std::variant<unixtime_t, interval<unixtime_t>> const& start_time) {
  return utl::visit(
      start_time,
      [](unixtime_t const t) { return interval{t, t + i32_minutes{1}}; },
      [](interval<unixtime_t> const i) {
        return interval{i.from_, std::max(i.to_, i.from_ + i32_minutes{1})};
      });
}

interval<unixtime_t> reachable_events(direction const search_dir,
                                      interval<unixtime_t> const& start_itv,
                                      duration_t const max_reach) {
  return search_dir == direction::kForward
             ? interval{start_itv.from_, start_itv.to_ + max_reach}
             : interval{start_itv.from_ - max_reach, start_itv.to_};
}

}  // namespace nigiri::routing
