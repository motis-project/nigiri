#include "nigiri/common/search_interval.h"

#include <cstdint>
#include <algorithm>
#include <limits>

#include "utl/overloaded.h"

namespace nigiri {

interval<unixtime_t> start_time_interval(
    std::variant<unixtime_t, interval<unixtime_t>> const& start_time) {
  return std::visit(
      utl::overloaded{
          [](unixtime_t const t) { return interval{t, t + i32_minutes{1}}; },
          [](interval<unixtime_t> const i) {
            return interval{i.from_, std::max(i.to_, i.from_ + i32_minutes{1})};
          }},
      start_time);
}

interval<unixtime_t> reachable_events(direction const search_dir,
                                      interval<unixtime_t> const& start_itv,
                                      duration_t const max_reach) {
  using rep_t = unixtime_t::duration::rep;
  auto const clamped = [](std::int64_t const v) {
    return unixtime_t{unixtime_t::duration{static_cast<rep_t>(
        std::clamp(v, std::int64_t{std::numeric_limits<rep_t>::min()},
                   std::int64_t{std::numeric_limits<rep_t>::max()}))}};
  };
  auto const mins = [](unixtime_t const t) {
    return static_cast<std::int64_t>(t.time_since_epoch().count());
  };

  auto const reach = std::int64_t{max_reach.count()} + 1;
  return search_dir == direction::kForward
             ? interval{start_itv.from_, clamped(mins(start_itv.to_) + reach)}
             : interval{clamped(mins(start_itv.from_) - reach), start_itv.to_};
}

}  // namespace nigiri
