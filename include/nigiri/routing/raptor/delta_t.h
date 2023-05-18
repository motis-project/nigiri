#pragma once

#include "nigiri/types.h"

namespace nigiri::routing {

using delta_t = std::int16_t;
static_assert(sizeof(delta_t) == 2);

template <direction SearchDir>
inline constexpr auto const kInvalidDelta =
    SearchDir == direction::kForward ? std::numeric_limits<delta_t>::max()
                                     : std::numeric_limits<delta_t>::min();

inline delta_t unix_to_delta(date::sys_days const base, unixtime_t const t) {
  return (t - std::chrono::time_point_cast<unixtime_t::duration>(base)).count();
}

inline delta_t tt_to_delta(day_idx_t const base,
                           day_idx_t const day,
                           minutes_after_midnight_t const mam) {
  auto const rel_day =
      (static_cast<int>(to_idx(day)) - static_cast<int>(to_idx(base)));
  return rel_day * 1440 + mam.count();
}

inline unixtime_t delta_to_unix(date::sys_days const base, delta_t const d) {
  return std::chrono::time_point_cast<unixtime_t::duration>(base) +
         d * unixtime_t::duration{1};
}

}  // namespace nigiri::routing