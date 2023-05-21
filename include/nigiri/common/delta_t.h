#pragma once

#include "nigiri/routing/raptor/debug.h"
#include "nigiri/types.h"

namespace nigiri {

using delta_t = std::int16_t;
static_assert(sizeof(delta_t) == 2);

template <direction SearchDir>
inline constexpr auto const kInvalidDelta =
    SearchDir == direction::kForward ? std::numeric_limits<delta_t>::max()
                                     : std::numeric_limits<delta_t>::min();

template <typename T>
inline delta_t clamp(T t) {
#if defined(NIGIRI_TRACING)
  if (t < std::numeric_limits<delta_t>::min()) {
    trace_upd("CLAMP {} TO {}\n", t, std::numeric_limits<delta_t>::min());
  }
  if (t > std::numeric_limits<delta_t>::max()) {
    trace_upd("CLAMP {} TO {}\n", t, std::numeric_limits<delta_t>::max());
  }
#endif

  return static_cast<delta_t>(
      std::clamp(t, static_cast<int>(std::numeric_limits<delta_t>::min()),
                 static_cast<int>(std::numeric_limits<delta_t>::max())));
}

inline delta_t unix_to_delta(date::sys_days const base, unixtime_t const t) {
  return clamp(
      (t - std::chrono::time_point_cast<unixtime_t::duration>(base)).count());
}

inline delta_t tt_to_delta(day_idx_t const base,
                           day_idx_t const day,
                           minutes_after_midnight_t const mam) {
  auto const rel_day =
      (static_cast<int>(to_idx(day)) - static_cast<int>(to_idx(base)));
  return clamp(rel_day * 1440 + mam.count());
}

inline unixtime_t delta_to_unix(date::sys_days const base, delta_t const d) {
  return std::chrono::time_point_cast<unixtime_t::duration>(base) +
         d * unixtime_t::duration{1};
}

inline std::pair<day_idx_t, minutes_after_midnight_t> split_day_mam(
    day_idx_t const base, delta_t const x) {
  assert(x != std::numeric_limits<delta_t>::min());
  assert(x != std::numeric_limits<delta_t>::max());
  if (x < 0) {
    auto const t = -x / 1440 + 1;
    auto const min = x + (t * 1440);
    return {static_cast<day_idx_t>(static_cast<int>(to_idx(base)) - t),
            minutes_after_midnight_t{min}};
  } else {
    return {static_cast<day_idx_t>(static_cast<int>(to_idx(base)) + x / 1440),
            minutes_after_midnight_t{x % 1440}};
  }
}

}  // namespace nigiri