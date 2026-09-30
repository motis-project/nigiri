#pragma once

#include "fmt/base.h"

#include "nigiri/types.h"

#include "cista/gpu_compat.h"

namespace nigiri {

using delta_t = std::int16_t;
static_assert(sizeof(delta_t) == 2);

template <direction SearchDir>
CISTA_GPU_DEVICE_COMPAT constexpr static auto const kInvalidDelta =
    SearchDir == direction::kForward ? std::numeric_limits<delta_t>::max()
                                     : std::numeric_limits<delta_t>::min();

template <typename T>
inline constexpr delta_t clamp(T t) {
#if defined(NIGIRI_TRACING) && !defined(__CUDA_ARCH__) && \
    !defined(__HIP_DEVICE_COMPILE__)
  if (t < std::numeric_limits<delta_t>::min()) {
    fmt::print("CLAMP {} TO {}\n", t, std::numeric_limits<delta_t>::min());
  }
  if (t > std::numeric_limits<delta_t>::max()) {
    fmt::print("CLAMP {} TO {}\n", t, std::numeric_limits<delta_t>::max());
  }
#endif

  // open-coded instead of std::clamp: its precondition check references
  // __glibcxx_assert_fail, a host function clang refuses to see from device
  // code
  constexpr auto const lo =
      static_cast<int>(std::numeric_limits<delta_t>::min());
  constexpr auto const hi =
      static_cast<int>(std::numeric_limits<delta_t>::max());
  return static_cast<delta_t>(t < lo ? lo : (t > hi ? hi : t));
}

inline constexpr delta_t unix_to_delta(date::sys_days const base,
                                       unixtime_t const t) {
  return clamp(
      (t - std::chrono::time_point_cast<unixtime_t::duration>(base)).count());
}

inline constexpr delta_t tt_to_delta(day_idx_t const base,
                                     day_idx_t const day,
                                     minutes_after_midnight_t const mam) {
  auto const rel_day =
      (static_cast<int>(to_idx(day)) - static_cast<int>(to_idx(base)));
  return clamp(rel_day * 1440 + mam.count());
}

inline constexpr unixtime_t delta_to_unix(date::sys_days const base,
                                          delta_t const d) {
  return std::chrono::time_point_cast<unixtime_t::duration>(base) +
         d * unixtime_t::duration{1};
}

inline constexpr std::pair<day_idx_t, minutes_after_midnight_t> split_day_mam(
    day_idx_t const base, delta_t const x) {
  auto const minutes = base.v_ * 1440 + x;
  return {day_idx_t{minutes / 1440}, minutes_after_midnight_t{minutes % 1440}};
}

}  // namespace nigiri