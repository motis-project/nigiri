#pragma once

#include <cstdint>

#include "cista/cuda_check.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/clasz_mask.h"

namespace nigiri {
struct timetable;
}

namespace nigiri::routing {

struct query;

// What McRAPTOR's criteria updates (mc_criteria.h) need per query; the search
// fills it from the query. Plain RAPTOR takes and ignores it.
struct mc_context {
  clasz_mask_t avoided_{to_mask(clasz::kAir)};  // avoided_modes_dim
  std::int16_t night_offset_{0};  // night_transfer_dim: night_offset()
};

// The night of night_transfer_dim: kNightStart to kNightEnd local time, with
// the query's fixed UTC offset (delta_t counts from midnight UTC).
inline constexpr int kNightStart = 0;  // [min after local midnight]
inline constexpr int kNightEnd = 360;

CISTA_CUDA_COMPAT constexpr int local_minute(delta_t const t,
                                             int const offset) {
  return ((static_cast<int>(t) + offset) % 1440 + 1440) % 1440;
}

CISTA_CUDA_COMPAT constexpr bool is_night(delta_t const t, int const offset) {
  auto const m = local_minute(t, offset);
  return m >= kNightStart && m < kNightEnd;
}

// night_transfer_dim's day, with its night where the search reaches it last:
// forward from the end of one night to the end of the next, backward from the
// start of one night to the start of the next
CISTA_CUDA_COMPAT constexpr int night_day(delta_t const t,
                                          int const offset,
                                          bool const fwd) {
  auto const m = static_cast<int>(t) + offset - (fwd ? kNightEnd : kNightStart);
  return (m >= 0 ? m : m - 1439) / 1440;
}

// The wait for the first trip does not count (after_wait()), so a later trip
// past a night can be the better first one: night_transfer_dim seeds each
// start again at the end of every night within the search's bound (backward:
// just before each starts). The day rule keeps them all.
CISTA_CUDA_COMPAT constexpr int after_night(int const t,
                                            int const offset,
                                            bool const fwd) {
  auto const m = local_minute(static_cast<delta_t>(t), offset);
  if (fwd) {
    auto const d = (kNightEnd - m + 1440) % 1440;
    return t + (d == 0 ? 1440 : d);
  }
  return t - (m - kNightStart + 1440) % 1440 - 1;
}

// whether the time between a and b (either order) meets a night
CISTA_CUDA_COMPAT constexpr bool meets_night(delta_t const a,
                                             delta_t const b,
                                             int const offset) {
  auto const from = static_cast<int>(a < b ? a : b);
  auto const to = static_cast<int>(a < b ? b : a);
  auto const m = local_minute(static_cast<delta_t>(from), offset);
  return is_night(static_cast<delta_t>(from), offset) ||
         to - from >= ((kNightStart - m) % 1440 + 1440) % 1440;
}

// The average UTC offset [min] of the start's and the destination's timezone
// at the query's start time.
std::int16_t night_offset(timetable const&, query const&);

}  // namespace nigiri::routing
