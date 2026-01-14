#pragma once

#include "nigiri/routing/query.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}  // namespace nigiri

namespace nigiri::routing {

static constexpr auto const kForward = 0U;
static constexpr auto const kReverse = 1U;
static constexpr auto const kMax = 0U;
static constexpr auto const kMin = 1U;
static constexpr auto const kAux = 2U;
static constexpr auto const kModeOffset = 2U;

enum class saw_type : std::uint8_t {
  kConstant,
  kDay,
  kTrafficDays,
  kTrafficDaysPower
};

static constexpr auto const kChSawType = saw_type::kDay;

struct tooth {
  friend bool operator<(tooth const& a, tooth const& b) {
    auto const mam_diff = a.mam_ - b.mam_;
    auto const remaining_travel_time =
        static_cast<std::int32_t>(b.travel_dur_.count()) - mam_diff;
    return remaining_travel_time > a.travel_dur_.count() ||
           (remaining_travel_time == a.travel_dur_.count() && mam_diff > 0);
  }
  std::int16_t mam_;
  u16_minutes travel_dur_;
  bitfield_idx_t traffic_days_;
};

}  // namespace nigiri::routing
