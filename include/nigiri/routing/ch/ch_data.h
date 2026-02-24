#pragma once

#include <cstdint>
#include "nigiri/routing/query.h"
#include "nigiri/types.h"
#include <limits>

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

static constexpr auto const kChSawType = saw_type::kTrafficDaysPower;
static constexpr auto const kChMaxEdgeTime =
    u16_minutes{routing::kMaxTravelTime.count()};  // TODO
static constexpr auto const kChMaxWaitingTime =
    u16_minutes{1440};  // TODO one day should be sufficient, but this
                        // prolonged avg times?
static constexpr auto const kChDay = u16_minutes{1440};  // TODO

struct tooth {
  bool operator<(tooth const& o) const {
    if (mam_ == o.mam_) {
      return travel_dur_ < o.travel_dur_;
    }
    return mam_ > o.mam_;
  }

  friend bool operator==(tooth const& a, tooth const& b) {
    return a.mam_ == b.mam_ && a.travel_dur_ == b.travel_dur_ &&
           a.traffic_days_ == b.traffic_days_;
  }
  friend std::ostream& operator<<(std::ostream& out, tooth const& a) {
    out << "(" << a.mam_ << "," << a.travel_dur_ << "," << a.traffic_days_
        << ")";
    return out;
  }

  bool dominates(tooth const& a) {
    auto const mam_diff = a.mam_ - mam_;
    auto const remaining_travel_time =
        static_cast<std::int32_t>(travel_dur_.count()) - mam_diff;
    return remaining_travel_time > a.travel_dur_.count() ||
           (remaining_travel_time == a.travel_dur_.count() && mam_diff > 0);
  }

  std::int16_t mam_;
  u16_minutes travel_dur_;
  bitfield_idx_t traffic_days_;
  ch_edge_idx_t start_{ch_edge_idx_t::invalid()};
  transport_idx_t start_transport_{transport_idx_t::invalid()};
  ch_edge_idx_t end_{ch_edge_idx_t::invalid()};
  transport_idx_t end_transport_{transport_idx_t::invalid()};
  std::uint16_t start_idx_{std::numeric_limits<std::uint16_t>::max()};
  std::uint16_t end_idx_{std::numeric_limits<std::uint16_t>::max()};
};

}  // namespace nigiri::routing
