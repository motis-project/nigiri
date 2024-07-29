#pragma once

#include <algorithm>

#include "cista/reflection/comparable.h"

#include "nigiri/routing/transfer_time_settings.h"
#include "nigiri/types.h"

namespace nigiri {

struct lb_entry {
  CISTA_COMPARABLE()

  static constexpr auto const kUnreachable =
      duration_t{std::numeric_limits<duration_t::rep>::max()};

  location_idx_t target() const { return target_; }

  duration_t duration(routing::transfer_time_settings const& tts) const {
    if (footpath_duration_ == kUnreachable) {
      return trip_duration_;
    } else {
      return std::min(routing::adjusted_transfer_time(tts, footpath_duration_),
                      trip_duration_);
    }
  }

  duration_t footpath_duration() const { return footpath_duration_; }
  duration_t trip_duration() const { return trip_duration_; }

  location_idx_t target_{};
  duration_t footpath_duration_{};
  duration_t trip_duration_{};
};

}  // namespace nigiri
