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
static constexpr auto const kModeOffset = 2U;

struct ch_label {
  using dist_t = std::uint16_t;
  friend bool operator>(ch_label const& a, ch_label const& b) {
    return a.d_[a.dir_ / kModeOffset] > b.d_[b.dir_ / kModeOffset];
  }
  location_idx_t l_;
  std::array<dist_t, 2> d_;
  std::uint8_t dir_;
};

struct ch_get_bucket {
  ch_label::dist_t operator()(ch_label const& l) const {
    return l.d_[l.dir_ / kModeOffset];
  }
};

struct ch_dist {
  using dist_t = std::uint16_t;
  std::array<dist_t, 2> d_{std::numeric_limits<ch_dist::dist_t>::max(),
                           std::numeric_limits<ch_dist::dist_t>::max()};
};

void obtain_relevant_stops(timetable const& tt,
                        routing::query const& q,
                        profile_idx_t const prf_idx,
                        bitvec& relevant_stops);

}  // namespace nigiri::routing
