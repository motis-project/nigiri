#pragma once

#include <cinttypes>

#include "nigiri/types.h"

namespace nigiri::routing {

static constexpr auto const kMaxTransfers = std::uint8_t{14U};
static constexpr auto const kMaxTravelTime = 5_days;
static constexpr auto const kMaxSearchIntervalSize =
    date::days{std::numeric_limits<duration_t::rep>::max() / 1440} -
    (kMaxTravelTime + 2_days);
static constexpr auto const kMaxVias = 2;

// Minutes a walk adds to the non_transit criterion (CPU and GPU mc engines).
// kNonTransitExcess > 0: every walk counts only its part beyond that many
// minutes; kNonTransitBin > 0: then rounded to the nearest multiple of it. Per
// walk, so the criterion stays additive. Both 0: every minute counts.
static constexpr auto const kNonTransitExcess = std::uint16_t{0U};
static constexpr auto const kNonTransitBin = std::uint16_t{0U};

constexpr std::uint16_t counted_non_transit(std::uint16_t const duration) {
  auto const d = duration > kNonTransitExcess
                     ? static_cast<std::uint16_t>(duration - kNonTransitExcess)
                     : std::uint16_t{0U};
  if constexpr (kNonTransitBin == 0U) {
    return d;
  } else {
    return static_cast<std::uint16_t>((d + kNonTransitBin / 2U) /
                                      kNonTransitBin * kNonTransitBin);
  }
}

}  // namespace nigiri::routing
