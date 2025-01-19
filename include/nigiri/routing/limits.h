#pragma once

#include <cinttypes>

#include "nigiri/types.h"

namespace nigiri::routing {

constexpr auto const kMaxTransfers = std::uint8_t{5U};
constexpr auto const kMaxTravelTime = 5_days;
constexpr auto const kMaxSearchIntervalSize =
    date::days{std::numeric_limits<duration_t::rep>::max() / 1440} -
    (kMaxTravelTime + 2_days);
constexpr auto const kMaxVias = 2;

}  // namespace nigiri::routing
