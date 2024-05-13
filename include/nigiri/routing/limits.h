#pragma once

#include <cinttypes>

#include "nigiri/types.h"

namespace nigiri::routing {

static constexpr auto const kMaxTransfers = std::uint8_t{7U};
static constexpr auto const kMaxTravelTime = 1_days;
constexpr auto const kMaxSearchIntervalSize =
    date::days{std::numeric_limits<duration_t::rep>::max() / 1440};

}  // namespace nigiri::routing
