#pragma once

#include <cinttypes>

#include "nigiri/types.h"

namespace nigiri::routing {

static constexpr auto const kMaxTransfers = std::uint8_t{7U};
static constexpr auto const kMaxTravelTime = 1_days;

}  // namespace nigiri::routing
