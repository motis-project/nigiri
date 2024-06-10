#pragma once

#include "nigiri/routing/journey.h"
#include "nigiri/routing/pareto_set.h"

namespace nigiri::qa {

using rating_t = double;
constexpr auto const kMaxRating = std::numeric_limits<rating_t>::max();
constexpr auto const kMinRating = std::numeric_limits<rating_t>::min();

rating_t rate(pareto_set<nigiri::routing::journey> const& a,
              pareto_set<nigiri::routing::journey> const& b);

}  // namespace nigiri::qa