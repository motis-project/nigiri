#pragma once

#include <cmath>
#include "nigiri/routing/journey.h"
#include "nigiri/routing/pareto_set.h"
#include <numbers>

namespace nigiri::qa {

using rating_t = double;
using criteria_t = std::array<rating_t, 3>;
constexpr auto const kMaxRating = std::numeric_limits<rating_t>::max();
constexpr auto const kMinRating = std::numeric_limits<rating_t>::min();
constexpr auto const kDefaultWeights = criteria_t{1.0, 1.0, 30.0};

static rating_t improvement(nigiri::routing::journey const& a,
                            nigiri::routing::journey b,
                            criteria_t const& weights) {
  auto const criteria = [](auto const& j) {
    return criteria_t{
        static_cast<rating_t>(-j.start_time_.time_since_epoch().count()),
        static_cast<rating_t>(j.dest_time_.time_since_epoch().count()),
        static_cast<rating_t>(j.transfers_)};
  };

  auto const crit_a = criteria(a);
  auto const crit_b = criteria(b);

  auto dist = rating_t{0.0};
  auto impr = rating_t{0.0};

  for (auto i = 0U; i != weights.size(); ++i) {
    auto const weighted_a = crit_a[i] * weights[i];
    auto const weighted_b = crit_b[i] * weights[i];
    auto const crit_dist = weighted_a - weighted_b;

    dist += std::pow(crit_dist, 2);
    if (crit_dist < 0) {
      impr += std::pow(crit_dist, 2);
    }
  }

  dist = std::sqrt(dist);
  impr = std::sqrt(impr);

  if (impr == 0.0) {
    return 0.0;
  }

  static constexpr auto const p = 30.0;
  static constexpr auto const q = 0.1;

  return std::log2(std::pow(impr, 2) / dist) *
         (std::atan(p * (dist - q)) + std::numbers::pi / 2.0);
}

static std::pair<rating_t, nigiri::routing::journey const*> min_improvement(
    nigiri::routing::journey const& j,
    pareto_set<nigiri::routing::journey> const& xjs,
    std::array<rating_t, 3> const& weights) {
  auto min_impr = kMaxRating;
  nigiri::routing::journey const* min = nullptr;

  for (auto const& x : xjs) {
    auto impr = improvement(j, x, weights);
    if (impr < min_impr) {
      min_impr = impr;
      min = &x;
    }
  }

  return {min_impr, min};
}

static rating_t set_improvement(pareto_set<nigiri::routing::journey> const& a,
                                pareto_set<nigiri::routing::journey> const& b,
                                criteria_t const& weights) {
  if (a.size() == 0 && b.size() == 0) {
    return 0.0;
  } else if (a.size() == 0) {
    return kMinRating;
  } else if (b.size() == 0) {
    return kMaxRating;
  }

  auto a_copy = a;
  auto b_copy = b;
  auto impr = 0.0;

  while (a_copy.size() != 0) {
    auto max_impr_a = kMinRating;
  }
}

static rating_t rate(pareto_set<nigiri::routing::journey> const& a,
                     pareto_set<nigiri::routing::journey> const& b) {
  auto const LR = set_improvement(a, b, kDefaultWeights);
  auto const RL = set_improvement(b, a, kDefaultWeights);
  return LR - RL;
}

}  // namespace nigiri::qa