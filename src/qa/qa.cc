#include "nigiri/qa/qa.h"

#include <cmath>
#include <numbers>

namespace nigiri::qa {

using criteria_t = std::array<rating_t, 3>;
constexpr auto const kDefaultWeights = criteria_t{1.0, 1.0, 30.0};

rating_t improvement(nigiri::routing::journey const& a,
                     nigiri::routing::journey const& b,
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

rating_t min_improvement(nigiri::routing::journey const& j,
                         std::vector<nigiri::routing::journey> const& xjs,
                         std::array<rating_t, 3> const& weights) {
  auto min_impr = kMaxRating;

  for (auto i = 0U; i != xjs.size(); ++i) {
    auto impr = improvement(j, xjs[i], weights);
    if (impr < min_impr) {
      min_impr = impr;
    }
  }

  return min_impr;
}

rating_t set_improvement(pareto_set<nigiri::routing::journey> const& a,
                         pareto_set<nigiri::routing::journey> const& b,
                         criteria_t const& weights) {
  auto a_copy = std::vector<nigiri::routing::journey>{};
  for (auto const& j : a) {
    a_copy.emplace_back(j);
  }
  auto b_copy = std::vector<nigiri::routing::journey>{};
  for (auto const& j : b) {
    b_copy.emplace_back(j);
  }

  auto impr = rating_t{0.0};

  while (a_copy.size() != 0) {
    auto max_impr_a = kMinRating;
    auto max_a = 0U;

    for (auto i = 0U; i != a.size(); ++i) {
      auto min_impr = min_improvement(a_copy[i], b_copy, weights);
      if (min_impr > max_impr_a) {
        max_impr_a = min_impr;
        max_a = i;
      }
    }

    impr += max_impr_a;

    b_copy.emplace_back(a_copy[max_a]);
    a_copy.erase(begin(a_copy) + max_a);
  }

  return impr;
}

rating_t rate(pareto_set<nigiri::routing::journey> const& a,
              pareto_set<nigiri::routing::journey> const& b) {
  if (a.size() == 0 && b.size() == 0) {
    return rating_t{0.0};
  } else if (a.size() == 0) {
    return kMinRating;
  } else if (b.size() == 0) {
    return kMaxRating;
  }

  auto const LR = set_improvement(a, b, kDefaultWeights);
  auto const RL = set_improvement(b, a, kDefaultWeights);
  return LR - RL;
}

}  // namespace nigiri::qa