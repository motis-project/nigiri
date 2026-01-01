#include "nigiri/qa/qa.h"

#include "cista/mmap.h"
#include "cista/serialization.h"

#include <cmath>
#include <numbers>

namespace nigiri::qa {

constexpr auto const kDefaultWeights = criteria_t{1.0, 1.0, 30.0};

double improvement(criteria_t const& a,
                   criteria_t const& b,
                   criteria_t const& weights) {
  auto dist = double{0.0};
  auto impr = double{0.0};

  for (auto i = 0U; i != weights.size(); ++i) {
    auto const weighted_a = a[i] * weights[i];
    auto const weighted_b = b[i] * weights[i];
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

  static constexpr auto const p = double{30.0};
  static constexpr auto const q = double{0.1};

  return std::log2(std::pow(impr, 2) / dist) *
         (std::atan(p * (dist - q)) + std::numbers::pi / 2.0);
}

double min_improvement(criteria_t const& j,
                       vector<criteria_t> const& xjs,
                       std::array<double, 3> const& weights) {
  auto min_impr = kMaxRating;

  for (auto const& xj : xjs) {
    auto const impr = improvement(j, xj, weights);
    if (impr < min_impr) {
      min_impr = impr;
    }
  }

  return min_impr;
}

double set_improvement(vector<criteria_t> const& a,
                       vector<criteria_t> const& b,
                       criteria_t const& weights) {
  auto a_copy = a;
  auto b_copy = b;

  auto impr = double{0.0};

  while (!a_copy.empty()) {
    auto max_impr_a = kMinRating;
    auto max_a = 0U;

    for (auto i = 0U; i != a_copy.size(); ++i) {
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

double rate(vector<criteria_t> const& a, vector<criteria_t> const& b) {
  if (a.empty() && b.empty()) {
    return double{0.0};
  } else if (a.empty()) {
    return kMinRating;
  } else if (b.empty()) {
    return kMaxRating;
  }

  if (a != b) {
    auto const print_set = [](vector<criteria_t> const& x) {
      for (auto const c : x) {
        auto const dep =
            unixtime_t{i32_minutes{static_cast<std::int32_t>(c[0])}};
        auto const arr =
            unixtime_t{i32_minutes{static_cast<std::int32_t>(c[1])}};
        auto const transfers = c[2];
        std::cout << "  " << dep << " - " << arr << " transfers=" << transfers
                  << "\n";
      }
    };

    std::cout << "cmp\n";
    print_set(a);

    std::cout << "ref\n";
    print_set(b);

    std::cout << "\n\n";
  }

  auto const LR = set_improvement(a, b, kDefaultWeights);
  auto const RL = set_improvement(b, a, kDefaultWeights);
  return LR - RL;
}

double rate(pareto_set<nigiri::routing::journey> const& a,
            pareto_set<nigiri::routing::journey> const& b) {
  auto const jc_from_ps = [](auto const& ps) {
    auto jc_vec_ = vector<criteria_t>{};
    for (auto const& j : ps) {
      jc_vec_.emplace_back(
          static_cast<double>(-j.start_time_.time_since_epoch().count()),
          static_cast<double>(j.dest_time_.time_since_epoch().count()),
          static_cast<double>(j.transfers_));
    }
    return std::move(jc_vec_);
  };
  return rate(jc_from_ps(a), jc_from_ps(b));
}

constexpr auto const kMode =
    cista::mode::WITH_INTEGRITY | cista::mode::WITH_STATIC_VERSION;

void benchmark_criteria::write(std::filesystem::path const& p) const {
  auto mmap = cista::mmap{p.string().c_str(), cista::mmap::protection::WRITE};
  auto writer = cista::buf<cista::mmap>(std::move(mmap));
  cista::serialize<kMode>(writer, *this);
}

cista::wrapped<benchmark_criteria> benchmark_criteria::read(
    cista::memory_holder&& mem) {
  return std::visit(
      utl::overloaded{[&](cista::buf<cista::mmap>& b) {
                        auto const ptr = reinterpret_cast<benchmark_criteria*>(
                            &b[cista::data_start(kMode)]);
                        return cista::wrapped{std::move(mem), ptr};
                      },
                      [&](cista::buffer& b) {
                        auto const ptr =
                            cista::deserialize<benchmark_criteria, kMode>(b);
                        return cista::wrapped{std::move(mem), ptr};
                      },
                      [&](cista::byte_buf& b) {
                        auto const ptr =
                            cista::deserialize<benchmark_criteria, kMode>(b);
                        return cista::wrapped{std::move(mem), ptr};
                      }},
      mem);
}

}  // namespace nigiri::qa