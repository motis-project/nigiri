#include "nigiri/qa/qa.h"

#include "cista/mmap.h"
#include "cista/serialization.h"

#include <cmath>
#include <numbers>

namespace nigiri::qa {

constexpr auto const kDefaultWeights = criteria_t{1.0, 1.0, 30.0};

rating_t improvement(criteria_t const& a,
                     criteria_t const& b,
                     criteria_t const& weights) {

  auto dist = rating_t{0.0};
  auto impr = rating_t{0.0};

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

  static constexpr auto const p = rating_t{30.0};
  static constexpr auto const q = rating_t{0.1};

  return std::log2(std::pow(impr, 2) / dist) *
         (std::atan(p * (dist - q)) + std::numbers::pi / 2.0);
}

rating_t min_improvement(criteria_t const& j,
                         vector<criteria_t> const& xjs,
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

rating_t set_improvement(vector<criteria_t> const& a,
                         vector<criteria_t> const& b,
                         criteria_t const& weights) {
  auto a_copy = a;
  auto b_copy = b;

  auto impr = rating_t{0.0};

  while (!a_copy.empty()) {
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

rating_t rate(vector<criteria_t> const& a, vector<criteria_t> const& b) {
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

rating_t rate(pareto_set<nigiri::routing::journey> const& a,
              pareto_set<nigiri::routing::journey> const& b) {
  auto const jc_from_ps = [](auto const& ps) {
    auto jc_vec_ = vector<criteria_t>{};
    for (auto const& j : ps) {
      jc_vec_.emplace_back(
          static_cast<rating_t>(-j.start_time_.time_since_epoch().count()),
          static_cast<rating_t>(j.dest_time_.time_since_epoch().count()),
          static_cast<rating_t>(j.transfers_));
    }
    return jc_vec_;
  };

  auto const jc_vec_a = jc_from_ps(a);
  auto const jc_vec_b = jc_from_ps(b);
  return rate(jc_vec_a, jc_vec_b);
}

constexpr auto const kMode =
    cista::mode::WITH_INTEGRITY | cista::mode::WITH_STATIC_VERSION;

void benchmark_criteria::write(cista::memory_holder& mem) const {
  std::visit(utl::overloaded{[&](cista::buf<cista::mmap>& writer) {
                               cista::serialize<kMode>(writer, *this);
                             },
                             [&](cista::buffer&) {
                               throw std::runtime_error{"not supported"};
                             },
                             [&](cista::byte_buf& b) {
                               auto writer = cista::buf{std::move(b)};
                               cista::serialize<kMode>(writer, *this);
                               b = std::move(writer.buf_);
                             }},
             mem);
}

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