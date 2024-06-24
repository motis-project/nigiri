#pragma once

#include "cista/memory_holder.h"

#include "nigiri/routing/journey.h"
#include "nigiri/routing/pareto_set.h"

#include <filesystem>

namespace nigiri::qa {

constexpr auto const kMaxRating = std::numeric_limits<double>::max();
constexpr auto const kMinRating = std::numeric_limits<double>::min();
using criteria_t = std::array<double, 3>;

struct query_criteria {
  std::uint64_t query_idx_;
  std::chrono::milliseconds query_time_;
  vector<criteria_t> jc_;
};

struct benchmark_criteria {
  void write(std::filesystem::path const&) const;
  static cista::wrapped<benchmark_criteria> read(cista::memory_holder&&);

  vector<query_criteria> qc_;
};

double rate(vector<criteria_t> const&, vector<criteria_t> const&);

double rate(pareto_set<nigiri::routing::journey> const&,
            pareto_set<nigiri::routing::journey> const&);

}  // namespace nigiri::qa