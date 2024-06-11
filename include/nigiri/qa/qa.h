#pragma once

#include "cista/memory_holder.h"

#include "nigiri/routing/journey.h"
#include "nigiri/routing/pareto_set.h"

#include <filesystem>

namespace nigiri::qa {

using rating_t = double;
constexpr auto const kMaxRating = std::numeric_limits<rating_t>::max();
constexpr auto const kMinRating = std::numeric_limits<rating_t>::min();
using criteria_t = std::array<rating_t, 3>;

struct query_criteria {
  std::uint64_t query_idx_;
  std::chrono::milliseconds query_time_;
  vector<criteria_t> jc_;
};

struct benchmark_criteria {
  void write(cista::memory_holder&) const;
  void write(std::filesystem::path const&) const;
  static cista::wrapped<benchmark_criteria> read(cista::memory_holder&&);

  template <std::size_t NMaxTypes>
  constexpr auto static_type_hash(benchmark_criteria const*,
                                  cista::hash_data<NMaxTypes> h) noexcept {
    return h.combine(cista::hash("nigiri::qa::benchmark_criteria"));
  }

  vector<query_criteria> qc_;
};

rating_t rate(vector<criteria_t> const&, vector<criteria_t> const&);

rating_t rate(pareto_set<nigiri::routing::journey> const&,
              pareto_set<nigiri::routing::journey> const&);

}  // namespace nigiri::qa