#pragma once

#include <cstdint>
#include <vector>

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/limits.h"

namespace nigiri::routing {

// BM-RAPTOR pruning bounds (Delling, Dibbelt, Pajor: "Fast and Exact Public
// Transit Routing with Restricted Pareto Sets", ALENEX'19). at(i, v) is the
// paper's tau_dep^<-(v, i): the worst time (main search direction, delta_t on
// its base day) at stop v that still reaches the target with i trips left.
// budget_: the main search's trip budget.
struct bmrap_bounds {
  bool empty() const { return n_locations_ == 0U; }
  std::size_t size() const {
    return static_cast<std::size_t>(budget_ + 1U) * n_locations_;
  }

  delta_t at(unsigned const rounds_left, std::uint32_t const l) const {
    return lat_[static_cast<std::size_t>(rounds_left) * n_locations_ + l];
  }
  delta_t& at(unsigned const rounds_left, std::uint32_t const l) {
    return lat_[static_cast<std::size_t>(rounds_left) * n_locations_ + l];
  }

  void resize(std::uint32_t const n_locations,
              std::uint8_t const budget,
              delta_t const invalid) {
    n_locations_ = n_locations;
    budget_ = budget;
    lat_.assign(static_cast<std::size_t>(budget + 1U) * n_locations, invalid);
  }

  std::vector<delta_t> lat_;
  std::uint32_t n_locations_{0U};
  std::uint8_t budget_{0U};
  // Row of a label in round k of a search with budget B: staggered (the anchor
  // runs of compute_bounds(), aligned to budget_) budget_ - k, else (a reach
  // matrix: row i = at most i trips) B - k.
  bool staggered_{false};
  // Built on the device: the matrix stays in the building state's slot until
  // the next one of its kind, device_tag_ stamps it; lat_ only on request.
  delta_t const* device_{nullptr};
  std::uint64_t device_tag_{0U};
};

}  // namespace nigiri::routing
