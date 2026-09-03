#pragma once

#include <cstdint>
#include <vector>

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/limits.h"

namespace nigiri::routing {

// Pruning bounds produced by the BM-RAPTOR backward pruning search
// (Delling, Dibbelt, Pajor: "Fast and Exact Public Transit Routing with
// Restricted Pareto Sets", ALENEX'19, doi:10.1137/1.9781611975499.5).
//
// at(i, v) is the paper's tau_dep^<-(v, i): the last point in time - "last"
// in the MAIN search's direction, i.e. the worst value still acceptable -
// at which the main search may be at stop v and still reach the target with
// i remaining trips without exceeding the arrival slack / trip slack of at
// least one anchor journey. Times are delta_t minutes relative to the same
// base day the main search uses.
//
// Rows are the MONOTONE PREFIX over the pruning raptor's round times:
// round_times_[k][v] is "exactly k trips", the bound needs "at most i
// trips". The prefix is also what makes the rRAPTOR reuse across the
// per-anchor backward runs sound: a run's label that was self-pruned by a
// better label of a smaller round is covered by the prefix of that round.
//
// budget_ is floor(sigma_tr * K) with K = number of trips of the anchor
// journey with the most trips: the trip budget of the whole main search. A
// label produced in round k has budget_ - k trips left.
struct bmrap_bounds {
  bool empty() const { return lat_.empty(); }

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
};

}  // namespace nigiri::routing
