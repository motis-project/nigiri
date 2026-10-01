#pragma once

#include <cstdint>
#include <vector>

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/limits.h"

namespace nigiri::routing {

// Pruning bounds of the BM-RAPTOR backward pruning search (Delling, Dibbelt,
// Pajor: "Fast and Exact Public Transit Routing with Restricted Pareto Sets",
// ALENEX'19, doi:10.1137/1.9781611975499.5).
//
// at(i, v) is the paper's tau_dep^<-(v, i): the worst time - in the MAIN
// search's direction - at which it may be at stop v and still reach the target
// with i trips left within some anchor's arrival and trip slack (delta_t
// minutes on the main search's base day).
//
// Rows are the monotone prefix over the pruning raptor's round times ("exactly
// k trips" vs the bound's "at most i"), which also makes the rRAPTOR reuse
// across the per-anchor backward runs sound.
//
// budget_ is floor(sigma_tr * K), K = trips of the anchor journey with the most
// trips: the trip budget of the whole main search. A label of round k has
// budget_ - k trips left.
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
  // How a search with a trip budget of its own reads the rows. true: the rows
  // of compute_bounds()' anchor runs are staggered relative to budget_ (a run
  // with budget b sits at rows budget_ - b + t), so row budget_ - k checks
  // every anchor for its own b - k trips left, whatever the reader's budget.
  // false: a single search's reach matrix (row i = at most i trips from the
  // origin), where a reader with budget B has B - k trips left - row B - k,
  // tighter than budget_ - k whenever B < budget_.
  bool staggered_{false};
  // Set by an engine that built this matrix on a device and still holds the
  // copy, so it can copy device-to-device instead of re-uploading `lat_`. 0 =
  // no device copy. Only the issuing engine may interpret it.
  std::uint64_t device_tag_{0U};
};

}  // namespace nigiri::routing
