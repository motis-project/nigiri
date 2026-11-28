#pragma once

// The restricted Pareto set of BM-RAPTOR (bmrap_profile.cc): anchors, their
// trip budget and deadline, and the restriction they define.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <utility>
#include <vector>

#include "utl/helpers/algorithm.h"

#include "nigiri/routing/limits.h"
#include "nigiri/routing/raptor/bmrap_bounds.h"
#include "nigiri/types.h"

namespace nigiri::routing {

template <direction Dir>
constexpr bool is_better(auto const a, auto const b) {
  return Dir == direction::kForward ? a < b : a > b;
}

// sigma_tr of the paper: the extra trips of a budget, clamped
inline constexpr auto kTripSlack = 1.5;
inline constexpr auto kTripSlackMin = 1.0;
inline constexpr auto kTripSlackCap = 2.0;

// a journey of J_A: `anchored_` the query-side time, `found_` the other end
struct anchor {
  unixtime_t anchored_, found_;
  std::uint8_t trips_;
};

// Latest arrival of a journey with A(J) = a. The single definition for bounds
// and restriction, which must agree.
inline unixtime_t anchor_deadline(anchor const& a) {
  // relax the magnitude (negative backward)
  auto const signed_travel = (a.found_ - a.anchored_).count();
  auto const sign = signed_travel < 0 ? -1 : 1;
  auto const relaxed =
      std::llround(relax_arr(static_cast<double>(std::abs(signed_travel))));
  return a.anchored_ + i32_minutes{static_cast<std::int32_t>(sign * relaxed)};
}

// floor(sigma_tr * |J|), clamped to `max`
inline std::uint8_t trip_budget(std::uint8_t const trips,
                                std::uint8_t const max) {
  auto const extra =
      std::clamp(trips * (kTripSlack - 1.0), kTripSlackMin, kTripSlackCap);
  return static_cast<std::uint8_t>(
      std::clamp<double>(std::floor(trips + extra), trips, max));
}

// A(J): of the anchors available at `anchored` (anchored no earlier) with at
// most `trips` trips, the earliest arriving, ties by fewest trips. On the
// (arrival, trips) front this is the paper's "most trips <= |J|"; on the
// available slice of the (dep, arr, trips) set that one can arrive later.
template <direction SearchDir>
anchor const* anchor_of(std::vector<anchor> const& anchors,
                        unixtime_t const anchored,
                        unsigned const trips) {
  anchor const* best = nullptr;
  for (auto const& a : anchors) {
    if (a.trips_ > trips || is_better<SearchDir>(a.anchored_, anchored)) {
      continue;
    }
    if (best == nullptr || is_better<SearchDir>(a.found_, best->found_) ||
        (a.found_ == best->found_ && a.trips_ < best->trips_)) {
      best = &a;
    }
  }
  return best;
}

// J_R: an anchor, or within A(J)'s trip budget and deadline
template <direction SearchDir>
bool outside_restriction(std::vector<anchor> const& anchors,
                         unixtime_t const anchored,
                         unixtime_t const found,
                         unsigned const trips) {
  if (utl::any_of(anchors, [&](anchor const& a) {
        return a.trips_ == trips && a.anchored_ == anchored &&
               a.found_ == found;
      })) {
    return false;
  }
  auto const* const a = anchor_of<SearchDir>(anchors, anchored, trips);
  return a == nullptr ||
         trips > trip_budget(a->trips_, std::uint8_t{kMaxTransfers + 1U}) ||
         is_better<SearchDir>(anchor_deadline(*a), found);
}

// tau_dep^<-(v, i): the paper's reverse search per anchor from its deadline,
// capped at its trip budget (compute_bounds() in bmrap_profile.cc). These are
// its runs before reduction: per anchor its deadline and trip budget. The
// paper's cap: a journey with as many trips as an earlier-arriving anchor has
// that one (or a better one) as A(J). A journey departing after that anchor is
// found by a later step, which no longer has it - if the anchor departs at or
// after the step (horizon + 1): the pong can move one before it, and the next
// step starts after the step's first departure.
using bound_run = std::pair<unixtime_t, std::uint8_t>;
template <direction SearchDir>
std::vector<bound_run> bound_runs(std::vector<anchor> const& anchors,
                                  unixtime_t const horizon,
                                  std::uint8_t const budget) {
  auto runs = std::vector<bound_run>{};
  runs.reserve(anchors.size());
  for (auto const& a : anchors) {
    auto b = trip_budget(a.trips_, budget);
    for (auto const& x : anchors) {
      if (x.trips_ > a.trips_ && is_better<SearchDir>(x.found_, a.found_) &&
          is_better<SearchDir>(horizon, x.anchored_)) {
        b = std::min(b, static_cast<std::uint8_t>(x.trips_ - 1U));
      }
    }
    runs.emplace_back(anchor_deadline(a), b);
  }
  return runs;
}

}  // namespace nigiri::routing
