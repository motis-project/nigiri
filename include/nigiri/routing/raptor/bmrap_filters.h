#pragma once

#include <cstdlib>
#include <optional>
#include <type_traits>
#include <vector>

#include "nigiri/routing/journey.h"
#include "nigiri/routing/raptor/mcraptor.h"
#include "nigiri/rt/run.h"

// Non-transit trade-off filters bmrap_profile.cc applies on top of the
// restricted pareto set, plus related metaprogramming helpers. Split out of
// bmrap_profile.cc's anonymous namespace so bmrap_test.cc can unit-test them
// directly, without a timetable/query.
namespace nigiri::routing {

template <typename T>
struct has_non_transit_dim_t : std::false_type {};
template <typename... Dims>
struct has_non_transit_dim_t<arr_with<Dims...>>
    : std::bool_constant<arr_with<Dims...>::template has<non_transit_dim>()> {
};
template <typename Criteria>
inline constexpr bool kHasNonTransitDim = has_non_transit_dim_t<Criteria>::value;

constexpr bool kFilterNonTransitTradeoff = false;
constexpr bool kFilterStationDedup = false;
// filter 1's rate: a candidate may cost at most this many minutes of EXTRA
// total journey time per minute of non-transit it saves versus its anchor.
constexpr double kNonTransitTradeoffMinutesPerMinute = 5.0;

// Groups `results` by (arrival_time(), transfers_): journeys that only
// differ in how they spend their non-transit time. The group member with
// the shortest travel_time() is the anchor and always survives; any other
// member that walks less is rejected if the extra travel time it costs
// versus the anchor exceeds kNonTransitTradeoffMinutesPerMinute per minute
// of non-transit saved.
// Uses arrival_time() rather than the raw dest_time_ field because
// bmrap_profile.cc calls this mid-scan, where dest_time_ actually holds the
// departure and start_time_ the arrival (opposite of the final,
// reconstructed convention) - arrival_time() picks the later of the two
// values regardless of which field holds it, so grouping stays correct in
// both phases.
inline void filter_non_transit_tradeoff(std::vector<journey> const& results,
                                        std::vector<bool>& rejected) {
  auto handled = std::vector<bool>(results.size(), false);
  for (auto i = std::size_t{0U}; i != results.size(); ++i) {
    if (handled[i] || rejected[i]) {
      continue;
    }
    auto const arr_i = results[i].arrival_time();
    auto group = std::vector<std::size_t>{i};
    for (auto j = i + 1U; j != results.size(); ++j) {
      if (!handled[j] && !rejected[j] &&
          results[j].arrival_time() == arr_i &&
          results[j].transfers_ == results[i].transfers_) {
        group.push_back(j);
      }
    }
    for (auto const g : group) {
      handled[g] = true;
    }
    if (group.size() < 2U) {
      continue;
    }

    auto ref_idx = group.front();
    for (auto const g : group) {
      if (results[g].travel_time() < results[ref_idx].travel_time()) {
        ref_idx = g;
      }
    }
    auto const& ref = results[ref_idx];
    auto const ref_travel_time = ref.travel_time();

    for (auto const idx : group) {
      if (idx == ref_idx) {
        continue;
      }
      auto const& cand = results[idx];
      if (cand.criteria_cost_ >= ref.criteria_cost_) {
        continue;  // not actually walking less than the reference
      }
      auto const saved =
          static_cast<double>(ref.criteria_cost_ - cand.criteria_cost_);
      auto const extra =
          static_cast<double>((cand.travel_time() - ref_travel_time).count());
      if (extra > saved * kNonTransitTradeoffMinutesPerMinute) {
        rejected[idx] = true;
      }
    }
  }
}

enum class bookend { kEntry, kExit };

// The walk immediately before (kEntry) / after (kExit) a reconstructed
// journey's outermost transit leg - a leading/trailing footpath or offset
// leg. Zero if the journey boards/alights right at that end with no bookend
// walk (or has no legs at all, which keeps this total).
template <bookend B>
duration_t bookend_walk(journey const& j) {
  if (j.legs_.empty()) {
    return duration_t{0};
  }
  auto const& l = (B == bookend::kEntry) ? j.legs_.front() : j.legs_.back();
  if (std::holds_alternative<journey::run_enter_exit>(l.uses_)) {
    return duration_t{0};
  }
  return duration_t{std::abs((l.arr_time_ - l.dep_time_).count())};
}

template <bookend B>
std::optional<rt::run> bookend_run(journey const& j) {
  if constexpr (B == bookend::kEntry) {
    for (auto const& l : j.legs_) {
      if (std::holds_alternative<journey::run_enter_exit>(l.uses_)) {
        return std::get<journey::run_enter_exit>(l.uses_).r_;
      }
    }
  } else {
    for (auto it = j.legs_.rbegin(); it != j.legs_.rend(); ++it) {
      if (std::holds_alternative<journey::run_enter_exit>(it->uses_)) {
        return std::get<journey::run_enter_exit>(it->uses_).r_;
      }
    }
  }
  return std::nullopt;
}

// The part of the journey that stays fixed while the bookend walk varies:
// physical ARRIVAL for kEntry (different enter stations, same first leg),
// physical DEPARTURE for kExit (different exit stations, same last leg).
// Uses journey::{arrival,departure}_time() rather than the raw fields since
// a backward (arriveBy) search swaps which of start_time_/dest_time_ holds
// which physical value; the raw fields would compare the wrong, varying end
// for one of the two directions.
template <bookend B>
unixtime_t bookend_anchor_field(journey const& j) {
  return (B == bookend::kEntry) ? j.arrival_time() : j.departure_time();
}

// Groups `results` by (transfers_, bookend_anchor_field<B>, bookend_run<B>)
// and keeps only the longest bookend walk (the anchor) and the shortest,
// rejecting everyone strictly in between. Matching the anchor field (not
// just the shared run) avoids collapsing journeys that share a first/last
// leg but diverge everywhere else.
template <bookend B>
void filter_bookend_variants(std::vector<journey> const& results,
                             std::vector<bool>& rejected) {
  auto handled = std::vector<bool>(results.size(), false);
  for (auto i = std::size_t{0U}; i != results.size(); ++i) {
    if (handled[i] || rejected[i]) {
      continue;
    }
    auto const run_i = bookend_run<B>(results[i]);
    if (!run_i.has_value()) {
      handled[i] = true;
      continue;
    }
    auto const anchor_field_i = bookend_anchor_field<B>(results[i]);
    auto group = std::vector<std::size_t>{i};
    for (auto j = i + 1U; j != results.size(); ++j) {
      if (!handled[j] && !rejected[j] &&
          results[j].transfers_ == results[i].transfers_ &&
          bookend_anchor_field<B>(results[j]) == anchor_field_i &&
          bookend_run<B>(results[j]) == run_i) {
        group.push_back(j);
      }
    }
    for (auto const g : group) {
      handled[g] = true;
    }
    if (group.size() < 2U) {
      continue;
    }

    auto anchor_idx = group.front();
    for (auto const g : group) {
      if (bookend_walk<B>(results[g]) > bookend_walk<B>(results[anchor_idx])) {
        anchor_idx = g;
      }
    }
    auto shortest_idx = group.front();
    for (auto const g : group) {
      if (bookend_walk<B>(results[g]) < bookend_walk<B>(results[shortest_idx])) {
        shortest_idx = g;
      }
    }

    for (auto const g : group) {
      if (g != anchor_idx && g != shortest_idx) {
        rejected[g] = true;
      }
    }
  }
}

// Runs the enabled, applicable filters over `results` without mutating it,
// returning which indices would be rejected. Shared by the final
// mark-and-erase application and n_results()'s preview count.
template <typename Criteria>
std::vector<bool> preview_non_transit_filters(
    std::vector<journey> const& results) {
  auto rejected = std::vector<bool>(results.size(), false);
  if constexpr (kHasNonTransitDim<Criteria>) {
    if constexpr (kFilterNonTransitTradeoff) {
      filter_non_transit_tradeoff(results, rejected);
    }
    if constexpr (kFilterStationDedup) {
      filter_bookend_variants<bookend::kEntry>(results, rejected);
      filter_bookend_variants<bookend::kExit>(results, rejected);
    }
  }
  return rejected;
}

// Whether `Algo`, run with bounds_ always set, still needs a real lb array.
// False if the engine never uses lb at all, or if it declares
// kNeedsLbWhenBounded = false (CPU raptor/basic_mcraptor: bound_prunes() is
// a tighter cutoff once bounded); true otherwise (e.g. gpu_mcraptor, the
// conservative default for an engine with no bounds_-aware fallback). Only
// valid for an Algo this file bounds unconditionally - ping never gets
// set_bounds(), so its need is decided by kUseLowerBounds alone.
template <typename Algo>
constexpr bool bounded_needs_lb() {
  if constexpr (!Algo::kUseLowerBounds) {
    return false;
  } else if constexpr (requires { Algo::kNeedsLbWhenBounded; }) {
    return Algo::kNeedsLbWhenBounded;
  } else {
    return true;
  }
}

}  // namespace nigiri::routing
