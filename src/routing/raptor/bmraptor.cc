#include "nigiri/routing/raptor/bmraptor.h"

#include <algorithm>
#include <chrono>
#include <vector>

#include "utl/erase_if.h"
#include "utl/verify.h"

#include "nigiri/routing/raptor/bmrap_bounds.h"
#include "nigiri/routing/raptor/bmrap_common.h"
#include "nigiri/routing/raptor/mcraptor.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/routing/search.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

namespace {

using namespace bmrap_detail;

template <direction SearchDir, typename Criteria>
routing_result bmrap(timetable const& tt,
                     rt_timetable const* rtt,
                     search_state& s_state,
                     basic_mcraptor_state<Criteria>& r_state,
                     query q,
                     std::optional<std::chrono::seconds> const timeout) {
  constexpr auto const kFwd = (SearchDir == direction::kForward);
  using main_algo_t = basic_mcraptor<SearchDir, Criteria, /*RangeReuse=*/true>;

  q.sanitize(tt);
  utl::verify(mcraptor_supported(q, rtt),
              "bmraptor: query not supported by the mcraptor main search");

  auto const t0 = std::chrono::steady_clock::now();

  // ====
  // PHASE 1: forward pruning search -> anchor pareto set
  // ====
  auto prune_state = raptor_state{};
  auto anchor_result =
      run_anchor_search<SearchDir>(tt, rtt, s_state, prune_state, q, timeout);
  auto const t_anchor = std::chrono::steady_clock::now();

  auto anchors = std::vector<anchor>{};
  for (auto const& j : *anchor_result.journeys_) {
    anchors.push_back({j.start_time_, j.dest_time_,
                       static_cast<std::uint8_t>(j.transfers_ + 1U)});
  }
  auto const anchor_interval = anchor_result.interval_;
  auto const anchor_stats = anchor_result.search_stats_;
  auto algo_stats = anchor_result.algo_stats_;

  if (anchors.empty() || anchor_interval.size() == duration_t{0}) {
    // nothing to restrict - the anchor set already is the answer
    anchor_result.search_stats_.execute_time_ =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0);
    return anchor_result;
  }

  // ====
  // PHASE 1b: close the anchor profile past the window
  // ====
  auto const anchor_margin = close_anchor_profile<SearchDir>(
      tt, rtt, q, anchor_interval, prune_state, timeout, anchors);
  auto max_trips = std::uint8_t{0U};
  for (auto const& a : anchors) {
    max_trips = std::max(max_trips, a.trips_);
  }

  // ====
  // PHASE 2: backward pruning search -> tau_dep^<-(v, i)
  // ====
  // ONE bound matrix for the whole window - the range approximation the
  // profile driver replaces with one matrix per step (see bmraptor.h). Same
  // base day the main search derives from its (now fixed) window, so the
  // delta_t values of both searches are comparable.
  auto const base =
      day_idx_t{std::chrono::duration_cast<date::days>(
                    std::chrono::round<std::chrono::days>(
                        anchor_interval.from_ +
                        ((anchor_interval.to_ - anchor_interval.from_) / 2)) -
                    tt.internal_interval().from_)
                    .count()};
  auto const budget = trip_budget(
      max_trips, static_cast<std::uint8_t>(std::min<unsigned>(
                     q.max_transfers_ + 1U, kMaxTransfers + 1U)));
  // the main search never produces a label past the near end of its window
  auto const horizon = kFwd ? anchor_interval.from_ - duration_t{1}
                            : anchor_interval.to_ + duration_t{1};

  auto prune_stats = raptor_stats{};
  auto const bounds =
      rtt == nullptr
          ? compute_bounds<SearchDir, false>(tt, rtt, q, anchors, base, horizon,
                                             budget, prune_state, prune_stats)
          : compute_bounds<SearchDir, true>(tt, rtt, q, anchors, base, horizon,
                                            budget, prune_state, prune_stats);
  auto const t_prune = std::chrono::steady_clock::now();

  // ====
  // PHASE 3: bounded range McRAPTOR over the anchor window
  // ====
  auto qm = q;
  qm.start_time_ = anchor_interval;
  qm.min_connection_count_ = 0U;  // window is fixed: no interval extension
  qm.extend_interval_earlier_ = false;
  qm.extend_interval_later_ = false;
  qm.max_transfers_ = static_cast<std::uint8_t>(budget - 1U);

  auto s = search<SearchDir, main_algo_t>{tt,      rtt,           s_state,
                                          r_state, std::move(qm), timeout};
  s.algo().set_bounds(&bounds);
  // PER-DEPARTURE TRIP BUDGET. floor(sigma_tr * K) with K taken over the whole
  // window would make a departure's result depend on the window it was queried
  // in: one high-transfer anchor anywhere in the range lifts the budget - and
  // with it the transfer limit and the bound row - for every other departure.
  // K is therefore evaluated per departure, over the anchors available AT that
  // departure, exactly as A(J) is in the restriction below.
  s.max_transfers_fn_ = [&anchors, budget](unixtime_t const d) {
    auto k = std::uint8_t{0U};
    for (auto const& a : anchors) {
      if (!is_better<SearchDir>(a.anchored_, d)) {
        k = std::max(k, a.trips_);
      }
    }
    // no anchor left at this departure: nothing justifies a restriction, fall
    // back to the window budget rather than over-pruning
    return static_cast<std::uint8_t>(
        (k == 0U ? budget : trip_budget(k, budget)) - 1U);
  };
  auto result = s.execute();
  auto const t_main = std::chrono::steady_clock::now();

  // ====
  // restrict to J_R
  // ====
  // This is where the per-departure exactness the window-wide bound matrix
  // gives up comes back: A(J) is looked up for J's OWN departure, so a journey
  // the loose bound let through is dropped here unless its own anchor
  // justifies it. Anchors are their own A(J), which also covers journeys the
  // main search did not produce (enrich_with_slow_direct).
  auto n_restricted = std::uint64_t{0U};
  utl::erase_if(s_state.results_, [&](journey const& j) {
    auto const drop = outside_restriction<SearchDir>(
        anchors, j.start_time_, j.dest_time_,
        static_cast<unsigned>(j.transfers_) + 1U);
    n_restricted += drop ? 1U : 0U;
    return drop;
  });

  // ====
  // stats
  // ====
  auto const ms = [](auto const a, auto const b) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count());
  };
  for (auto const& [k, v] : prune_stats.to_map()) {
    algo_stats["prune_" + k] = v;
  }
  for (auto const& [k, v] : result.algo_stats_) {
    algo_stats["main_" + k] = v;
  }
  algo_stats["bmrap_ms_anchor"] = ms(t0, t_anchor);
  algo_stats["bmrap_ms_prune"] = ms(t_anchor, t_prune);
  algo_stats["bmrap_ms_main"] = ms(t_prune, t_main);
  algo_stats["bmrap_anchors"] = anchors.size();
  algo_stats["bmrap_anchor_margin"] =
      static_cast<std::uint64_t>(anchor_margin.count());
  algo_stats["bmrap_trip_budget"] = budget;
  algo_stats["bmrap_restricted_away"] = n_restricted;
  result.algo_stats_ = std::move(algo_stats);
  result.search_stats_.lb_time_ += anchor_stats.lb_time_;
  result.search_stats_.n_execute_fwd_ += anchor_stats.n_execute_fwd_;
  result.search_stats_.n_execute_bwd_ += anchor_stats.n_execute_bwd_;
  result.search_stats_.execute_time_ =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - t0);
  return result;
}

}  // namespace

template <typename Criteria>
routing_result bmrap_range_search(
    timetable const& tt,
    rt_timetable const* rtt,
    search_state& s_state,
    basic_mcraptor_state<Criteria>& algo_state,
    query q,
    direction const search_dir,
    std::optional<std::chrono::seconds> const timeout) {
  return search_dir == direction::kForward
             ? bmrap<direction::kForward>(tt, rtt, s_state, algo_state,
                                          std::move(q), timeout)
             : bmrap<direction::kBackward>(tt, rtt, s_state, algo_state,
                                           std::move(q), timeout);
}

// Same criteria configurations as bmrap_profile_search.
#define NIGIRI_BMRAP_INSTANTIATE(C)                                         \
  template routing_result bmrap_range_search<C>(                            \
      timetable const&, rt_timetable const*, search_state&,                 \
      basic_mcraptor_state<C>&, query, direction,                           \
      std::optional<std::chrono::seconds>);

NIGIRI_BMRAP_INSTANTIATE(arr_criteria)
NIGIRI_BMRAP_INSTANTIATE(arr_non_transit_criteria)
NIGIRI_BMRAP_INSTANTIATE(arr_mode_filter_criteria)
NIGIRI_BMRAP_INSTANTIATE(arr_mode_switches_criteria)
NIGIRI_BMRAP_INSTANTIATE(arr_non_transit_mode_filter_criteria)
NIGIRI_BMRAP_INSTANTIATE(arr_non_transit_mode_switches_criteria)
NIGIRI_BMRAP_INSTANTIATE(arr_mode_filter_mode_switches_criteria)
NIGIRI_BMRAP_INSTANTIATE(arr_non_transit_mode_filter_mode_switches_criteria)

#undef NIGIRI_BMRAP_INSTANTIATE

}  // namespace nigiri::routing
