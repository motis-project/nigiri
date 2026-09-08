#include "nigiri/routing/raptor/bmraptor.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <variant>
#include <vector>

#include "utl/erase_if.h"
#include "utl/helpers/algorithm.h"
#include "utl/verify.h"

#include "nigiri/routing/raptor/bmrap_bounds.h"
#include "nigiri/routing/raptor/bmrap_common.h"
#include "nigiri/routing/raptor/mcraptor.h"
#include "nigiri/routing/raptor/pong.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/routing/raptor_search.h"
#include "nigiri/routing/start_times.h"
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
  auto max_trips = std::uint8_t{0U};
  for (auto const& j : *anchor_result.journeys_) {
    auto const trips = static_cast<std::uint8_t>(j.transfers_ + 1U);
    anchors.push_back({j.start_time_, j.dest_time_, trips});
    max_trips = std::max(max_trips, trips);
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
  max_trips = 0U;
  for (auto const& a : anchors) {
    max_trips = std::max(max_trips, a.trips_);
  }

  // ====
  // PHASE 2: backward pruning search -> tau_dep^<-(v, i)
  // ====
  // same base day the main search derives from its (now fixed) window, so
  // the delta_t values of both searches are comparable
  auto const base =
      day_idx_t{std::chrono::duration_cast<date::days>(
                    std::chrono::round<std::chrono::days>(
                        anchor_interval.from_ + ((anchor_interval.to_ -
                                                  anchor_interval.from_) /
                                                 2)) -
                    tt.internal_interval().from_)
                    .count()};
  auto const budget = trip_budget(
      max_trips, static_cast<std::uint8_t>(
                     std::min<unsigned>(q.max_transfers_ + 1U,
                                        kMaxTransfers + 1U)));
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

  auto s = search<SearchDir, main_algo_t>{tt,       rtt,           s_state,
                                          r_state,  std::move(qm), timeout};
  if (!get_slack().no_bounds_) {
    s.algo().set_bounds(&bounds);
    // PER-DEPARTURE TRIP BUDGET. floor(sigma_tr * K) with K taken over the
    // whole window would make a departure's result depend on the window it
    // was queried in: one high-transfer anchor anywhere in the range lifts
    // the budget - and with it the transfer limit and the bound row - for
    // every other departure, so the same departure answers differently
    // under a 1 h and an 8 h searchWindow. K is therefore evaluated per
    // departure, over the anchors available AT that departure, exactly as
    // A(J) is in the restriction below.
    s.max_transfers_fn_ = [&anchors, budget](unixtime_t const d) {
      auto k = std::uint8_t{0U};
      for (auto const& a : anchors) {
        if (!(kFwd ? a.anchored_ < d : a.anchored_ > d)) {
          k = std::max(k, a.trips_);
        }
      }
      // no anchor left at this departure: nothing justifies a restriction,
      // fall back to the window budget rather than over-pruning
      return static_cast<std::uint8_t>(
          (k == 0U ? budget : trip_budget(k, budget)) - 1U);
    };
  }
  auto result = s.execute();
  auto const t_main = std::chrono::steady_clock::now();

  // ====
  // restrict to J_R
  // ====
  // This is where the per-departure exactness the single window-wide bound
  // matrix gives up (see compute_bounds) comes back: A(J) is looked up for
  // J's OWN departure, so a journey the loose bound let through is dropped
  // here unless its own anchor justifies it.
  auto const is_anchor = [&](journey const& j) {
    return utl::any_of(anchors, [&](anchor const& a) {
      return a.trips_ == j.transfers_ + 1U && a.anchored_ == j.start_time_ &&
             a.found_ == j.dest_time_;
    });
  };
  auto n_restricted = std::uint64_t{0U};
  utl::erase_if(s_state.results_, [&](journey const& j) {
    if (get_slack().no_restrict_) {
      return false;
    }
    // an anchor journey is its own A(J) and therefore always in J_R; the
    // explicit check also covers journeys that were not produced by the
    // main search at all (enrich_with_slow_direct)
    if (is_anchor(j)) {
      return false;
    }
    auto const trips = static_cast<unsigned>(j.transfers_) + 1U;
    auto const* a = anchor_of<SearchDir>(anchors, j.start_time_, trips);
    if (a == nullptr) {
      ++n_restricted;
      return true;
    }
    // Same deadline the bounds were built from, so the pruning and the
    // filter cannot disagree - see anchor_deadline().
    auto const drop =
        trips > trip_budget(a->trips_, std::uint8_t{kMaxTransfers + 1U}) ||
        misses_deadline<SearchDir>(*a, j.dest_time_);
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

template <typename AlgoState>
routing_result bmrap_search(timetable const& tt,
                            rt_timetable const* rtt,
                            search_state& s_state,
                            AlgoState& algo_state,
                            query q,
                            direction const search_dir,
                            std::optional<std::chrono::seconds> const timeout) {
  if (search_dir == direction::kForward) {
    return bmrap<direction::kForward>(tt, rtt, s_state, algo_state,
                                      std::move(q), timeout);
  } else {
    return bmrap<direction::kBackward>(tt, rtt, s_state, algo_state,
                                       std::move(q), timeout);
  }
}

template routing_result bmrap_search(timetable const&,
                                     rt_timetable const*,
                                     search_state&,
                                     mcraptor_cost_state&,
                                     query,
                                     direction,
                                     std::optional<std::chrono::seconds>);

template routing_result bmrap_search(timetable const&,
                                     rt_timetable const*,
                                     search_state&,
                                     mcraptor_walk_state&,
                                     query,
                                     direction,
                                     std::optional<std::chrono::seconds>);

template routing_result bmrap_search(timetable const&,
                                     rt_timetable const*,
                                     search_state&,
                                     mcraptor_air_state&,
                                     query,
                                     direction,
                                     std::optional<std::chrono::seconds>);

template routing_result bmrap_search(timetable const&,
                                     rt_timetable const*,
                                     search_state&,
                                     mcraptor_walk_air_state&,
                                     query,
                                     direction,
                                     std::optional<std::chrono::seconds>);

template routing_result bmrap_search(timetable const&,
                                     rt_timetable const*,
                                     search_state&,
                                     mcraptor_clasz_state&,
                                     query,
                                     direction,
                                     std::optional<std::chrono::seconds>);

template routing_result bmrap_search(timetable const&,
                                     rt_timetable const*,
                                     search_state&,
                                     mcraptor_walk_clasz_state&,
                                     query,
                                     direction,
                                     std::optional<std::chrono::seconds>);
template routing_result bmrap_search(
    timetable const&, rt_timetable const*, search_state&, mcraptor_air_clasz_state&, query,
    direction, std::optional<std::chrono::seconds>);

template routing_result bmrap_search(
    timetable const&, rt_timetable const*, search_state&, mcraptor_walk_air_clasz_state&, query,
    direction, std::optional<std::chrono::seconds>);


}  // namespace nigiri::routing
