#pragma once

// Pieces shared by the two BM-RAPTOR drivers: the RANGE one
// (bmraptor.cc - one bound matrix for the whole departure window) and the
// PROFILE one (bmrap_profile.cc - a full single-departure BM-RAPTOR per
// step, which is what the paper actually describes). Keeping the slack
// configuration, the anchor definition and the backward pruning search in
// one place is what makes the two comparable: any difference between them
// is then the search structure, not a drifted definition.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <variant>
#include <vector>

#include "utl/helpers/algorithm.h"

#include "nigiri/routing/journey.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/bmrap_bounds.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/routing/raptor/pong.h"
#include "nigiri/routing/raptor/raptor_stats.h"
#include "nigiri/routing/raptor_search.h"
#include "nigiri/routing/search.h"
#include "nigiri/routing/start_times.h"
#include "nigiri/routing/transfer_time_settings.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

namespace nigiri::routing::bmrap_detail {

constexpr auto const kVias = via_offset_t{0U};


// sigma_arr / sigma_tr of the paper. 1.25 each is the configuration
// BM-RAPTOR is usually evaluated with (Sauer'24).
struct slack_cfg {
  double arr_{1.25};
  double trip_{1.25};
  // Fixed, ADDITIVE arrival slack in minutes - an alternative to the
  // multiplicative sigma_arr above, both applied to the same reference
  // quantity (see relax_arr()): >= 0 switches to "at most N minutes worse
  // than the anchor implies", < 0 (the default) keeps sigma_arr.
  double arr_fixed_min_{-1.0};
  // The paper's sigma_tr is ADDITIVE (a number of extra trips, Eq. 3.1);
  // sigma_tr = 1.25 above is the multiplicative convention (Sauer'24).
  // >= 0 switches to "at most N trips more than the anchor".
  double trip_fixed_{-1.0};
  // validation switches: run the same three phases but skip the pruning /
  // the final restriction, so a mismatch can be attributed to one of them
  bool no_bounds_{false};
  bool no_restrict_{false};
};

inline slack_cfg const& get_slack() {
  static auto const cfg = [] {
    auto c = slack_cfg{};
    if (auto const* v = std::getenv("NIGIRI_BMRAP_ARR_SLACK"); v != nullptr) {
      c.arr_ = std::max(1.0, std::atof(v));
    }
    if (auto const* v = std::getenv("NIGIRI_BMRAP_TRIP_SLACK"); v != nullptr) {
      c.trip_ = std::max(1.0, std::atof(v));
    }
    if (auto const* v = std::getenv("NIGIRI_BMRAP_ARR_SLACK_MIN");
        v != nullptr) {
      c.arr_fixed_min_ = std::max(0.0, std::atof(v));
    }
    if (auto const* v = std::getenv("NIGIRI_BMRAP_TRIP_SLACK_ADD");
        v != nullptr) {
      c.trip_fixed_ = std::max(0.0, std::atof(v));
    }
    c.no_bounds_ = std::getenv("NIGIRI_BMRAP_NO_BOUNDS") != nullptr;
    c.no_restrict_ = std::getenv("NIGIRI_BMRAP_NO_RESTRICT") != nullptr;
    return c;
  }();
  return cfg;
}

// Applies the configured arrival slack to a reference duration (minutes):
// the paper's sigma_arr * reference, or a fixed number of minutes added to
// it instead, per NIGIRI_BMRAP_ARR_SLACK_MIN. Both call sites relax the
// SAME reference quantity - (anchor arrival - this journey's own
// departure), the paper's tau_arr(A(J)) - tau_dep - so the two slack modes
// stay directly comparable: sigma_arr scales it, the fixed variant pads it
// by a constant number of minutes regardless of how long that reference
// already is.
inline double relax_arr(double const reference_minutes) {
  auto const& cfg = get_slack();
  return cfg.arr_fixed_min_ >= 0.0 ? reference_minutes + cfg.arr_fixed_min_
                                   : reference_minutes * cfg.arr_;
}

// One journey of the anchor pareto set J_A. `anchored_` is the query-side
// time (the departure for a forward query, the arrival for a backward one),
// `found_` the time the search produced on the other side - the same
// convention journey::start_time_ / dest_time_ use.
struct anchor {
  unixtime_t anchored_, found_;
  std::uint8_t trips_;
};

// The absolute deadline an anchor implies: the latest time a journey
// anchored to it may arrive. Computed ONCE, here, and used by both sides of
// the restriction - the backward pruning searches start from it, and the
// final filter tests against it. That is what keeps them consistent: a
// multiplicative sigma_arr has no canonical origin to scale from, so if the
// bounds relaxed "from the anchor's departure" while the filter asked
// "travel(J) <= sigma * (arr(A) - dep(J))", the two would disagree by
// (sigma - 1) * (dep(A) - dep(J)) and the bounds would prune journeys the
// restriction keeps. With one shared deadline the question does not arise
// (and with an additive sigma_arr the origin cancels anyway).
inline unixtime_t anchor_deadline(anchor const& a) {
  auto const travel = static_cast<double>((a.found_ - a.anchored_).count());
  return a.anchored_ +
         i32_minutes{static_cast<std::int32_t>(std::llround(relax_arr(travel)))};
}

// Does a journey arriving (departing, for a backward query) at `arrival`
// blow the deadline its anchor implies?
template <direction SearchDir>
bool misses_deadline(anchor const& a, unixtime_t const arrival) {
  auto const d = anchor_deadline(a);
  return SearchDir == direction::kForward ? arrival > d : arrival < d;
}

// the anchor's slack-relaxed trip budget floor(sigma_tr * |J|), clamped to
// what the query and the round-times matrix allow
inline std::uint8_t trip_budget(std::uint8_t const trips,
                               std::uint8_t const max) {
  auto const& cfg = get_slack();
  auto const b = cfg.trip_fixed_ >= 0.0
                     ? static_cast<double>(trips) + cfg.trip_fixed_
                     : static_cast<double>(trips) * cfg.trip_;
  return static_cast<std::uint8_t>(
      std::clamp<double>(std::floor(b), trips, max));
}

// the anchor journey A(J) of a journey with `trips` trips departing
// (arriving, for a backward query) at `anchored`: among the anchors that are
// still AVAILABLE at that time (anchored no earlier than J - a traveller
// departing at J's departure can still take an anchor departing later) and
// need AT MOST as many trips as J, the one with the EARLIEST arrival.
//
// This is the paper's "highest trip count <= |J|" rule on the raw anchor
// Pareto set (arrival, trips) alone is monotone there - a higher-trip point
// only survives dominance by arriving strictly earlier, so "most trips"
// and "earliest arrival" agree. They stop agreeing once the set is cut down
// to "available at d": that slice is a slice of the full (dep, arr, trips)
// Pareto set, and within it a later-departing, higher-trip anchor can still
// have a WORSE arrival than an earlier-departing, lower-trip one (both
// survive globally on the departure axis, but only one is the better
// reference at a shared departure). "Most trips" would then pick the worse
// arrival; "earliest arrival" always picks the objectively best alternative
// achievable with no more trips than J - trips_budget()/the arrival-slack
// check compare J against a reference that cannot be beaten "for free" at
// its own departure, which is the comparison the restriction actually
// wants. It also plays better with the phase 1b window closure: closing the
// anchor profile up to the best arrival within reach is enough to make
// "earliest arrival with trips <= T" complete, whereas "most trips" has no
// natural completion bound (the highest-trip anchor's departure is
// unrelated to arrival time).
template <direction SearchDir>
anchor const* anchor_of(std::vector<anchor> const& anchors,
                        unixtime_t const anchored,
                        unsigned const trips) {
  constexpr auto const kFwd = (SearchDir == direction::kForward);
  auto const is_better = [](unixtime_t const a, unixtime_t const b) {
    return kFwd ? a < b : a > b;
  };
  anchor const* best = nullptr;
  for (auto const& a : anchors) {
    if (a.trips_ > trips || is_better(a.anchored_, anchored)) {
      continue;
    }
    if (best == nullptr || is_better(a.found_, best->found_)) {
      best = &a;
    }
  }
  return best;
}

// PHASE 1: forward pruning search. PONG is the fastest engine for the
// two-criteria range problem, but it only covers the two "good direction"
// interval-extension configurations (see the table in the motis routing
// endpoint); rRAPTOR answers the rest. Both return the same anchor set.
template <direction SearchDir>
routing_result run_anchor_search(timetable const& tt,
                                 rt_timetable const* rtt,
                                 search_state& s_state,
                                 raptor_state& r_state,
                                 query const& q,
                                 std::optional<std::chrono::seconds> timeout) {
  auto const pretrip =
      std::holds_alternative<interval<unixtime_t>>(q.start_time_);
  auto const pong_applicable =
      pretrip &&
      ((SearchDir == direction::kBackward) != q.extend_interval_later_);
  if (pong_applicable) {
    try {
      return pong_search(tt, rtt, s_state, r_state, q, SearchDir, timeout);
    } catch (std::exception const&) {
      // fall through to rRAPTOR (same contract as the motis endpoint)
    }
  }
  return raptor_search(tt, rtt, s_state, r_state, q, SearchDir, timeout);
}

// PHASE 2: backward pruning search.
//
// For every anchor journey the paper runs one reverse search from the
// target, started at the slack-relaxed time and capped at the slack-relaxed
// trip budget. Three adaptations for the range setting:
//
//  * The anchors are first reduced to the pareto frontier over
//    (relaxed time, budget): an anchor whose relaxed start is no looser and
//    whose budget is no larger than another's cannot contribute anything -
//    without this a range query would run one one-to-all search per anchor,
//    and a window easily holds a hundred of them. What remains is at most
//    one search per distinct trip count.
//
//  * The remaining searches share ONE round-times matrix (they are just the
//    start times of one rRAPTOR): the accumulated maximum over the runs is
//    exactly the union the paper takes.
//
//  * ONE bound matrix covers the whole departure window. That is the
//    deliberate range approximation: a departure's slack is measured from
//    that departure, so the union over the window is dominated by its LAST
//    departure and earlier departures are bounded more loosely than a tight
//    per-departure BM-RAPTOR would bound them (by up to the window width;
//    measured: 666 min mean window vs a 41 min mean slack allowance). The
//    tight variant - cut the range into slices, give each its own anchors,
//    bounds and destination cap, down to one slice per departure - was
//    implemented and measured on this instance: the backward pruning cost
//    grows linearly with the number of slices (492 -> 916 -> 1431 -> 2143
//    -> 2762 ms for 1/2/3/5/7 slices) while the main search barely improves
//    (825 -> 723 ms), because mcraptor's own worst_at_dest_ + dest_bag_
//    destination pruning already covers most of what the arrival slack adds.
//    It is a net loss here, so the window-wide bound is what ships; the
//    trip-budget half of the bound, which is what actually pays off, is
//    departure-independent anyway. The remaining per-departure exactness is
//    restored by the final restriction to J_R below.
// Turn a finished PruneDir search's round times into a tau_arr^->(v, i)
// matrix. Split out of compute_reach_bounds() so the ping can supply the
// round times directly - with relaxed target pruning (raptor::set_dest_relax)
// they are valid at every stop, and a separate one-to-all search is not
// needed at all.
template <direction PruneDir>
bmrap_bounds build_reach_matrix(timetable const& tt,
                                query const& q,
                                raptor_state& state,
                                std::uint8_t const budget) {
  constexpr auto const kInvalid = kInvalidDelta<PruneDir>;
  auto const is_looser = [](auto const a, auto const b) {
    return PruneDir == direction::kForward ? a < b : a > b;
  };
  // Relax by one transfer buffer per stop, baked into the matrix so no
  // consumer has to know. nigiri's round_times_ hold POST-transfer values in
  // both directions: forward that is "earliest time you can board at p"
  // (arrival + transfer), backward it is "latest time you may arrive at p"
  // (boarding - transfer). Comparing the two directly would demand
  // latest_arrival >= earliest_arrival + transfer, one buffer stricter than
  // the meet-in-the-middle condition, and would drop short journeys whose
  // total duration is on the order of a transfer time. Footpath arrivals do
  // not pay the buffer at all, so subtracting the full transfer time is the
  // conservative choice: it can only ever weaken the bound.
  auto const dir_prune = [](auto const x) {
    return PruneDir == direction::kForward ? x : -x;
  };

  auto bounds = bmrap_bounds{};
  bounds.resize(tt.n_locations(), budget, kInvalid);
  auto const round_times = state.get_round_times<kVias>();
  for (auto i = 0U; i <= budget; ++i) {
    for (auto l = 0U; l != tt.n_locations(); ++l) {
      auto const cur = round_times[i][l][kVias];
      auto const prev = (i == 0U) ? kInvalid : bounds.at(i - 1U, l);
      auto const best = is_looser(cur, prev) ? cur : prev;
      if (best == kInvalid) {
        bounds.at(i, l) = kInvalid;
        continue;
      }
      auto const tt_min = adjusted_transfer_time(
          q.transfer_time_settings_,
          tt.locations_.transfer_time_[location_idx_t{l}].count());
      bounds.at(i, l) = static_cast<delta_t>(best - dir_prune(tt_min));
    }
  }
  return bounds;
}

// Reachability bounds for the OPPOSITE search direction: tau_arr^->(v, i),
// the earliest time the main search's origin can put you at v using at most
// i trips, departing no earlier than `from`.
//
// This is the mirror of compute_bounds(): where that one prunes a forward
// search with "you must be at v by this time to still make it", this prunes
// a BACKWARD search with "you cannot possibly be at v before this time".
// Together they are the classic meet-in-the-middle prune.
//
// This variant runs its own one-to-all search, which is only necessary when
// the ping cannot supply the round times itself - i.e. when its target
// pruning has NOT been relaxed by the arrival slack. With
// raptor::set_dest_relax() the ping's own state can be handed straight to
// build_reach_matrix() instead, which is what the profile driver does.
template <direction PruneDir, bool Rt>
bmrap_bounds compute_reach_bounds(timetable const& tt,
                                  rt_timetable const* rtt,
                                  query const& q,
                                  unixtime_t const from,
                                  unixtime_t const horizon,
                                  day_idx_t const base,
                                  std::uint8_t const budget,
                                  raptor_state& state,
                                  raptor_stats& stats) {
  constexpr auto const kInvalid = kInvalidDelta<PruneDir>;
  auto const is_looser = [](auto const a, auto const b) {
    return PruneDir == direction::kForward ? a < b : a > b;
  };

  auto is_dest = bitvec{tt.n_locations()};
  auto is_via = std::array<bitvec, kMaxVias>{};
  auto dist_to_dest = std::vector<std::uint16_t>{};
  auto td_dist_to_dest = hash_map<location_idx_t, std::vector<td_offset>>{};
  auto via_stops = std::vector<via_stop>{};
  auto lb = std::vector<std::uint16_t>(tt.n_locations(), std::uint16_t{0U});

  auto r = raptor<PruneDir, Rt, kVias, search_mode::kOneToAll>{
      tt,
      rtt,
      state,
      is_dest,
      is_via,
      dist_to_dest,
      td_dist_to_dest,
      lb,
      via_stops,
      base,
      q.allowed_claszes_,
      q.require_bike_transport_,
      q.require_car_transport_,
      q.prf_idx_ == 2U,
      q.transfer_time_settings_};

  auto starts = std::vector<start>{};
  get_starts(PruneDir, tt, rtt, from, q.start_, q.td_start_, q.via_stops_,
             q.max_start_offset_, q.start_match_mode_, q.use_start_footpaths_,
             starts, false, q.prf_idx_, q.transfer_time_settings_);
  r.reset_arrivals();
  r.next_start_time();
  for (auto const& st : starts) {
    r.add_start(st.stop_, st.time_at_stop_);
  }
  auto results = pareto_set<journey>{};
  // no destination: `horizon` is the far end of what the search it prunes
  // can ever look at, and doubles as this one-to-all search's global cutoff
  r.execute(from, static_cast<std::uint8_t>(budget - 1U), horizon, q.prf_idx_,
            results);
  stats = stats + r.get_stats();

  return build_reach_matrix<PruneDir>(tt, q, state, budget);
}

template <direction SearchDir, bool Rt>
bmrap_bounds compute_bounds(timetable const& tt,
                            rt_timetable const* rtt,
                            query const& q,
                            std::vector<anchor> const& anchors,
                            day_idx_t const base,
                            unixtime_t const horizon,
                            std::uint8_t const budget,
                            raptor_state& state,
                            raptor_stats& stats,
                            bmrap_bounds const* reach = nullptr) {
  constexpr auto const kPruneDir = flip(SearchDir);
  constexpr auto const kInvalid = kInvalidDelta<kPruneDir>;
  // "better" in the PRUNING search's direction (= looser as a bound)
  auto const is_looser = [](auto const a, auto const b) {
    return kPruneDir == direction::kForward ? a < b : a > b;
  };

  // slack-relaxed start time + trip budget per anchor
  using run_t = std::pair<unixtime_t, std::uint8_t>;
  auto runs = std::vector<run_t>{};
  runs.reserve(anchors.size());
  for (auto const& a : anchors) {
    runs.emplace_back(anchor_deadline(a), trip_budget(a.trips_, budget));
  }
  // Most trips first (the paper processes anchors from most to fewest used
  // trips), ties broken by looseness, then the pareto reduction: with
  // budgets non-increasing, a run is redundant as soon as an already-kept
  // run is at least as loose. What survives is at most one search per
  // distinct trip count instead of one per anchor journey.
  std::sort(begin(runs), end(runs), [&](run_t const& a, run_t const& b) {
    return a.second != b.second ? a.second > b.second
                                : is_looser(a.first, b.first);
  });
  auto reduced = std::vector<run_t>{};
  for (auto const& x : runs) {
    if (utl::any_of(reduced, [&](run_t const& o) {
          return !is_looser(x.first, o.first);
        })) {
      continue;
    }
    reduced.push_back(x);
  }
  if (std::getenv("NIGIRI_BMRAP_NO_REDUCE") == nullptr) {
    runs = std::move(reduced);
  }

  auto is_dest = bitvec{tt.n_locations()};
  auto is_via = std::array<bitvec, kMaxVias>{};
  auto dist_to_dest = std::vector<std::uint16_t>{};
  auto td_dist_to_dest = hash_map<location_idx_t, std::vector<td_offset>>{};
  auto via_stops = std::vector<via_stop>{};
  auto lb = std::vector<std::uint16_t>(tt.n_locations(), std::uint16_t{0U});

  auto r = raptor<kPruneDir, Rt, kVias, search_mode::kOneToAll>{
      tt,
      rtt,
      state,
      is_dest,
      is_via,
      dist_to_dest,
      td_dist_to_dest,
      lb,
      via_stops,
      base,
      q.allowed_claszes_,
      q.require_bike_transport_,
      q.require_car_transport_,
      q.prf_idx_ == 2U,
      q.transfer_time_settings_};

  // stage 1 prunes stage 2 (see raptor::set_bounds)
  r.set_bounds(reach);

  // the pruning search starts where the main search ends
  auto qf = q;
  qf.flip_dir();

  // Diagnostic: run every anchor in its OWN search space and merge the
  // staggered matrices by hand. This gives up the cross-run reuse (the
  // paper's "search space is not cleared between runs") but also removes
  // nigiri's always-on local pruning from the picture - which the paper
  // explicitly forbids in stage 2 - so it isolates the shift arithmetic
  // from the sharing.
  auto const isolated = std::getenv("NIGIRI_BMRAP_STAGGER_ISOLATED") != nullptr;
  auto acc = std::vector<delta_t>{};
  if (isolated) {
    acc.assign(static_cast<std::size_t>(budget + 1U) * tt.n_locations(),
               kInvalid);
  }

  auto starts = std::vector<start>{};
  auto results = pareto_set<journey>{};
  for (auto const& [t, b] : runs) {
    starts.clear();
    get_starts(kPruneDir, tt, rtt, t, qf.start_, qf.td_start_, qf.via_stops_,
               qf.max_start_offset_, qf.start_match_mode_,
               qf.start_match_mode_ != location_match_mode::kIntermodal,
               starts, false, q.prf_idx_, q.transfer_time_settings_);
    r.next_start_time();
    // the paper's staggered alignment: this run is allowed b trips out of a
    // global budget of `budget`, so its rounds occupy slots
    // (budget - b) + 1 ... budget. Slot i then means "i trips remaining"
    // on one scale for every anchor, and the run starts from the labels the
    // previous (higher-budget) run left at slot budget - b.
    r.set_start_round(std::getenv("NIGIRI_BMRAP_NO_STAGGER") != nullptr
                          ? 0U
                          : static_cast<unsigned>(budget - b));
    for (auto const& s : starts) {
      r.add_start(s.stop_, s.time_at_stop_);
    }
    // `horizon` is the far end of the main search's window: the main search
    // never holds a label beyond it, so bounds beyond it are dead weight.
    if (isolated) {
      r.reset_arrivals();
      r.next_start_time();
      r.set_start_round(0U);
      for (auto const& st : starts) {
        r.add_start(st.stop_, st.time_at_stop_);
      }
    }
    r.execute(t, static_cast<std::uint8_t>(b - 1U), horizon, q.prf_idx_,
              results);
    if (isolated) {
      auto const shift = static_cast<unsigned>(budget - b);
      auto const rt = state.get_round_times<kVias>();
      for (auto j = 0U; j <= b; ++j) {
        auto& dst_row = acc[static_cast<std::size_t>(j + shift) *
                            tt.n_locations()];
        auto* dst = &dst_row;
        for (auto l = 0U; l != tt.n_locations(); ++l) {
          auto const cur = rt[j][l][kVias];
          if (cur != kInvalid && is_looser(cur, dst[l])) {
            dst[l] = cur;
          }
        }
      }
    }
  }
  stats = stats + r.get_stats();

  auto bounds = bmrap_bounds{};
  bounds.resize(tt.n_locations(), budget, kInvalid);
  auto const round_times = state.get_round_times<kVias>();
  for (auto i = 0U; i <= budget; ++i) {
    for (auto l = 0U; l != tt.n_locations(); ++l) {
      auto const cur =
          isolated ? acc[static_cast<std::size_t>(i) * tt.n_locations() + l]
                   : round_times[i][l][kVias];
      auto const prev = (i == 0U) ? kInvalid : bounds.at(i - 1U, l);
      bounds.at(i, l) = is_looser(cur, prev) ? cur : prev;
    }
  }
  return bounds;
}

}  // namespace nigiri::routing::bmrap_detail
