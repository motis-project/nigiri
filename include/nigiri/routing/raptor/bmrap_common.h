#pragma once

// Slack configuration, anchor definition and pruning searches shared by the
// BM-RAPTOR driver in bmrap_profile.cc.

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
#include "nigiri/routing/raptor/mcraptor.h"
#include "nigiri/routing/raptor/raptor.h"
#if defined(NIGIRI_CUDA)
#include "nigiri/routing/gpu/mcraptor.h"
#include "nigiri/routing/gpu/raptor.h"
#endif
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

// The scalar (two-criteria) engine for the ping, the pong and the backward
// pruning searches, picked by state type the way pong_algo_for does in
// pong.cc. The multicriteria phases always stay on the CPU, so every
// criteria configuration works regardless of the scalar engine.
template <direction SearchDir, bool Rt, typename AlgoState>
struct bmrap_algo_for {
  using type = raptor<SearchDir, Rt, kVias, search_mode::kOneToOne>;
};

// kOneToAll differs from kOneToOne only in skipping result collection and
// reconstruct and in disabling target pruning; compute_bounds() runs with an
// empty destination set, where all three are no-ops (verified: identical
// journeys on 200/200 Berlin queries). So an engine lacking the mode - the
// GPU raptor - serves here unchanged.
template <direction SearchDir, bool Rt, typename AlgoState>
struct bmrap_prune_algo_for {
  using type = raptor<SearchDir, Rt, kVias, search_mode::kOneToAll>;
};

#if defined(NIGIRI_CUDA)
// BM-RAPTOR prunes via its own bm_bounds_ (set_bounds(bmrap_bounds
// const*), unconstrained by WithBounds) - never master's round/via bound
// matrix, so WithBounds=false here.
template <direction SearchDir, bool Rt>
struct bmrap_algo_for<SearchDir, Rt, gpu::gpu_raptor_state> {
  using type = gpu::gpu_raptor<SearchDir, false>;
};

template <direction SearchDir, bool Rt>
struct bmrap_prune_algo_for<SearchDir, Rt, gpu::gpu_raptor_state> {
  using type = gpu::gpu_raptor<SearchDir, false>;
};
#endif

// Engine for the MULTICRITERIA phases (4, 5 and 5b). The CPU mcraptor takes
// any composed criteria; the device one implements exactly two label shapes -
// arrival alone, and arrival plus the generalized-cost extras - so it can
// only stand in for those. GpuMc is resolved by the caller, which is what
// keeps the two bodies from both being instantiated for the CPU engines.
template <direction SearchDir, typename Criteria, bool GpuMc>
struct bmrap_mc_algo_for {
  using type = basic_mcraptor<SearchDir, Criteria, /*RangeReuse=*/false>;
  using state = basic_mcraptor_state<Criteria>;
};

#if defined(NIGIRI_CUDA)
// Which criteria have a device equivalent, and whether the scalar engine is
// on the device at all - running phases 4/5 on the GPU while the ping and
// pong stay on the CPU would only add transfers.
template <typename Criteria, typename AlgoState>
inline constexpr bool kGpuMcSupported =
    std::is_same_v<AlgoState, gpu::gpu_raptor_state> &&
    (std::is_same_v<Criteria, arr_criteria> ||
     std::is_same_v<Criteria, arr_cost_criteria>);

template <direction SearchDir, typename Criteria>
struct bmrap_mc_algo_for<SearchDir, Criteria, true> {
  using type =
      gpu::gpu_mcraptor<SearchDir,
                        std::is_same_v<Criteria, arr_cost_criteria>>;
  using state = gpu::gpu_mcraptor_state;
};
#else
template <typename Criteria, typename AlgoState>
inline constexpr bool kGpuMcSupported = false;
#endif

// How much of the multicriteria work runs on the device, for the two label
// shapes the device mcraptor implements (arr, arr+cost):
//
//   0  everything on the CPU
//   1  the mc PING only (phases 4 and 5b) - phase 4 is one big search per step
//      and gains ~2.5x on the device
//   2  the mc PONG (phase 5) as well - a sequence of tiny per-journey searches
//      the device loses on (0.48 -> 1.94 ms/query), kept only to exercise it
constexpr int kBmrapGpuMcMode = 1;

// Host view of an engine's round times, in raptor_state's layout either way.
// The GPU engine copies them back into `buf`, which is why the caller owns
// the buffer.
template <typename AlgoState, typename Algo>
flat_matrix_view<std::array<delta_t, kVias + 1> const> host_round_times(
    AlgoState& state,
    [[maybe_unused]] Algo& algo,
    [[maybe_unused]] std::vector<std::array<delta_t, kVias + 1>>& buf,
    [[maybe_unused]] unsigned const n_locations) {
#if defined(NIGIRI_CUDA)
  if constexpr (std::is_same_v<AlgoState, gpu::gpu_raptor_state>) {
    algo.copy_round_times(buf);
    return {{buf.data(), buf.size()}, kMaxTransfers + 2U, n_locations};
  } else
#endif
  {
    return static_cast<AlgoState const&>(state)
        .template get_round_times<kVias>();
  }
}


// sigma_arr / sigma_tr of the paper (Sauer'24 evaluates 1.25; 1.5 widens the
// restricted set, and the caps/floors keep that in hand at the extremes).
struct slack_cfg {
  double arr_{1.5};
  double trip_{1.5};
  // >= 0 switches that dimension from the multiplicative sigma to the
  // paper's ADDITIVE form (Eq. 3.1): "at most N minutes / N trips worse than
  // the anchor". Both forms relax the same reference quantity, see
  // relax_arr().
  double arr_fixed_min_{-1.0};
  double trip_fixed_{-1.0};
  // sigma is a RATIO, so on a 20 h journey sigma_arr = 1.5 admits arrivals
  // 10 h late - a restricted set wide enough to stop restricting, and the
  // dominant cost on long-haul queries. At sigma = 1.5 the arrival cap binds
  // above 6 h of travel, the trip cap from 4 trips.
  double arr_cap_min_{180.0};
  double trip_cap_{2.0};
  // Floors, for the opposite reason: a 50% ratio on a 30 min trip grants
  // 15 min - about one missed connection - and the restricted set collapses
  // onto the anchors themselves.
  double arr_min_min_{20.0};
  double trip_min_{1.0};
};

inline slack_cfg const& get_slack() {
  static constexpr auto const cfg = slack_cfg{};
  return cfg;
}

// Applies the arrival slack to the paper's tau_arr(A(J)) - tau_dep, in
// minutes. Both slack modes relax that same reference, so they stay directly
// comparable: sigma_arr scales it, the fixed variant pads it by a constant.
inline double relax_arr(double const reference_minutes) {
  auto const& cfg = get_slack();
  auto const extra = cfg.arr_fixed_min_ >= 0.0
                         ? cfg.arr_fixed_min_
                         : reference_minutes * (cfg.arr_ - 1.0);
  return reference_minutes +
         std::clamp(extra, std::min(cfg.arr_min_min_, cfg.arr_cap_min_),
                    cfg.arr_cap_min_);
}

// One journey of the anchor pareto set J_A. Same convention as
// journey::start_time_ / dest_time_: `anchored_` is the query-side time (the
// departure forward, the arrival backward), `found_` the other end.
struct anchor {
  unixtime_t anchored_, found_;
  std::uint8_t trips_;
};

// Latest time a journey anchored to `a` may arrive. Both sides of the
// restriction go through here - the pruning searches start from it, the
// final filter tests against it - and that is the point: a multiplicative
// sigma_arr has no canonical origin, so bounds relaxed from the anchor's
// departure and a filter asking travel(J) <= sigma * (arr(A) - dep(J)) would
// disagree by (sigma - 1) * (dep(A) - dep(J)), and the bounds would prune
// journeys the restriction keeps.
inline unixtime_t anchor_deadline(anchor const& a) {
  // found_ - anchored_ is negative for a backward query (departure -
  // arrival), and relax_arr()'s floor/cap clamp is only meaningful on a
  // magnitude: fed a negative reference it clamps the slack up to the +20 min
  // floor, moving the deadline INSIDE the anchor's own travel time so the
  // anchor prunes itself. Relax the magnitude, put the sign back - the way
  // raptor::update_time_at_dest() does with dir().
  auto const signed_travel = (a.found_ - a.anchored_).count();
  auto const sign = signed_travel < 0 ? -1 : 1;
  auto const relaxed = std::llround(
      relax_arr(static_cast<double>(std::abs(signed_travel))));
  return a.anchored_ +
         i32_minutes{static_cast<std::int32_t>(sign * relaxed)};
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
  auto const extra = cfg.trip_fixed_ >= 0.0
                         ? cfg.trip_fixed_
                         : static_cast<double>(trips) * (cfg.trip_ - 1.0);
  auto const b =
      static_cast<double>(trips) +
      std::clamp(extra, std::min(cfg.trip_min_, cfg.trip_cap_), cfg.trip_cap_);
  return static_cast<std::uint8_t>(
      std::clamp<double>(std::floor(b), trips, max));
}

// A(J) for a journey with `trips` trips anchored at `anchored`: among the
// anchors AVAILABLE there (anchored no earlier than J - a traveller can
// always take a later-departing anchor) and needing at most as many trips,
// the one with the EARLIEST arrival.
//
// The paper says "highest trip count <= |J|", which agrees with "earliest
// arrival" on the raw (arrival, trips) Pareto set - a higher-trip point only
// survives by arriving strictly earlier. They stop agreeing on the "available
// at d" slice of the full (dep, arr, trips) set, where a later-departing,
// higher-trip anchor can have a WORSE arrival than an earlier-departing one.
// "Most trips" then picks the worse reference. "Earliest arrival" always
// compares J against the best alternative it could have had for free at its
// own anchor time, which is the comparison the restriction wants, and it has
// a natural completion bound - see close_anchor_profile().
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

// PHASE 1: the two-criteria range search whose result is the anchor set J_A.
// PONG is fastest but only covers the two interval-extension configurations
// that run with the search direction; rRAPTOR answers the rest. Same set
// either way.
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

// PHASE 1b: close the anchor profile past the window.
//
// A(J) is looked up over the anchors available at J, so a set collected over
// the query window alone is TRUNCATED near its far end and the same departure
// answers differently under a 1 h and an 8 h searchWindow (measured: 16 of 46
// queries, always in the trips dimension, since the budget follows the anchor
// with the most trips - exactly the one a longer window adds).
//
// The truncation is bounded: nothing departing after A(d) can arrive before
// it, so d's profile only depends on departures in [d, A(d)]. One plain
// ontrip earliest-arrival probe from the window boundary gives that bound
// exactly - whatever arrival it finds is by definition the earliest reachable
// from any departure at or beyond the boundary - and re-running the anchor
// search over that margin closes every in-window departure's profile. The
// margin only feeds the restriction; it is not part of the reported window,
// and it comes out <= 0 (probe skipped) when phase 1 already extended past it.
// Returns the margin, for the stats.
template <direction SearchDir>
duration_t close_anchor_profile(
    timetable const& tt,
    rt_timetable const* rtt,
    query const& q,
    interval<unixtime_t> const anchor_interval,
    raptor_state& prune_state,
    std::optional<std::chrono::seconds> const timeout,
    std::vector<anchor>& anchors) {
  constexpr auto const kFwd = (SearchDir == direction::kForward);

  auto a_ceiling = std::optional<unixtime_t>{};
  {
    auto q_ceiling = q;
    q_ceiling.start_time_ = kFwd ? anchor_interval.to_ : anchor_interval.from_;
    q_ceiling.min_connection_count_ = 0U;
    q_ceiling.extend_interval_earlier_ = false;
    q_ceiling.extend_interval_later_ = false;
    auto ceiling_state = search_state{};
    auto ceiling_algo = raptor_state{};
    try {
      auto const r = raptor_search(tt, rtt, ceiling_state, ceiling_algo,
                                   q_ceiling, SearchDir, timeout);
      for (auto const& j : *r.journeys_) {
        if (!a_ceiling.has_value() ||
            (kFwd ? j.dest_time_ < *a_ceiling : j.dest_time_ > *a_ceiling)) {
          a_ceiling = j.dest_time_;
        }
      }
    } catch (std::exception const&) {
      // nothing reachable beyond the window, so nothing to close
    }
  }
  auto const margin =
      a_ceiling.has_value()
          ? std::min(
                duration_t{static_cast<duration_t::rep>(std::abs(
                    (kFwd ? *a_ceiling - anchor_interval.to_
                          : anchor_interval.from_ - *a_ceiling)
                        .count()))},
                q.max_travel_time_)
          : duration_t{0};
  if (margin > duration_t{0}) {
    auto q_ext = q;
    q_ext.start_time_ =
        kFwd ? interval<unixtime_t>{anchor_interval.to_,
                                    anchor_interval.to_ + margin}
             : interval<unixtime_t>{anchor_interval.from_ - margin,
                                    anchor_interval.from_};
    q_ext.min_connection_count_ = 0U;
    q_ext.extend_interval_earlier_ = false;
    q_ext.extend_interval_later_ = false;
    auto ext_state = search_state{};
    try {
      auto const ext = run_anchor_search<SearchDir>(tt, rtt, ext_state,
                                                    prune_state, q_ext, timeout);
      for (auto const& j : *ext.journeys_) {
        anchors.push_back({j.start_time_, j.dest_time_,
                           static_cast<std::uint8_t>(j.transfers_ + 1U)});
      }
    } catch (std::exception const&) {
      // extension is an accuracy refinement, never a hard requirement
    }
  }
  // The union of two windows is not a pareto set, and a dominated anchor must
  // never become somebody's A(J) - which of the two was found would then
  // change the restriction.
  {
    auto keep = std::vector<anchor>{};
    auto const is_better = [](unixtime_t const a, unixtime_t const b) {
      return kFwd ? a < b : a > b;
    };
    for (auto const& a : anchors) {
      auto const dominated = utl::any_of(anchors, [&](anchor const& b) {
        return (&b != &a) && b.trips_ <= a.trips_ &&
               !is_better(b.anchored_, a.anchored_) &&
               !is_better(a.found_, b.found_) &&
               (b.trips_ < a.trips_ || b.anchored_ != a.anchored_ ||
                b.found_ != a.found_ || &b < &a);
      });
      if (!dominated) {
        keep.push_back(a);
      }
    }
    anchors = std::move(keep);
  }
  return margin;
}

// A finished PruneDir search's round times as a tau_arr^->(v, i) matrix.
// Separate from compute_bounds() so the PING can supply them directly: with
// relaxed target pruning (raptor::set_dest_relax) they are valid at every
// stop, so no dedicated one-to-all search is needed.
template <direction PruneDir>
bmrap_bounds build_reach_matrix(
    timetable const& tt,
    query const& q,
    flat_matrix_view<std::array<delta_t, kVias + 1> const> const round_times,
    std::uint8_t const budget) {
  constexpr auto const kInvalid = kInvalidDelta<PruneDir>;
  auto const is_looser = [](auto const a, auto const b) {
    return PruneDir == direction::kForward ? a < b : a > b;
  };
  // round_times_ hold POST-transfer values in both directions, so comparing
  // a forward and a backward matrix directly would demand one transfer buffer
  // more than meet-in-the-middle needs and would drop journeys whose total
  // duration is on the order of a transfer time. Subtract the buffer here,
  // once, so no consumer has to know; footpath arrivals never paid it, which
  // makes this the conservative direction - it can only weaken the bound.
  auto const dir_prune = [](auto const x) {
    return PruneDir == direction::kForward ? x : -x;
  };

  auto bounds = bmrap_bounds{};
  bounds.resize(tt.n_locations(), budget, kInvalid);
  for (auto i = 0U; i <= budget; ++i) {
    for (auto l = 0U; l != tt.n_locations(); ++l) {
      auto const cur = round_times[i][l][kVias];
      auto const prev_out = (i == 0U) ? kInvalid : bounds.at(i - 1U, l);
      // Most cells are unreachable on a large timetable, so bail out before
      // the transfer-time lookup: best can only be valid if one of these is.
      if (cur == kInvalid && prev_out == kInvalid) {
        bounds.at(i, l) = kInvalid;
        continue;
      }
      auto const tt_min = adjusted_transfer_time(
          q.transfer_time_settings_,
          tt.locations_.transfer_time_[location_idx_t{l}].count());
      // The prefix has to run over the RAW round times for the buffer to come
      // off once. Row i - 1 already has it taken off, so add it back rather
      // than keep a second matrix around: carrying the prefix in the output
      // row - the obvious way to write a running prefix in place - subtracts
      // the buffer again at every round the prefix survives, which leaves row
      // i up to i buffers weaker than this comment block promises.
      auto const prev =
          prev_out == kInvalid
              ? kInvalid
              : static_cast<delta_t>(prev_out + dir_prune(tt_min));
      auto const best = is_looser(cur, prev) ? cur : prev;
      bounds.at(i, l) = static_cast<delta_t>(best - dir_prune(tt_min));
    }
  }
  return bounds;
}

// Same for a CPU raptor_state. The view-taking overload above is what lets a
// GPU engine feed in host-copied round times without this header knowing
// about device buffers.
template <direction PruneDir>
bmrap_bounds build_reach_matrix(timetable const& tt,
                                query const& q,
                                raptor_state& state,
                                std::uint8_t const budget) {
  return build_reach_matrix<PruneDir>(
      tt, q,
      static_cast<raptor_state const&>(state).get_round_times<kVias>(), budget);
}

// Build the matrix from `algo`'s finished round times, wherever they live.
// The GPU engine reduces them in place and returns only the rows the caller
// keeps; the host engines transform their own state. PruneDir is always the
// engine's own direction, so the caller never has to reconcile the two.
template <direction PruneDir, typename AlgoState, typename Algo>
bmrap_bounds reach_matrix(
    timetable const& tt,
    query const& q,
    AlgoState& state,
    [[maybe_unused]] Algo& algo,
    [[maybe_unused]] std::vector<std::array<delta_t, kVias + 1>>& buf,
    std::uint8_t const budget,
    bool const sub_transfer) {
#if defined(NIGIRI_CUDA)
  if constexpr (std::is_same_v<AlgoState, gpu::gpu_raptor_state>) {
    auto out = bmrap_bounds{};
    algo.build_reach_bounds(out, budget, sub_transfer);
    return out;
  } else
#endif
  {
    if (sub_transfer) {
      return build_reach_matrix<PruneDir>(tt, q, state, budget);
    }
    // PHASE 2b: the prefix without the transfer-buffer subtraction.
    constexpr auto const kInvalid = kInvalidDelta<PruneDir>;
    auto const is_looser = [](auto const a, auto const b) {
      return PruneDir == direction::kForward ? a < b : a > b;
    };
    auto const round_times =
        host_round_times(state, algo, buf, tt.n_locations());
    auto bounds = bmrap_bounds{};
    bounds.resize(tt.n_locations(), budget, kInvalid);
    for (auto i = 0U; i <= budget; ++i) {
      for (auto l = 0U; l != tt.n_locations(); ++l) {
        auto const cur = round_times[i][l][kVias];
        auto const prev = (i == 0U) ? kInvalid : bounds.at(i - 1U, l);
        bounds.at(i, l) = is_looser(cur, prev) ? cur : prev;
      }
    }
    return bounds;
  }
}

// PHASE 2: backward pruning search -> tau_dep^<-(v, i).
//
// The paper runs one reverse search per anchor from the target, started at
// the anchor's slack-relaxed time and capped at its slack-relaxed trip
// budget. Two adaptations: the anchors are first reduced to the pareto
// frontier over (relaxed time, budget), leaving at most one search per
// distinct trip count; and the survivors share one round-times matrix as the
// start times of a single rRAPTOR, whose accumulated maximum is the union the
// paper takes.
//
// The matrix is built PER STEP, anchored at that step's own departure - a
// window-wide variant bounds early departures loosely by up to the window
// width (666 min mean window vs 41 min mean slack), and slicing finer just
// grows backward pruning linearly without helping the main search. The
// per-step bound is affordable only because of the driver's structure: one
// single-departure BM-RAPTOR per step, anchor set cached across steps that
// cannot change it.
template <direction SearchDir, bool Rt, typename AlgoState>
bmrap_bounds compute_bounds(timetable const& tt,
                            rt_timetable const* rtt,
                            query const& q,
                            std::vector<anchor> const& anchors,
                            day_idx_t const base,
                            unixtime_t const horizon,
                            std::uint8_t const budget,
                            AlgoState& state,
                            raptor_stats& stats,
                            bmrap_bounds const* reach = nullptr) {
  // `horizon` is the far end of the main search's window: it never holds a
  // label beyond that, so bounds beyond it are dead weight.
  constexpr auto const kPruneDir = flip(SearchDir);
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
  // Most trips first (as the paper processes them), ties by looseness. With
  // budgets non-increasing, a run is redundant as soon as an already-kept one
  // is at least as loose.
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
  runs = std::move(reduced);

  auto is_dest = bitvec{tt.n_locations()};
  auto is_via = std::array<bitvec, kMaxVias>{};
  auto dist_to_dest = std::vector<std::uint16_t>{};
  auto td_dist_to_dest = hash_map<location_idx_t, std::vector<td_offset>>{};
  auto via_stops = std::vector<via_stop>{};
  auto lb = std::vector<std::uint16_t>(tt.n_locations(), std::uint16_t{0U});

  auto r = typename bmrap_prune_algo_for<kPruneDir, Rt, AlgoState>::type{
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
      q.no_compulsory_reservation_,
      q.transfer_time_settings_,
      q.prf_idx_};

  // stage 1 prunes stage 2 (see raptor::set_bounds)
  r.set_bounds(reach);

  // staging for host_round_times() - unused by the CPU engines
  auto rt_buf = std::vector<std::array<delta_t, kVias + 1>>{};

  // the pruning search starts where the main search ends
  auto qf = q;
  qf.flip_dir();

  auto starts = std::vector<start>{};
  auto results = pareto_set<journey>{};
  for (auto const& [t, b] : runs) {
    starts.clear();
    get_starts(kPruneDir, tt, rtt, t, qf.start_, qf.td_start_, qf.via_stops_,
               qf.max_start_offset_, qf.start_match_mode_,
               qf.start_match_mode_ != location_match_mode::kIntermodal,
               starts, false, q.prf_idx_, q.transfer_time_settings_);
    r.next_start_time();
    // The paper's staggered alignment: a run allowed b of `budget` trips
    // occupies slots (budget - b) + 1 ... budget, so slot i means "i trips
    // remaining" on one scale for every anchor and the run starts from the
    // labels the previous, higher-budget run left at slot budget - b.
    r.set_start_round(static_cast<unsigned>(budget - b));
    for (auto const& s : starts) {
      r.add_start(s.stop_, s.time_at_stop_);
    }
    r.execute(t, static_cast<std::uint8_t>(b - 1U), horizon, results);
  }
  stats = stats + r.get_stats();

  return reach_matrix<kPruneDir>(tt, q, state, r, rt_buf, budget,
                                 /*sub_transfer=*/false);
}

}  // namespace nigiri::routing::bmrap_detail
