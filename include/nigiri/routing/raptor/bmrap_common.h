#pragma once

// Slack configuration, anchors and pruning searches of the BM-RAPTOR driver
// (bmrap_profile.cc).

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <algorithm>
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
#include "nigiri/routing/raptor/pong.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/routing/raptor/raptor_stats.h"
#include "nigiri/routing/raptor_search.h"
#include "nigiri/routing/search.h"
#include "nigiri/routing/start_times.h"
#include "nigiri/routing/transfer_time_settings.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

namespace nigiri::routing::bmrap_detail {

constexpr auto const kVias = via_offset_t{0U};

// "better" in the search direction: earlier forward, later backward
template <direction Dir>
constexpr bool is_better(auto const a, auto const b) {
  return Dir == direction::kForward ? a < b : a > b;
}

// Scalar (two-criteria) engine for ping, pong and the backward pruning
// searches, picked by state type like pong_algo_for in pong.cc.
template <direction SearchDir, bool Rt, typename AlgoState>
struct bmrap_algo_for {
  using type = raptor<SearchDir, Rt, kVias, search_mode::kOneToOne>;
};

// kOneToAll only skips result collection, reconstruct and target pruning, all
// no-ops for the empty destination set of compute_bounds(), so the GPU raptor
// (which lacks the mode) serves here unchanged.
template <direction SearchDir, bool Rt, typename AlgoState>
struct bmrap_prune_algo_for {
  using type = raptor<SearchDir, Rt, kVias, search_mode::kOneToAll>;
};

#if defined(NIGIRI_CUDA)
// BM-RAPTOR prunes via its own bounds, not the round/via bound matrix.
template <direction SearchDir, bool Rt>
struct bmrap_algo_for<SearchDir, Rt, gpu::gpu_raptor_state> {
  using type = gpu::gpu_raptor<SearchDir, false>;
};

template <direction SearchDir, bool Rt>
struct bmrap_prune_algo_for<SearchDir, Rt, gpu::gpu_raptor_state> {
  using type = gpu::gpu_raptor<SearchDir, false>;
};
#endif

// Engine for the multicriteria phases (4, 5, 5b): the CPU mcraptor takes any
// criteria, the device one only the shapes that fit its packed label (see
// mc_crit). The caller resolves GpuMc.
template <direction SearchDir, typename Criteria, bool GpuMc>
struct bmrap_mc_algo_for {
  using type = basic_mcraptor<SearchDir, Criteria, /*RangeReuse=*/false>;
  using state = basic_mcraptor_state<Criteria>;
};

#if defined(NIGIRI_CUDA)
// CPU criteria -> device label configuration (see kGpuMcSupported)
template <typename Criteria>
constexpr gpu::mc_crit mc_crit_of() {
  if constexpr (std::is_same_v<Criteria, arr_criteria>) {
    return gpu::mc_crit::arr;
  } else if constexpr (std::is_same_v<Criteria, arr_cost_criteria>) {
    return gpu::mc_crit::cost;
  } else if constexpr (std::is_same_v<Criteria, arr_non_transit_criteria>) {
    return gpu::mc_crit::non_transit;
  } else if constexpr (std::is_same_v<Criteria, arr_mode_filter_criteria>) {
    return gpu::mc_crit::mode_filter;
  } else {
    static_assert(
        std::is_same_v<Criteria, arr_non_transit_mode_filter_criteria>,
        "criteria has no device mcraptor equivalent");
    return gpu::mc_crit::non_transit_mode_filter;
  }
}

// non_transit + mode_filter (two live fields in one label slot) is not wired up
// yet and stays on the CPU.
template <typename Criteria>
inline constexpr bool kMcCritSupported =
    std::is_same_v<Criteria, arr_criteria> ||
    std::is_same_v<Criteria, arr_cost_criteria> ||
    std::is_same_v<Criteria, arr_non_transit_criteria> ||
    std::is_same_v<Criteria, arr_mode_filter_criteria>;

// Which (criteria, state) combinations run the mc phases on the device; the
// scalar engine must be there too, else it only adds transfers.
template <typename Criteria, typename AlgoState>
inline constexpr bool kGpuMcSupported =
    std::is_same_v<AlgoState, gpu::gpu_raptor_state> &&
    kMcCritSupported<Criteria>;

template <direction SearchDir, typename Criteria>
struct bmrap_mc_algo_for<SearchDir, Criteria, true> {
  using type = gpu::gpu_mcraptor<SearchDir, mc_crit_of<Criteria>()>;
  using state = gpu::gpu_mcraptor_state;
};
#else
template <typename Criteria, typename AlgoState>
inline constexpr bool kGpuMcSupported = false;
#endif

// sigma_arr / sigma_tr of the paper (Sauer'24 evaluates 1.25).
struct slack_cfg {
  double arr_{1.5};
  double trip_{1.5};
  // >= 0 switches a dimension to the paper's additive form (Eq. 3.1): at most
  // N minutes / N trips worse than the anchor.
  double arr_fixed_min_{-1.0};
  double trip_fixed_{-1.0};
  // Caps: sigma is a ratio, so on a 20 h journey 1.5 would admit arrivals 10 h
  // late. The arrival cap binds above 6 h of travel, the trip cap from 4 trips.
  double arr_cap_min_{180.0};
  double trip_cap_{2.0};
  // Floors: 50% of a 30 min trip is about one missed connection, which would
  // collapse the restricted set onto the anchors.
  double arr_min_min_{20.0};
  double trip_min_{1.0};
};

inline slack_cfg const& get_slack() {
  static constexpr auto const cfg = slack_cfg{};
  return cfg;
}

// Applies the arrival slack to the paper's tau_arr(A(J)) - tau_dep in minutes:
// sigma_arr scales it, the fixed variant pads it.
inline double relax_arr(double const reference_minutes) {
  auto const& cfg = get_slack();
  auto const extra = cfg.arr_fixed_min_ >= 0.0
                         ? cfg.arr_fixed_min_
                         : reference_minutes * (cfg.arr_ - 1.0);
  return reference_minutes +
         std::clamp(extra, std::min(cfg.arr_min_min_, cfg.arr_cap_min_),
                    cfg.arr_cap_min_);
}

// One journey of the anchor pareto set J_A: `anchored_` is the query-side time
// (departure forward, arrival backward), `found_` the other end.
struct anchor {
  unixtime_t anchored_, found_;
  std::uint8_t trips_;
};

// Latest time a journey anchored to `a` may arrive. The pruning searches and
// the final filter both use it: a multiplicative sigma_arr has no canonical
// origin, so two formulations would disagree and the bounds would prune
// journeys the restriction keeps.
inline unixtime_t anchor_deadline(anchor const& a) {
  // found_ - anchored_ is negative backward, and relax_arr()'s clamp only makes
  // sense on a magnitude (a negative one would put the deadline inside the
  // anchor's own travel time): relax the magnitude, restore the sign.
  auto const signed_travel = (a.found_ - a.anchored_).count();
  auto const sign = signed_travel < 0 ? -1 : 1;
  auto const relaxed =
      std::llround(relax_arr(static_cast<double>(std::abs(signed_travel))));
  return a.anchored_ + i32_minutes{static_cast<std::int32_t>(sign * relaxed)};
}

template <direction SearchDir>
bool misses_deadline(anchor const& a, unixtime_t const arrival) {
  return is_better<SearchDir>(anchor_deadline(a), arrival);
}

// floor(sigma_tr * |J|), clamped to what the query and the round times allow
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
// anchors available there (anchored no earlier - a traveller can take a
// later-departing anchor) with at most as many trips, the one arriving
// EARLIEST.
//
// The paper says "highest trip count <= |J|". That agrees on the raw
// (arrival, trips) Pareto set but not on the "available at d" slice of the
// (dep, arr, trips) set, where a later-departing, higher-trip anchor can arrive
// later and would become a worse reference. Earliest arrival compares J with
// the best alternative it had for free, and has a natural completion bound
// (see close_anchor_profile()).
template <direction SearchDir>
anchor const* anchor_of(std::vector<anchor> const& anchors,
                        unixtime_t const anchored,
                        unsigned const trips) {
  anchor const* best = nullptr;
  for (auto const& a : anchors) {
    if (a.trips_ > trips || is_better<SearchDir>(a.anchored_, anchored)) {
      continue;
    }
    if (best == nullptr || is_better<SearchDir>(a.found_, best->found_)) {
      best = &a;
    }
  }
  return best;
}

// Is a journey with `trips` trips, anchored at `anchored` and found at `found`,
// outside the restricted set? An anchor is its own A(J); anything else must
// stay within A(J)'s trip budget and deadline - the same deadline the bounds
// were built from, so pruning and filter cannot disagree.
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
         misses_deadline<SearchDir>(*a, found);
}

// PHASE 1: the two-criteria range search yielding the anchor set J_A. PONG is
// fastest but only covers the interval extensions running with the search
// direction; rRAPTOR answers the rest with the same set.
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
      // fall through to rRAPTOR
    }
  }
  return raptor_search(tt, rtt, s_state, r_state, q, SearchDir, timeout);
}

// PHASE 1b: close the anchor profile past the window.
//
// A(J) is looked up over the anchors available at J, so a set collected over
// the window alone is truncated near its far end, and a departure answers
// differently under a 1 h and an 8 h searchWindow (16 of 46 queries measured,
// always in the trips dimension).
//
// The truncation is bounded: nothing departing after A(d) arrives before it, so
// d's profile only depends on departures in [d, A(d)]. One ontrip
// earliest-arrival probe from the window boundary gives that bound, and
// re-running the anchor search over the margin closes every in-window
// departure's profile. The margin only feeds the restriction, not the reported
// window. Returns it, for the stats.
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
            is_better<SearchDir>(j.dest_time_, *a_ceiling)) {
          a_ceiling = j.dest_time_;
        }
      }
    } catch (std::exception const&) {
      // nothing reachable beyond the window
    }
  }
  auto const margin =
      a_ceiling.has_value()
          ? std::min(duration_t{static_cast<duration_t::rep>(
                         std::abs((kFwd ? *a_ceiling - anchor_interval.to_
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
      auto const ext = run_anchor_search<SearchDir>(
          tt, rtt, ext_state, prune_state, q_ext, timeout);
      for (auto const& j : *ext.journeys_) {
        anchors.push_back({j.start_time_, j.dest_time_,
                           static_cast<std::uint8_t>(j.transfers_ + 1U)});
      }
    } catch (std::exception const&) {
      // the extension only refines
    }
  }
  // The union of two windows is not a pareto set, and a dominated anchor must
  // never become somebody's A(J).
  {
    auto keep = std::vector<anchor>{};
    for (auto const& a : anchors) {
      auto const dominated = utl::any_of(anchors, [&](anchor const& b) {
        return (&b != &a) && b.trips_ <= a.trips_ &&
               !is_better<SearchDir>(b.anchored_, a.anchored_) &&
               !is_better<SearchDir>(a.found_, b.found_) &&
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

// tau_arr^->(v, i) from a finished search's round times: row i is the monotone
// prefix over rounds <= i, since round_times_ hold "exactly k trips" but the
// bound needs "at most i".
//
// round_times_ are post-transfer values in both directions. With `sub_transfer`
// the transfer buffer comes off once, so a forward and a backward matrix
// compare without demanding one buffer more than meet-in-the-middle needs;
// footpath arrivals never paid it, so this can only weaken the bound. The
// prefix runs over the raw times (row i - 1 has the buffer taken off, so it is
// added back) to avoid subtracting it again in every round the prefix survives.
template <direction PruneDir, typename AlgoState, typename Algo>
bmrap_bounds reach_matrix(timetable const& tt,
                          query const& q,
                          AlgoState& state,
                          [[maybe_unused]] Algo& algo,
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
    constexpr auto const kInvalid = kInvalidDelta<PruneDir>;
    auto const round_times =
        static_cast<AlgoState const&>(state).template get_round_times<kVias>();
    auto bounds = bmrap_bounds{};
    bounds.resize(tt.n_locations(), budget, kInvalid);
    for (auto i = 0U; i <= budget; ++i) {
      for (auto l = 0U; l != tt.n_locations(); ++l) {
        auto const cur = round_times[i][l][kVias];
        auto const prev_out = (i == 0U) ? kInvalid : bounds.at(i - 1U, l);
        // most cells of a large timetable are unreachable
        if (cur == kInvalid && prev_out == kInvalid) {
          bounds.at(i, l) = kInvalid;
          continue;
        }
        auto const buffer =
            sub_transfer
                ? (PruneDir == direction::kForward ? 1 : -1) *
                      adjusted_transfer_time(
                          q.transfer_time_settings_,
                          tt.locations_.transfer_time_[location_idx_t{l}]
                              .count())
                : 0;
        auto const prev = prev_out == kInvalid
                              ? kInvalid
                              : static_cast<delta_t>(prev_out + buffer);
        auto const best = is_better<PruneDir>(cur, prev) ? cur : prev;
        bounds.at(i, l) = static_cast<delta_t>(best - buffer);
      }
    }
    return bounds;
  }
}

// PHASE 2: backward pruning search -> tau_dep^<-(v, i).
//
// The paper runs one reverse search per anchor from the target, started at the
// anchor's slack-relaxed time and capped at its trip budget. Here the anchors
// are first reduced to the pareto frontier over (relaxed time, budget), leaving
// one search per distinct trip count, and the survivors are the start times of
// a single rRAPTOR, whose accumulated maximum is the paper's union.
//
// The matrix is built per step, anchored at that step's own departure: a
// window-wide one would bound early departures loosely by up to the window
// width (666 min mean vs 41 min mean slack). That is affordable because the
// anchor set is cached across steps that cannot change it.
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
  // `horizon` is the far end of the main search's window; bounds beyond it are
  // dead weight.
  constexpr auto const kPruneDir = flip(SearchDir);
  // better in the pruning direction = looser as a bound
  auto const is_looser = [](auto const a, auto const b) {
    return is_better<kPruneDir>(a, b);
  };

  // slack-relaxed start time + trip budget per anchor
  using run_t = std::pair<unixtime_t, std::uint8_t>;
  auto runs = std::vector<run_t>{};
  runs.reserve(anchors.size());
  for (auto const& a : anchors) {
    runs.emplace_back(anchor_deadline(a), trip_budget(a.trips_, budget));
  }
  // Most trips first, ties by looseness; a run is redundant once a kept one is
  // at least as loose.
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

  r.set_bounds(reach);

  // the pruning search starts where the main search ends
  auto qf = q;
  qf.flip_dir();

  auto starts = std::vector<start>{};
  auto results = pareto_set<journey>{};
  for (auto const& [t, b] : runs) {
    starts.clear();
    get_starts(kPruneDir, tt, rtt, t, qf.start_, qf.td_start_, qf.via_stops_,
               qf.max_start_offset_, qf.start_match_mode_,
               qf.start_match_mode_ != location_match_mode::kIntermodal, starts,
               false, q.prf_idx_, q.transfer_time_settings_);
    r.next_start_time();
    // Staggered alignment: a run allowed b of `budget` trips occupies slots
    // (budget - b) + 1 ... budget, so slot i means "i trips remaining" for
    // every anchor.
    r.set_start_round(static_cast<unsigned>(budget - b));
    for (auto const& s : starts) {
      r.add_start(s.stop_, s.time_at_stop_);
    }
    r.execute(t, static_cast<std::uint8_t>(b - 1U), horizon, results);
  }
  stats = stats + r.get_stats();

  return reach_matrix<kPruneDir>(tt, q, state, r, budget,
                                 /*sub_transfer=*/false);
}

}  // namespace nigiri::routing::bmrap_detail
