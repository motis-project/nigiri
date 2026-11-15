#pragma once

// Slack configuration, anchors and pruning searches of the BM-RAPTOR driver
// (bmrap_profile.cc).

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <array>
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

inline std::array<unixtime_t, kMaxTransfers + 2U> all_rounds(
    unixtime_t const t) {
  auto a = std::array<unixtime_t, kMaxTransfers + 2U>{};
  a.fill(t);
  return a;
}

template <direction Dir>
constexpr bool is_better(auto const a, auto const b) {
  return Dir == direction::kForward ? a < b : a > b;
}

// 2-criteria engine for ping, pong and the pruning searches, by state type
template <direction SearchDir, bool Rt, typename AlgoState>
struct bmrap_algo_for {
  using type = raptor<SearchDir, Rt, kVias, search_mode::kOneToOne>;
};

// kOneToAll: compute_bounds() has no destinations (the GPU raptor lacks the
// mode and serves unchanged)
template <direction SearchDir, bool Rt, typename AlgoState>
struct bmrap_prune_algo_for {
  using type = raptor<SearchDir, Rt, kVias, search_mode::kOneToAll>;
};

#if defined(NIGIRI_CUDA)
template <direction SearchDir, bool Rt>
struct bmrap_algo_for<SearchDir, Rt, gpu::gpu_raptor_state> {
  using type = gpu::gpu_raptor<SearchDir, false>;
};

template <direction SearchDir, bool Rt>
struct bmrap_prune_algo_for<SearchDir, Rt, gpu::gpu_raptor_state> {
  using type = gpu::gpu_raptor<SearchDir, false>;
};
#endif

// engine of the mc phases: the device one only takes the criteria that fit its
// packed label (mc_crit)
template <direction SearchDir, typename Criteria, bool GpuMc>
struct bmrap_mc_algo_for {
  using type = basic_mcraptor<SearchDir, Criteria, /*RangeReuse=*/false>;
  using state = basic_mcraptor_state<Criteria>;
};

#if defined(NIGIRI_CUDA)
// one mc_crit bit per criteria dimension
template <typename Dim>
constexpr std::uint8_t mc_crit_bit() {
  if constexpr (std::is_same_v<Dim, non_transit_dim>) {
    return static_cast<std::uint8_t>(gpu::mc_crit::non_transit);
  } else if constexpr (std::is_same_v<Dim, mode_filter_dim>) {
    return static_cast<std::uint8_t>(gpu::mc_crit::mode_filter);
  } else {
    static_assert(std::is_same_v<Dim, mode_switches_dim>,
                  "dimension has no device mcraptor equivalent");
    return static_cast<std::uint8_t>(gpu::mc_crit::mode_switches);
  }
}

template <typename Criteria>
struct mc_crit_of_t;

template <typename... Dims>
struct mc_crit_of_t<arr_with<Dims...>> {
  static constexpr auto value =
      static_cast<gpu::mc_crit>((std::uint8_t{0U} | ... | mc_crit_bit<Dims>()));
};

template <typename Criteria>
constexpr gpu::mc_crit mc_crit_of() {
  return mc_crit_of_t<Criteria>::value;
}

template <typename Criteria>
inline constexpr bool kMcCritSupported =
    requires { mc_crit_of_t<Criteria>::value; };

// mc phases on the device only with the scalar engine there too
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

// sigma_arr / sigma_tr of the paper
struct slack_cfg {
  double arr_{1.5};
  double trip_{1.5};
  // >= 0: the paper's additive form (Eq. 3.1), N minutes / trips
  double arr_fixed_min_{-1.0};
  double trip_fixed_{-1.0};
  // caps on the extra: a ratio admits hours on long journeys
  double arr_cap_min_{180.0};
  double trip_cap_{2.0};
  // floors on the extra: short journeys would collapse onto their anchors
  double arr_min_min_{20.0};
  double trip_min_{1.0};
};

inline slack_cfg const& get_slack() {
  static constexpr auto const cfg = slack_cfg{};
  return cfg;
}

// the arrival slack on tau_arr(A(J)) - tau_dep (minutes)
inline double relax_arr(double const reference_minutes) {
  auto const& cfg = get_slack();
  auto const extra = cfg.arr_fixed_min_ >= 0.0
                         ? cfg.arr_fixed_min_
                         : reference_minutes * (cfg.arr_ - 1.0);
  return reference_minutes +
         std::clamp(extra, std::min(cfg.arr_min_min_, cfg.arr_cap_min_),
                    cfg.arr_cap_min_);
}

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

template <direction SearchDir>
bool misses_deadline(anchor const& a, unixtime_t const arrival) {
  return is_better<SearchDir>(anchor_deadline(a), arrival);
}

// floor(sigma_tr * |J|), clamped to `max`
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
         misses_deadline<SearchDir>(*a, found);
}

// J_A by a 2-criteria range search: PONG where it covers the extension side,
// else rRAPTOR
template <direction SearchDir, typename AlgoState>
routing_result run_anchor_search(timetable const& tt,
                                 rt_timetable const* rtt,
                                 search_state& s_state,
                                 AlgoState& r_state,
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

// A window's anchor set is truncated at its far end: A(J) of a departure near
// it would depend on the window size. The anchors past the boundary that can
// be A(J) are the (arrival, trips) front of a ping from the boundary,
// re-anchored by a pong. Returns the number of anchors added.
template <direction SearchDir, bool Rt, typename AlgoState>
std::size_t close_anchor_profile_impl(
    timetable const& tt,
    rt_timetable const* rtt,
    query const& q,
    interval<unixtime_t> const anchor_interval,
    AlgoState& state,
    std::vector<anchor>& anchors) {
  constexpr auto const kFwd = (SearchDir == direction::kForward);
  constexpr auto const kDir = duration_t{kFwd ? 1 : -1};
  using ping_t = typename bmrap_algo_for<SearchDir, Rt, AlgoState>::type;
  using pong_t = typename bmrap_algo_for<flip(SearchDir), Rt, AlgoState>::type;

  auto const t = kFwd ? anchor_interval.to_ : anchor_interval.from_;
  auto const base_day = day_idx_t{std::chrono::duration_cast<date::days>(
                                      std::chrono::round<std::chrono::days>(t) -
                                      tt.internal_interval().from_)
                                      .count()};
  auto qf = q;
  qf.flip_dir();
  auto fwd_is_dest = bitvec{};
  auto fwd_dist = std::vector<std::uint16_t>{};
  collect_destinations(tt, q.destination_, q.dest_match_mode_, fwd_is_dest,
                       fwd_dist);
  auto bwd_is_dest = bitvec{};
  auto bwd_dist = std::vector<std::uint16_t>{};
  collect_destinations(tt, qf.destination_, qf.dest_match_mode_, bwd_is_dest,
                       bwd_dist);
  auto is_via = std::array<bitvec, kMaxVias>{};
  auto no_via = std::vector<via_stop>{};
  auto lb = std::vector<std::uint16_t>(tt.n_locations(), std::uint16_t{0U});
  auto const make = [&]<typename Algo>(bitvec& is_dest,
                                       std::vector<std::uint16_t>& dist,
                                       auto const& td_dest) {
    return Algo{tt,
                rtt,
                state,
                is_dest,
                is_via,
                dist,
                td_dest,
                lb,
                no_via,
                base_day,
                q.allowed_claszes_,
                q.require_bike_transport_,
                q.require_car_transport_,
                q.prf_idx_ == 2U,
                q.no_compulsory_reservation_,
                q.transfer_time_settings_,
                q.prf_idx_};
  };
  auto ping =
      make.template operator()<ping_t>(fwd_is_dest, fwd_dist, q.td_dest_);
  auto pong =
      make.template operator()<pong_t>(bwd_is_dest, bwd_dist, qf.td_dest_);

  auto starts = std::vector<start>{};
  get_starts(SearchDir, tt, rtt, t, q.start_, q.td_start_, q.via_stops_,
             q.max_start_offset_, q.start_match_mode_, q.use_start_footpaths_,
             starts, false, q.prf_idx_, q.transfer_time_settings_);
  ping.reset_arrivals();
  ping.next_start_time();
  for (auto const& s : starts) {
    ping.add_start(s.stop_, s.time_at_stop_);
  }
  auto front = pareto_set<journey>{};
  ping.execute(t, q.max_transfers_,
               all_rounds(t + (kFwd ? 1 : -1) *
                                  std::min(q.max_travel_time_ + kMinLookAhead,
                                           kMaxTravelTime)),
               front);
  utl::sort(front, [](journey const& a, journey const& b) {
    return is_better<SearchDir>(a.dest_time_, b.dest_time_);
  });

  auto tight = pareto_set<journey>{};
  pong.reset_arrivals();
  for (auto it = begin(front); it != end(front);) {
    auto const arr = it->dest_time_;
    auto const last = std::find_if(
        it, end(front), [&](journey const& j) { return j.dest_time_ != arr; });
    auto max_tr = std::uint8_t{0U};
    for (auto x = it; x != last; ++x) {
      max_tr = std::max(max_tr, x->transfers_);
    }
    starts.clear();
    get_starts(flip(SearchDir), tt, rtt, arr, qf.start_, qf.td_start_,
               qf.via_stops_, qf.max_start_offset_, qf.start_match_mode_,
               qf.start_match_mode_ != location_match_mode::kIntermodal, starts,
               false, q.prf_idx_, q.transfer_time_settings_);
    pong.next_start_time();
    for (auto const& s : starts) {
      pong.add_start(s.stop_, s.time_at_stop_);
    }
    pong.execute(arr, max_tr, all_rounds(t - kDir), tight);
    it = last;
  }

  for (auto const& j : tight) {
    anchors.push_back({j.dest_time_, j.start_time_,
                       static_cast<std::uint8_t>(j.transfers_ + 1U)});
  }
  auto const n_added = tight.size();

  // a dominated anchor must not be an A(J)
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
  return n_added;
}

template <direction SearchDir, typename AlgoState>
std::size_t close_anchor_profile(timetable const& tt,
                                 rt_timetable const* rtt,
                                 query const& q,
                                 interval<unixtime_t> const anchor_interval,
                                 AlgoState& state,
                                 std::vector<anchor>& anchors) {
  return rtt == nullptr ? close_anchor_profile_impl<SearchDir, false>(
                              tt, rtt, q, anchor_interval, state, anchors)
                        : close_anchor_profile_impl<SearchDir, true>(
                              tt, rtt, q, anchor_interval, state, anchors);
}

// tau_arr^->(v, i) from a finished search's round times: row i is the prefix
// over rounds <= i ("at most i trips"). `sub_transfer` takes the transfer
// buffer off the post-transfer times once, so a forward and a backward matrix
// meet without demanding it twice (the prefix runs over the raw times). A
// device-built matrix stays on the device unless `host_copy`.
template <direction PruneDir, typename AlgoState, typename Algo>
bmrap_bounds reach_matrix(timetable const& tt,
                          query const& q,
                          AlgoState& state,
                          [[maybe_unused]] Algo& algo,
                          std::uint8_t const budget,
                          bool const sub_transfer,
                          [[maybe_unused]] bool const host_copy) {
#if defined(NIGIRI_CUDA)
  if constexpr (std::is_same_v<AlgoState, gpu::gpu_raptor_state>) {
    auto out = bmrap_bounds{};
    algo.build_reach_bounds(out, budget, sub_transfer, host_copy);
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

// tau_dep^<-(v, i): the paper's reverse search per anchor from its deadline,
// capped at its trip budget. The runs are reduced to the front over
// (deadline, budget) and share one rRAPTOR, whose accumulated maximum is the
// union. `horizon`: the far end of the main search.
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
                            bmrap_bounds const* reach,
                            bool const host_copy) {
  constexpr auto const kPruneDir = flip(SearchDir);
  auto const is_looser = [](auto const a, auto const b) {
    return is_better<kPruneDir>(a, b);
  };

  using run_t = std::pair<unixtime_t, std::uint8_t>;
  auto runs = std::vector<run_t>{};
  runs.reserve(anchors.size());
  // The paper's cap: a journey with as many trips as an earlier-arriving
  // anchor has that one (or a better one) as A(J). A journey departing after
  // that anchor is found by a later step, which no longer has it.
  for (auto const& a : anchors) {
    auto b = trip_budget(a.trips_, budget);
    for (auto const& x : anchors) {
      if (x.trips_ > a.trips_ && is_better<SearchDir>(x.found_, a.found_)) {
        b = std::min(b, static_cast<std::uint8_t>(x.trips_ - 1U));
      }
    }
    runs.emplace_back(anchor_deadline(a), b);
  }
  // most trips first: a run is redundant if a kept one is at least as loose
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
    // staggered: slot i = "i trips remaining" for every run
    r.set_start_round(static_cast<unsigned>(budget - b));
    for (auto const& s : starts) {
      r.add_start(s.stop_, s.time_at_stop_);
    }
    r.execute(t, static_cast<std::uint8_t>(b - 1U), all_rounds(horizon),
              results);
  }
  stats = stats + r.get_stats();

  auto bounds = reach_matrix<kPruneDir>(tt, q, state, r, budget,
                                        /*sub_transfer=*/false, host_copy);
  bounds.staggered_ = true;
  return bounds;
}

}  // namespace nigiri::routing::bmrap_detail
