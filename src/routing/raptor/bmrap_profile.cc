#include "nigiri/routing/raptor/bmraptor.h"

#include <algorithm>
#include <optional>
#include <span>
#include <variant>
#include <vector>

#include "utl/erase_if.h"
#include "utl/helpers/algorithm.h"
#include "utl/verify.h"

#include "nigiri/routing/dijkstra.h"
#include "nigiri/routing/direct.h"
#include "nigiri/routing/get_fastest_direct.h"
#include "nigiri/routing/raptor/bmrap_common.h"
#include "nigiri/routing/raptor/mcraptor.h"
#include "nigiri/routing/raptor/pong.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/raptor_search.h"
#include "nigiri/routing/start_times.h"
#include "nigiri/special_stations.h"
#if defined(NIGIRI_CUDA)
#include "nigiri/routing/gpu/mcraptor.h"
#include "nigiri/routing/gpu/raptor.h"
#endif

namespace nigiri::routing {

namespace {

using clk = std::chrono::steady_clock;

constexpr auto const kVias = via_offset_t{0U};

struct stopwatch {
  void lap(clk::duration& acc) {
    auto const now = clk::now();
    acc += now - t_;
    t_ = now;
  }
  clk::time_point t_{clk::now()};
};

// 2-criteria engine of ping, pong and the pruning searches (kOneToAll:
// compute_bounds() has no destinations; the GPU raptor lacks the mode and
// serves unchanged)
template <direction SearchDir,
          bool Rt,
          typename AlgoState,
          search_mode Mode = search_mode::kOneToOne>
struct scalar_algo {
  using type = raptor<SearchDir, Rt, kVias, Mode>;
};

// engine and state of the mc phases
template <direction SearchDir, typename Criteria, bool GpuMc>
struct mc_algo {
  using type = basic_mcraptor<SearchDir, Criteria>;
  using state = basic_mcraptor_state<Criteria>;
};

#if defined(NIGIRI_CUDA)
template <direction SearchDir, bool Rt, search_mode Mode>
struct scalar_algo<SearchDir, Rt, gpu::gpu_raptor_state, Mode> {
  using type = gpu::gpu_raptor<SearchDir, false>;
};

template <direction SearchDir, typename Criteria>
struct mc_algo<SearchDir, Criteria, true> {
  using type = gpu::gpu_mcraptor<SearchDir, Criteria>;
  using state = gpu::gpu_mcraptor_state;
};
#endif

// incl. rt virtual locations
std::uint32_t n_search_locations(timetable const& tt, rt_timetable const* rtt) {
  return rtt == nullptr ? tt.n_locations() : rtt->n_locations();
}

day_idx_t base_day_of(timetable const& tt, unixtime_t const t) {
  return day_idx_t{std::chrono::duration_cast<date::days>(
                       std::chrono::round<std::chrono::days>(t) -
                       tt.internal_interval().from_)
                       .count()};
}

std::array<unixtime_t, kMaxTransfers + 2U> all_rounds(unixtime_t const t) {
  auto a = std::array<unixtime_t, kMaxTransfers + 2U>{};
  a.fill(t);
  return a;
}

// Bounds the arrival relative to the step, so it looks past max_travel_time
// (the next useful departure can be hours away); travel time is filtered on
// the results.
template <direction SearchDir>
unixtime_t worst_at_dest(query const& q, unixtime_t const t) {
  return t + (SearchDir == direction::kForward ? 1 : -1) *
                 std::min(q.max_travel_time_ + kMinLookAhead, kMaxTravelTime);
}

std::uint8_t trips(journey const& j) {
  return static_cast<std::uint8_t>(j.transfers_ + 1U);
}

bool same_criteria(journey const& a, journey const& b) {
  return a.transfers_ == b.transfers_ && a.criteria_cost_ == b.criteria_cost_ &&
         a.criteria_avoided_modes_ == b.criteria_avoided_modes_ &&
         a.criteria_mode_switches_ == b.criteria_mode_switches_ &&
         a.criteria_night_transfer_ == b.criteria_night_transfer_;
}

// The engines keep references to all of these.
template <typename Algo, typename State>
Algo make_engine(
    timetable const& tt,
    rt_timetable const* rtt,
    State& state,
    query const& q,
    day_idx_t const base,
    bitvec& is_dest,
    std::vector<std::uint16_t> const& dist,
    hash_map<location_idx_t, std::vector<td_offset>> const& td_dest,
    std::vector<std::uint16_t> const& lb) {
  static auto const kNoVias = std::array<bitvec, kMaxVias>{};
  static auto const kNoViaStops = std::vector<via_stop>{};
  return Algo{tt,
              rtt,
              state,
              is_dest,
              kNoVias,
              dist,
              td_dest,
              lb,
              kNoViaStops,
              base,
              q.allowed_claszes_,
              q.require_bike_transport_,
              q.require_car_transport_,
              q.prf_idx_ == 2U,
              q.no_compulsory_reservation_,
              q.transfer_time_settings_,
              q.prf_idx_,
              mc_context{.avoided_ = q.avoided_claszes_}};
}

// the next start time of `algo`: q's starts at t (q flipped for the pongs)
template <typename Algo>
void seed(Algo& algo,
          timetable const& tt,
          rt_timetable const* rtt,
          query const& q,
          direction const dir,
          bool const use_start_footpaths,
          unixtime_t const t,
          std::vector<start>& starts) {
  starts.clear();
  get_starts(dir, tt, rtt, t, q.start_, q.td_start_, q.via_stops_,
             q.max_start_offset_, q.start_match_mode_, use_start_footpaths,
             starts, false, q.prf_idx_, q.transfer_time_settings_);
  algo.next_start_time();
  for (auto const& s : starts) {
    algo.add_start(s.stop_, s.time_at_stop_);
  }
}

// js sorted by arrival, then fn(arrival, group, most transfers) per arrival
template <direction SearchDir, typename Fn>
void for_each_arrival_group(pareto_set<journey>& js, Fn&& fn) {
  utl::sort(js, [](journey const& a, journey const& b) {
    return is_better<SearchDir>(a.dest_time_, b.dest_time_);
  });
  for (auto it = begin(js); it != end(js);) {
    auto const arr = it->dest_time_;
    auto const last = std::find_if(
        it, end(js), [&](journey const& j) { return j.dest_time_ != arr; });
    auto const group = std::span<journey const>{it, last};
    fn(arr, group,
       utl::max_element(group, [](journey const& a, journey const& b) {
         return a.transfers_ < b.transfers_;
       })->transfers_);
    it = last;
  }
}

// Keeps `anchors` a (departure, arrival, trips) Pareto set: a dominated anchor
// must not be an A(J).
template <direction SearchDir>
void add_anchor(std::vector<anchor>& anchors, anchor const& a) {
  auto const dominates = [](anchor const& x, anchor const& y) {
    return !is_better<SearchDir>(x.anchored_, y.anchored_) &&
           !is_better<SearchDir>(y.found_, x.found_) && x.trips_ <= y.trips_;
  };
  if (utl::none_of(anchors, [&](anchor const& o) { return dominates(o, a); })) {
    utl::erase_if(anchors, [&](anchor const& o) { return dominates(a, o); });
    anchors.push_back(a);
  }
}

// A window's anchor set is truncated at its far end t: A(J) of a departure
// near it would depend on the window size. The anchors past the boundary that
// can be A(J) are the (arrival, trips) front of a ping from the boundary,
// re-anchored by a pong.
template <direction SearchDir, bool Rt, typename AlgoState>
void close_anchor_profile(timetable const& tt,
                          rt_timetable const* rtt,
                          query const& q,
                          unixtime_t const t,
                          AlgoState& state,
                          std::vector<anchor>& anchors) {
  constexpr auto const kDir =
      duration_t{SearchDir == direction::kForward ? 1 : -1};
  auto const base = base_day_of(tt, t);
  auto qf = q;
  qf.flip_dir();
  auto fwd_is_dest = bitvec{};
  auto fwd_dist = std::vector<std::uint16_t>{};
  collect_destinations(tt, rtt, q.destination_, q.dest_match_mode_, fwd_is_dest,
                       fwd_dist);
  auto bwd_is_dest = bitvec{};
  auto bwd_dist = std::vector<std::uint16_t>{};
  collect_destinations(tt, rtt, qf.destination_, qf.dest_match_mode_,
                       bwd_is_dest, bwd_dist);
  auto const lb = std::vector<std::uint16_t>(n_search_locations(tt, rtt), 0U);
  auto ping = make_engine<typename scalar_algo<SearchDir, Rt, AlgoState>::type>(
      tt, rtt, state, q, base, fwd_is_dest, fwd_dist, q.td_dest_, lb);
  auto pong =
      make_engine<typename scalar_algo<flip(SearchDir), Rt, AlgoState>::type>(
          tt, rtt, state, q, base, bwd_is_dest, bwd_dist, qf.td_dest_, lb);

  auto starts = std::vector<start>{};
  auto front = pareto_set<journey>{};
  ping.reset_arrivals();
  seed(ping, tt, rtt, q, SearchDir, q.use_start_footpaths_, t, starts);
  ping.execute(t, q.max_transfers_, all_rounds(worst_at_dest<SearchDir>(q, t)),
               front);

  auto tight = pareto_set<journey>{};
  pong.reset_arrivals();
  for_each_arrival_group<SearchDir>(front, [&](unixtime_t const arr, auto,
                                               std::uint8_t const max_tr) {
    seed(pong, tt, rtt, qf, flip(SearchDir),
         qf.start_match_mode_ != location_match_mode::kIntermodal, arr, starts);
    pong.execute(arr, max_tr, all_rounds(t - kDir), tight);
  });
  for (auto const& j : tight) {
    add_anchor<SearchDir>(anchors, {j.dest_time_, j.start_time_, trips(j)});
  }
}

// Banned same-stop transfers leave no post-transfer time to bound walk-ins by,
// real-time locations take their parent's transfer time: neither prunes.
bool has_reach_bound(timetable const& tt, std::uint32_t const l) {
  return l < tt.n_locations() &&
         tt.locations_.transfer_time_[location_idx_t{l}] != kNoTransferAllowed;
}

// tau_arr^->(v, i) from a finished search's round times: row i is the prefix
// over rounds <= i ("at most i trips"). `sub_transfer` takes the transfer
// buffer off the post-transfer times once, so a forward and a backward matrix
// meet without demanding it twice. A device-built matrix stays on the device
// unless `host_copy`.
template <direction PruneDir, typename AlgoState, typename Algo>
bmrap_bounds reach_matrix(timetable const& tt,
                          query const& q,
                          AlgoState& state,
                          [[maybe_unused]] Algo& algo,
                          std::uint8_t const budget,
                          bool const sub_transfer,
                          [[maybe_unused]] bool const host_copy) {
  auto bounds = bmrap_bounds{};
#if defined(NIGIRI_CUDA)
  if constexpr (std::is_same_v<AlgoState, gpu::gpu_raptor_state>) {
    algo.build_reach_bounds(bounds, budget, sub_transfer, host_copy);
    return bounds;
  } else
#endif
  {
    constexpr auto const kInvalid = kInvalidDelta<PruneDir>;
    auto const round_times = state.template get_round_times<kVias>();
    // incl. rt virtual locations
    auto const n_locations = state.n_locations_;
    bounds.resize(n_locations, budget, kInvalid);
    for (auto l = 0U; l != n_locations; ++l) {
      if (!has_reach_bound(tt, l)) {
        for (auto i = 0U; i <= budget; ++i) {
          bounds.at(i, l) = kInvalidDelta<flip(PruneDir)>;
        }
        continue;
      }
      auto const buffer =
          sub_transfer
              ? (PruneDir == direction::kForward ? 1 : -1) *
                    adjusted_transfer_time(
                        q.transfer_time_settings_,
                        tt.locations_.transfer_time_[location_idx_t{l}].count())
              : 0;
      auto prefix = kInvalid;
      for (auto i = 0U; i <= budget; ++i) {
        auto const cur = round_times[i][l][kVias];
        prefix = is_better<PruneDir>(cur, prefix) ? cur : prefix;
        bounds.at(i, l) = prefix == kInvalid
                              ? kInvalid
                              : static_cast<delta_t>(prefix - buffer);
      }
    }
    return bounds;
  }
}

// tau_dep^<-(v, i) (see bound_runs()): the runs are reduced to the front over
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

  // most trips first: a run is redundant if a kept one is at least as loose
  auto runs = bound_runs<SearchDir>(anchors, horizon, budget);
  utl::sort(runs, [](bound_run const& a, bound_run const& b) {
    return a.second != b.second ? a.second > b.second
                                : is_better<kPruneDir>(a.first, b.first);
  });
  auto reduced = std::vector<bound_run>{};
  for (auto const& x : runs) {
    if (utl::none_of(reduced, [&](bound_run const& o) {
          return !is_better<kPruneDir>(x.first, o.first);
        })) {
      reduced.push_back(x);
    }
  }

  auto is_dest = bitvec{n_search_locations(tt, rtt)};
  auto const dist_to_dest = std::vector<std::uint16_t>{};
  auto const td_dist_to_dest =
      hash_map<location_idx_t, std::vector<td_offset>>{};
  auto const lb = std::vector<std::uint16_t>(n_search_locations(tt, rtt), 0U);
  auto r = make_engine<typename scalar_algo<kPruneDir, Rt, AlgoState,
                                            search_mode::kOneToAll>::type>(
      tt, rtt, state, q, base, is_dest, dist_to_dest, td_dist_to_dest, lb);
  r.set_bounds(reach);

  auto qf = q;
  qf.flip_dir();
  auto starts = std::vector<start>{};
  auto results = pareto_set<journey>{};
  for (auto const& [t, b] : reduced) {
    // staggered: slot i = "i trips remaining" for every run
    r.set_start_round(static_cast<unsigned>(budget - b));
    seed(r, tt, rtt, qf, kPruneDir,
         qf.start_match_mode_ != location_match_mode::kIntermodal, t, starts);
    r.execute(t, static_cast<std::uint8_t>(b - 1U), all_rounds(horizon),
              results);
  }
  stats = stats + r.get_stats();

  auto bounds = reach_matrix<kPruneDir>(tt, q, state, r, budget,
                                        /*sub_transfer=*/false, host_copy);
  bounds.staggered_ = true;
  return bounds;
}

// PROFILE BM-RAPTOR: one single-departure BM-RAPTOR per step of a PONG-style
// scan, so every departure gets bounds anchored at itself (one matrix spread
// over a window bottoms out at the window start).
//
//   1. ping          2-criteria earliest arrival from the step
//   2. pong          re-anchors each anchor to its latest departure
//   3. slacked pong  tau_dep^<-(v, i) from the slack-relaxed anchor deadlines
//   4. mc ping       bounded multicriteria search from the step
//   5. mc pong       re-anchors each mc journey to its latest departure
//   5b. realize      forward rerun for the earliest connections
//
// Only the ping uses lower bounds: bounded, bound_prunes() is the tighter
// cutoff. One scalar state serves ping, pong and slacked pong, one mc state all
// mc phases: they run in sequence, and a phase guided by another's labels reads
// a copy of them.
template <direction SearchDir,
          bool Rt,
          typename Criteria,
          typename AlgoState,
          bool GpuMc>
routing_result bmrap_profile(
    timetable const& tt,
    rt_timetable const* rtt,
    search_state& s_state,
    AlgoState& r_state,
    typename mc_algo<SearchDir, Criteria, GpuMc>::state& mc_state,
    query q,
    std::optional<std::chrono::seconds> const timeout) {
  constexpr auto const kFwd = (SearchDir == direction::kForward);
  static constexpr auto const kDir = duration_t{kFwd ? 1 : -1};
  using ping_t = typename scalar_algo<SearchDir, Rt, AlgoState>::type;
  using pong_t = typename scalar_algo<flip(SearchDir), Rt, AlgoState>::type;
  using mc_ping_t = typename mc_algo<SearchDir, Criteria, GpuMc>::type;
  using mc_pong_t = typename mc_algo<flip(SearchDir), Criteria, GpuMc>::type;

  q.sanitize(tt);
  utl::verify(mcraptor_supported(q, rtt),
              "bmrap_profile: query not supported by the mcraptor engine");

  auto const t0 = clk::now();
  s_state.results_.clear();

  auto const is_better = [](auto const a, auto const b) {
    return routing::is_better<SearchDir>(a, b);
  };
  auto const fastest_direct = get_fastest_direct(tt, q, SearchDir);
  auto const search_interval = std::visit(
      utl::overloaded{
          [](interval<unixtime_t> const i) { return i; },
          [](unixtime_t const t) { return interval<unixtime_t>{t, t}; }},
      q.start_time_);
  auto const base_day =
      base_day_of(tt, search_interval.from_ +
                          ((search_interval.to_ - search_interval.from_) / 2));

  auto fwd_is_dest = bitvec{};
  auto fwd_dist = std::vector<std::uint16_t>{};
  collect_destinations(tt, rtt, q.destination_, q.dest_match_mode_, fwd_is_dest,
                       fwd_dist);
  auto qf = q;
  qf.flip_dir();
  auto bwd_is_dest = bitvec{};
  auto bwd_dist = std::vector<std::uint16_t>{};
  collect_destinations(tt, rtt, qf.destination_, qf.dest_match_mode_,
                       bwd_is_dest, bwd_dist);

  auto sw = stopwatch{};
  auto lb = std::vector<std::uint16_t>{};
  auto const no_lb = std::vector<std::uint16_t>{};
  if constexpr (ping_t::kUseLowerBounds) {
    dijkstra(tt, rtt, q, SearchDir, lb);
  }
  auto ms_lb = clk::duration{};
  sw.lap(ms_lb);

  auto ping = make_engine<ping_t>(tt, rtt, r_state, q, base_day, fwd_is_dest,
                                  fwd_dist, q.td_dest_, lb);
  auto pong = make_engine<pong_t>(tt, rtt, r_state, q, base_day, bwd_is_dest,
                                  bwd_dist, qf.td_dest_, no_lb);
  auto mc_ping = make_engine<mc_ping_t>(tt, rtt, mc_state, q, base_day,
                                        fwd_is_dest, fwd_dist, q.td_dest_, lb);
  auto mc_pong =
      make_engine<mc_pong_t>(tt, rtt, mc_state, q, base_day, bwd_is_dest,
                             bwd_dist, qf.td_dest_, no_lb);
  // the starts' walks, as search.h tells the plain engines: td offsets may
  // wait, locations of a start stop are reached by a transfer
  mc_ping.set_td_start(q.td_start_);
  mc_ping.set_start_stops(q.start_, q.start_match_mode_);
  mc_pong.set_td_start(qf.td_start_);
  mc_pong.set_start_stops(qf.start_, qf.start_match_mode_);
  auto ms_setup = clk::duration{};
  sw.lap(ms_setup);

  auto prune_stats = raptor_stats{};
  auto all_anchors = std::vector<anchor>{};
  auto n_steps = std::uint64_t{0U};
  auto n_anchor_recomputes = std::uint64_t{0U};
  auto n_mc_pong_skipped = std::uint64_t{0U};
  auto n_pong_missing = std::uint64_t{0U};
  auto exit_reason = std::uint64_t{0U};  // 0=cond 2=anchors
  auto ms_ping = clk::duration{};
  auto ms_pong = clk::duration{};
  auto ms_bounds = clk::duration{};
  auto ms_mc = clk::duration{};
  auto ms_mc_pong = clk::duration{};
  auto ms_fwd_bounds = clk::duration{};
  auto ms_range = clk::duration{};
  auto ms_realize = clk::duration{};
  auto fwd_bounds = bmrap_bounds{};
  auto bounds = bmrap_bounds{};
  auto const budget_cap = static_cast<std::uint8_t>(
      std::min<unsigned>(q.max_transfers_ + 1U, kMaxTransfers + 1U));

  auto starts = std::vector<start>{};
  auto const seed_fwd = [&](auto& algo, unixtime_t const t) {
    seed(algo, tt, rtt, q, SearchDir, q.use_start_footpaths_, t, starts);
  };
  auto const seed_bwd = [&](auto& algo, unixtime_t const t) {
    seed(algo, tt, rtt, qf, flip(SearchDir),
         qf.start_match_mode_ != location_match_mode::kIntermodal, t, starts);
  };
  auto const run_ping = [&](unixtime_t const t) {
    auto results = pareto_set<journey>{};
    ping.reset_arrivals();
    // the arrival slack on the target pruning makes the round times a valid
    // tau_arr^->(v, i) matrix (paper, Sec. 4.3)
    ping.set_dest_relax(t);
    seed_fwd(ping, t);
    ping.execute(t, q.max_transfers_,
                 all_rounds(worst_at_dest<SearchDir>(q, t)), results);
    return results;
  };
  auto const run_mc_ping =
      [&](unixtime_t const t, std::uint8_t const max_transfers,
          unixtime_t const worst, pareto_set<journey>& results) {
        mc_ping.reset_arrivals();
        seed_fwd(mc_ping, t);
        mc_ping.execute(t, max_transfers, all_rounds(worst), results);
      };

  // The scan only grows the window on the side it runs towards ("aligned").
  // Opposed (paging the other way), a 2-criteria range search runs up front:
  // it extends the window as asked and its journeys are the anchor set, so
  // numItineraries is met on 2-criteria journeys (possibly more come back).
  auto const pretrip =
      std::holds_alternative<interval<unixtime_t>>(q.start_time_);
  auto const aligned = !pretrip || ((SearchDir == direction::kBackward) !=
                                    q.extend_interval_later_);

  auto scan_interval = search_interval;
  if (!aligned) {
    sw = stopwatch{};
    auto range_s_state = search_state{};
    auto const ar =
        raptor_search(tt, rtt, range_s_state, r_state, q, SearchDir, timeout);
    scan_interval = ar.interval_;
    for (auto const& j : *ar.journeys_) {
      add_anchor<SearchDir>(all_anchors,
                            {j.start_time_, j.dest_time_, trips(j)});
    }
    // a windowed anchor set is truncated at its far end
    if (!all_anchors.empty()) {
      close_anchor_profile<SearchDir, Rt>(
          tt, rtt, q, kFwd ? scan_interval.to_ : scan_interval.from_, r_state,
          all_anchors);
    }
    sw.lap(ms_range);
  }

  auto start_time =
      kFwd ? scan_interval.from_ : scan_interval.to_ - duration_t{1};
  auto const end_time =
      kFwd ? scan_interval.to_ : scan_interval.from_ - duration_t{1};

  // ---- 5b. REALIZE ----
  // The mc pong's legs are the latest connections (no slack at any transfer).
  // Forward queries rerun each departure forward, guided by the mc pong's
  // labels (take_guide()), and splice in the earliest connections; for arriveBy
  // the mc pong runs forward and already has them. Guided searches keep only
  // the labels that join one of the other direction's labels into one of the
  // targets. Complete as long as the mc phases project with a zero lower bound
  // (they are bounded): destination pruning then never cuts a label that can
  // still complete to a target.
  auto realized = std::vector<journey>{};
  auto n_realized = std::uint64_t{0U};
  auto n_unrealized = std::uint64_t{0U};
  auto const realize = [&](unixtime_t const arr,
                           pareto_set<journey> const& pong_results) {
    auto deps = std::vector<unixtime_t>{};
    for (auto const& j : pong_results) {
      if (j.start_time_ == arr && !j.error_ &&
          utl::find(deps, j.dest_time_) == end(deps)) {
        deps.push_back(j.dest_time_);
      }
    }
    for (auto const d : deps) {
      auto targets = std::vector<journey const*>{};
      auto max_tr = std::uint8_t{0U};
      for (auto const& j : pong_results) {
        if (j.start_time_ == arr && !j.error_ && j.dest_time_ == d) {
          targets.push_back(&j);
          max_tr = std::max(max_tr, j.transfers_);
        }
      }
      auto fwd = pareto_set<journey>{};
      mc_ping.set_guide(targets);
      run_mc_ping(d, max_tr, arr + kDir, fwd);
      mc_ping.set_guide({});
      for (auto& f : fwd) {
        if (!f.is_reconstructed_ && !f.error_) {
          try {
            mc_ping.reconstruct(q, f);
          } catch (std::exception const& e) {
            f.error_ = true;
            log(log_lvl::error, "bmrap_profile", "realize failed: {}",
                e.what());
          }
        }
        if (!f.error_ && f.is_reconstructed_) {
          realized.emplace_back(f);
        }
      }
    }
  };

  auto anchors = std::vector<anchor>{};
  auto max_trips = std::uint8_t{0U};
  auto budget = std::uint8_t{0U};
  auto anchors_valid_until = std::optional<unixtime_t>{};

  // In the scan, results have the mc pong's convention (dest_time_ = departure,
  // start_time_ = arrival). The restriction applies when counting, not on
  // insertion: A(J) is only final once the scan has passed J's departure.
  auto const n_results = [&](bool const include_too_slow) {
    return utl::count_if(s_state.results_, [&](journey const& j) {
      if (!is_better(j.dest_time_, start_time)) {
        return false;
      }
      if (outside_restriction<SearchDir>(
              all_anchors, j.dest_time_, j.start_time_,
              static_cast<unsigned>(j.transfers_) + 1U)) {
        return false;
      }
      if (!include_too_slow && !(j.travel_time() < fastest_direct &&
                                 j.travel_time() < q.max_travel_time_)) {
        return false;
      }
      return !utl::any_of(s_state.results_, [&](journey const& o) {
        if (&o == &j || !o.dominates(j)) {
          return false;
        }
        // every Pareto-optimal journey counts, as in the response; exact
        // duplicates once
        return !j.dominates(o) || &o < &j;
      });
    });
  };

  // stepping past the far end grows the window: aligned only
  while ((is_better(start_time, end_time) ||
          (aligned && n_results(true) + n_results(false) <
                          2 * static_cast<int>(q.min_connection_count_))) &&
         tt.external_interval().contains(start_time) &&
         !(timeout && (clk::now() - t0) >= *timeout)) {
    // The anchors (and phases 1-3) only go stale once the step passes the
    // earliest anchor departure; steps in between skip them.
    if (!anchors_valid_until.has_value() ||
        is_better(*anchors_valid_until, start_time)) {
      ++n_anchor_recomputes;

      if (!aligned) {
        // The (arrival, trips) front of the anchors still available, as the
        // aligned pong reports it. An anchor off the front is A(J) only for
        // journeys departing after the one beating it: a later step's front.
        auto available = std::vector<anchor>{};
        for (auto const& a : all_anchors) {
          if (!is_better(a.anchored_, start_time)) {
            available.push_back(a);
          }
        }
        utl::sort(available, [&](anchor const& a, anchor const& b) {
          return a.found_ != b.found_ ? is_better(a.found_, b.found_)
                                      : a.trips_ < b.trips_;
        });
        anchors.clear();
        for (auto const& a : available) {
          if (anchors.empty() || a.trips_ < anchors.back().trips_) {
            anchors.push_back(a);
          }
        }
        if (anchors.empty()) {
          exit_reason = 2U;
          break;
        }
        max_trips = anchors.front().trips_;
        // the step's own tau_arr^->(v, i): one from the window start is loose
        // by the window width and sized for that step's trips
        sw = stopwatch{};
        run_ping(start_time);
        fwd_bounds = reach_matrix<SearchDir>(
            tt, q, r_state, ping, trip_budget(max_trips, budget_cap),
            /*sub_transfer=*/true, /*host_copy=*/!GpuMc);
        sw.lap(ms_fwd_bounds);
        mc_pong.set_bounds(&fwd_bounds);
      } else {
        // ---- 1. PING ----
        sw = stopwatch{};
        auto ping_results = run_ping(start_time);
        sw.lap(ms_ping);

        // before the pong overwrites the round times; sized by the ping's
        // trips (the pong moves departures, not trip counts)
        auto ping_trips = std::uint8_t{0U};
        for (auto const& j : ping_results) {
          ping_trips = std::max(ping_trips, trips(j));
        }
        fwd_bounds = reach_matrix<SearchDir>(
            tt, q, r_state, ping, trip_budget(ping_trips, budget_cap),
            /*sub_transfer=*/true, /*host_copy=*/!GpuMc);
        sw.lap(ms_fwd_bounds);
        mc_pong.set_bounds(&fwd_bounds);
        pong.set_bounds(&fwd_bounds);

        // ---- 2. PONG ----
        auto tight = pareto_set<journey>{};
        pong.reset_arrivals();
        for_each_arrival_group<SearchDir>(
            ping_results,
            [&](unixtime_t const arr, std::span<journey const> const group,
                std::uint8_t const max_tr) {
              auto const loosest =
                  utl::min_element(group, [&](journey const& a,
                                              journey const& b) {
                    return is_better(a.start_time_, b.start_time_);
                  })->start_time_;
              seed_bwd(pong, arr);
              pong.execute(arr, max_tr, all_rounds(loosest - kDir), tight);
              // Each ping journey has its latest departure (pong_search
              // throws). A miss only costs this step its anchor: log, count.
              for (auto const& j : group) {
                if (utl::none_of(tight, [&](journey const& t) {
                      return t.transfers_ == j.transfers_ &&
                             t.start_time_ == arr;
                    })) {
                  ++n_pong_missing;
                  log(log_lvl::error, "bmrap_profile",
                      "no pong for transfers={}, start_time={} found",
                      j.transfers_, arr);
                }
              }
            });
        sw.lap(ms_pong);

        anchors.clear();
        max_trips = 0U;
        for (auto const& j : tight) {
          anchors.push_back({j.dest_time_, j.start_time_, trips(j)});
          max_trips = std::max(max_trips, trips(j));
        }
        if (anchors.empty()) {
          exit_reason = 2U;
          break;
        }
        // A later step can re-anchor a journey to a departure an earlier step
        // covered with a better arrival.
        for (auto const& a : anchors) {
          add_anchor<SearchDir>(all_anchors, a);
        }
      }

      budget = trip_budget(max_trips, budget_cap);
      anchors_valid_until =
          utl::min_element(anchors, [&](anchor const& a, anchor const& b) {
            return is_better(a.anchored_, b.anchored_);
          })->anchored_;

      // ---- 3. SLACKED PONG ----
      sw = stopwatch{};
      bounds = compute_bounds<SearchDir, Rt>(
          tt, rtt, q, anchors, base_day, /*horizon=*/start_time - kDir, budget,
          r_state, prune_stats, &fwd_bounds, /*host_copy=*/!GpuMc);
      sw.lap(ms_bounds);
      mc_ping.set_bounds(&bounds);
    }
    ++n_steps;

    // ---- 4. MC PING ----
    // horizon: the latest anchor deadline (the step can lie days before the
    // journeys it finds)
    auto mc_worst = start_time;
    for (auto const& a : anchors) {
      if (auto const d = anchor_deadline(a); is_better(mc_worst, d)) {
        mc_worst = d;
      }
    }
    sw = stopwatch{};
    auto mc_results = pareto_set<journey>{};
    run_mc_ping(start_time, static_cast<std::uint8_t>(budget - 1U),
                mc_worst + kDir, mc_results);
    mc_ping.take_guide();
    sw.lap(ms_mc);

    // ---- 5. MC PONG ----
    // An mc journey an earlier result dominates is still re-anchored: its
    // latest departure can lie after the dominating one, where no later step
    // finds it. An arrival group is skipped when all its journeys already are
    // re-anchored in the results (the mc ping finds most of the previous
    // step's again). Results have the scan's convention, mc journeys not.
    auto const re_anchored = [&](journey const& x) {
      return utl::any_of(s_state.results_, [&](journey const& y) {
        return y.start_time_ == x.dest_time_ &&
               !is_better(y.dest_time_, x.start_time_) && same_criteria(x, y);
      });
    };

    auto step_results = pareto_set<journey>{};
    auto skipped_deps = std::vector<unixtime_t>{};
    for_each_arrival_group<SearchDir>(
        mc_results,
        [&](unixtime_t const arr, std::span<journey const> const group,
            std::uint8_t const group_tr) {
          if (utl::all_of(group, re_anchored)) {
            // its departures still set the advance (step 6)
            ++n_mc_pong_skipped;
            for (auto const& r : s_state.results_) {
              if (r.start_time_ == arr && r.transfers_ <= group_tr) {
                skipped_deps.push_back(r.dest_time_);
              }
            }
            return;
          }
          // guided by the mc ping: a re-anchored journey's prefixes depart no
          // earlier than the step, so the ping has a label at each of its stops
          auto targets = std::vector<journey const*>{};
          for (auto const& j : group) {
            targets.push_back(&j);
          }
          // one search per group: the bags outlive next_start_time()
          mc_pong.reset_arrivals();
          seed_bwd(mc_pong, arr);
          mc_pong.set_guide(targets);
          mc_pong.execute(arr, group_tr, all_rounds(start_time - kDir),
                          step_results);
          // before the realization reuses the state
          for (auto& j : step_results) {
            if (!j.is_reconstructed_ && !j.error_) {
              try {
                mc_pong.reconstruct(qf, j);
              } catch (std::exception const& e) {
                j.error_ = true;
                log(log_lvl::error, "bmrap_profile", "reconstruct failed: {}",
                    e.what());
              }
            }
          }

          if constexpr (kFwd) {
            mc_pong.take_guide();
            sw.lap(ms_mc_pong);
            realize(arr, step_results);
            sw.lap(ms_realize);
          }
        });
    sw.lap(ms_mc_pong);
    for (auto const& j : step_results) {
      if (!j.error_) {
        s_state.results_.add(journey{j});
      }
    }

    // ---- 6. ADVANCE ----
    // past the loosest departure at or after the step over anchors, mc journeys
    // and skipped groups: the mc set can change between anchor departures; one
    // before the step (pong bound loosest - 1 min) would stall the scan
    auto loosest_dep = std::optional<unixtime_t>{};
    auto const consider = [&](unixtime_t const d) {
      if (!is_better(d, start_time) &&
          (!loosest_dep.has_value() || is_better(d, *loosest_dep))) {
        loosest_dep = d;
      }
    };
    for (auto const& a : anchors) {
      consider(a.anchored_);
    }
    for (auto const& j : step_results) {
      consider(j.dest_time_);
    }
    for (auto const d : skipped_deps) {
      consider(d);
    }
    auto const next = loosest_dep.value_or(start_time) + kDir;
    if (!is_better(start_time, next)) {
      throw utl::fail("no bmrapp progress: start_time={}, next={}", start_time,
                      next);
    }
    start_time = next;
  }
  // opposed, the range search fixed the window
  auto const scanned =
      aligned ? (kFwd ? interval<unixtime_t>{scan_interval.from_, start_time}
                      : interval<unixtime_t>{start_time + duration_t{1},
                                             scan_interval.to_})
              : scan_interval;
  utl::erase_if(s_state.results_, [&](journey const& j) {
    return !j.is_reconstructed_ || j.error_ ||
           !scanned.contains(j.dest_time_) ||
           j.travel_time() >= fastest_direct ||
           j.travel_time() > q.max_travel_time_;
  });
  for (auto& x : s_state.results_) {
    std::swap(x.start_time_, x.dest_time_);
  }

  // legs reconstructed on qf have mirrored special stations (unlike the
  // realized ones spliced in below)
  auto const swap_special = [](location_idx_t const l) {
    switch (to_idx(l)) {
      case to_idx(get_special_station(special_station::kStart)):
        return get_special_station(special_station::kEnd);
      case to_idx(get_special_station(special_station::kEnd)):
        return get_special_station(special_station::kStart);
      default: return l;
    }
  };
  for (auto& j : s_state.results_) {
    if (!j.legs_.empty()) {
      j.legs_.front().from_ = swap_special(j.legs_.front().from_);
      j.legs_.back().to_ = swap_special(j.legs_.back().to_);
    }
  }

  auto n_restricted = std::uint64_t{0U};
  utl::erase_if(s_state.results_, [&](journey const& j) {
    auto const drop = outside_restriction<SearchDir>(
        all_anchors, j.start_time_, j.dest_time_,
        static_cast<unsigned>(j.transfers_) + 1U);
    n_restricted += drop ? 1U : 0U;
    return drop;
  });

  if constexpr (kFwd) {
    for (auto& x : s_state.results_) {
      auto const it = utl::find_if(realized, [&](journey const& f) {
        return f.start_time_ == x.start_time_ && f.dest_time_ == x.dest_time_ &&
               same_criteria(f, x);
      });
      if (it != end(realized)) {
        x.legs_ = it->legs_;
        ++n_realized;
      } else {
        ++n_unrealized;
      }
    }
  }

  enrich_with_slow_direct<SearchDir>(tt, rtt, q, scanned, s_state.results_);
  utl::sort(s_state.results_, [](journey const& a, journey const& b) {
    return std::tuple{a.start_time_, a.transfers_, a.dest_time_,
                      a.criteria_cost_} < std::tuple{b.start_time_,
                                                     b.transfers_, b.dest_time_,
                                                     b.criteria_cost_};
  });

  auto const ms = [](auto const d) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(d).count());
  };
  auto algo_stats = (ping.get_stats() + pong.get_stats() + mc_ping.get_stats() +
                     mc_pong.get_stats())
                        .to_map();
  auto const mc_stats = mc_ping.get_stats() + mc_pong.get_stats();
  algo_stats["bmrapp_mc_labels"] =
      mc_stats.n_earliest_arrival_updated_by_route_ +
      mc_stats.n_earliest_arrival_updated_by_footpath_;
  for (auto const& [k, v] : prune_stats.to_map()) {
    algo_stats["prune_" + k] = v;
  }
  algo_stats["bmrapp_steps"] = n_steps;
  algo_stats["bmrapp_anchor_recomputes"] = n_anchor_recomputes;
  algo_stats["bmrapp_ms_realize"] = ms(ms_realize);
  algo_stats["bmrapp_realized"] = n_realized;
  algo_stats["bmrapp_unrealized"] = n_unrealized;
  algo_stats["bmrapp_exit"] = exit_reason;
  algo_stats["bmrapp_anchors"] = all_anchors.size();
  algo_stats["bmrapp_restricted_away"] = n_restricted;
  algo_stats["bmrapp_ms_ping"] = ms(ms_ping);
  algo_stats["bmrapp_ms_pong"] = ms(ms_pong);
  algo_stats["bmrapp_ms_bounds"] = ms(ms_bounds);
  algo_stats["bmrapp_ms_mc_ping"] = ms(ms_mc);
  algo_stats["bmrapp_ms_mc_pong"] = ms(ms_mc_pong);
  algo_stats["bmrapp_ms_fwd_bounds"] = ms(ms_fwd_bounds);
  algo_stats["bmrapp_ms_lb"] = ms(ms_lb);
  algo_stats["bmrapp_ms_setup"] = ms(ms_setup);
  algo_stats["bmrapp_aligned"] = aligned ? 1U : 0U;
  algo_stats["bmrapp_ms_range"] = ms(ms_range);
  algo_stats["bmrapp_mc_pong_skipped"] = n_mc_pong_skipped;
  algo_stats["bmrapp_pong_missing"] = n_pong_missing;

  return routing_result{
      .journeys_ = &s_state.results_,
      .interval_ = scanned,
      .search_stats_ =
          {.lb_time_ = ms(ms_lb),
           .execute_time_ =
               std::chrono::duration_cast<std::chrono::milliseconds>(
                   clk::now() - t0)},
      .algo_stats_ = std::move(algo_stats)};
}

}  // namespace

template <typename Criteria, typename AlgoState>
routing_result bmrap_profile_search(
    timetable const& tt,
    rt_timetable const* rtt,
    search_state& s_state,
    AlgoState& algo_state,
    basic_mcraptor_state<Criteria>& mc_state,
    query q,
    direction const search_dir,
    std::optional<std::chrono::seconds> const timeout,
    [[maybe_unused]] bool const gpu_mc) {
  // A fixed start time has no profile to restrict (and no window to scan):
  // the mcraptor Pareto set of that departure.
  if (std::holds_alternative<unixtime_t>(q.start_time_)) {
    return raptor_search(tt, rtt, s_state, mc_state, std::move(q), search_dir,
                         timeout);
  }
  // Rt as in raptor_search: the CPU raptor sizes its state by it
  auto const run = [&]<bool GpuMc>(auto& mc) {
    auto const go = [&]<direction Dir, bool Rt>() {
      return bmrap_profile<Dir, Rt, Criteria, AlgoState, GpuMc>(
          tt, rtt, s_state, algo_state, mc, std::move(q), timeout);
    };
    auto const fwd = search_dir == direction::kForward;
    return rtt == nullptr
               ? (fwd ? go.template operator()<direction::kForward, false>()
                      : go.template operator()<direction::kBackward, false>())
               : (fwd ? go.template operator()<direction::kForward, true>()
                      : go.template operator()<direction::kBackward, true>());
  };
#if defined(NIGIRI_CUDA)
  // the device mc state is a large allocation that can fail: the CPU mc phases
  // then
  if constexpr (std::is_same_v<AlgoState, gpu::gpu_raptor_state>) {
    if (auto* const mc = gpu_mc ? algo_state.try_mc_state() : nullptr;
        mc != nullptr) {
      return run.template operator()<true>(*mc);
    }
  }
#endif
  return run.template operator()<false>(mc_state);
}

#define NIGIRI_BMRAPP_INSTANTIATE(C, S)                         \
  template routing_result bmrap_profile_search<C>(              \
      timetable const&, rt_timetable const*, search_state&, S&, \
      basic_mcraptor_state<C>&, query, direction,               \
      std::optional<std::chrono::seconds>, bool);

NIGIRI_BMRAPP_INSTANTIATE(arr_criteria, raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_non_transit_criteria, raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_avoided_modes_criteria, raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_mode_switches_criteria, raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_non_transit_avoided_modes_criteria, raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_non_transit_mode_switches_criteria, raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_avoided_modes_mode_switches_criteria,
                          raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_non_transit_avoided_modes_mode_switches_criteria,
                          raptor_state)
// the night transfer criterion: CPU only
NIGIRI_BMRAPP_INSTANTIATE(arr_night_transfer_criteria, raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_non_transit_night_transfer_criteria, raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_avoided_modes_night_transfer_criteria,
                          raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_mode_switches_night_transfer_criteria,
                          raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_non_transit_avoided_modes_night_transfer_criteria,
                          raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_non_transit_mode_switches_night_transfer_criteria,
                          raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(
    arr_avoided_modes_mode_switches_night_transfer_criteria, raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(
    arr_non_transit_avoided_modes_mode_switches_night_transfer_criteria,
    raptor_state)

#if defined(NIGIRI_CUDA)
NIGIRI_BMRAPP_INSTANTIATE(arr_criteria, gpu::gpu_raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_non_transit_criteria, gpu::gpu_raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_avoided_modes_criteria, gpu::gpu_raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_mode_switches_criteria, gpu::gpu_raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_non_transit_avoided_modes_criteria,
                          gpu::gpu_raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_non_transit_mode_switches_criteria,
                          gpu::gpu_raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_avoided_modes_mode_switches_criteria,
                          gpu::gpu_raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_non_transit_avoided_modes_mode_switches_criteria,
                          gpu::gpu_raptor_state)
#endif

#undef NIGIRI_BMRAPP_INSTANTIATE

}  // namespace nigiri::routing
