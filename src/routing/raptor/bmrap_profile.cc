#include "nigiri/routing/raptor/bmraptor.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <vector>

#include "utl/erase_if.h"
#include "utl/helpers/algorithm.h"
#include "utl/verify.h"

#include "nigiri/routing/dijkstra.h"
#include "nigiri/routing/direct.h"
#include "nigiri/routing/get_fastest_direct.h"
#include "nigiri/routing/raptor/bmrap_common.h"
#include "nigiri/routing/raptor/mcraptor.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/special_stations.h"

namespace nigiri::routing {

namespace {

using namespace bmrap_detail;

// PROFILE BM-RAPTOR.
//
// The range driver in bmraptor.cc computes ONE bound matrix for the whole
// departure window, which is a deviation from the paper: BM-RAPTOR is
// defined for a single departure time tau_dep, and its backward bound
// tau_dep^<-(v,i) bottoms out AT tau_dep. Spread over a window, the bound
// bottoms out at the window start instead, so near the origin it is the
// window - not the arrival slack - that binds, and the slack stops paying
// for itself (measured: tightening it left the main search bit-identical).
//
// This driver instead runs a COMPLETE single-departure BM-RAPTOR per step
// of a PONG-style scan, so every departure gets bounds anchored at itself:
//
//   1. ping           2-criteria earliest-arrival from the step's departure
//   2. pong           backward, re-anchors each anchor to its LATEST
//                     departure (the paper's anchor set, made tight)
//   3. slacked pong   backward pruning search from the slack-relaxed anchor
//                     times -> tau_dep^<-(v,i), horizon = this departure
//   4. mc ping        the bounded multicriteria search from this departure
//   5. mc pong        backward, re-anchors each multicriteria journey to
//                     its latest departure
//
// Step 5 has no counterpart in the range driver and fixes a defect of it:
// search.h does not call set_tight_start(), so a range McRAPTOR reports a
// journey at whichever enumerated start event it was found from, which can
// be earlier than the latest departure that actually achieves it. Those
// slack-departure duplicates then survive the result pareto set whenever
// their walking differs. Re-anchoring makes every reported departure tight,
// exactly as PONG does for the two-criteria case.
template <direction SearchDir, bool Rt, typename Criteria>
routing_result bmrap_profile(timetable const& tt,
                             rt_timetable const* rtt,
                             search_state& s_state,
                             basic_mcraptor_state<Criteria>& r_state,
                             query q,
                             std::optional<std::chrono::seconds> const timeout) {
  constexpr auto const kFwd = (SearchDir == direction::kForward);
  using ping_t = raptor<SearchDir, Rt, kVias, search_mode::kOneToOne>;
  using pong_t = raptor<flip(SearchDir), Rt, kVias, search_mode::kOneToOne>;
  using mc_ping_t = basic_mcraptor<SearchDir, Criteria, /*RangeReuse=*/false>;
  using mc_pong_t =
      basic_mcraptor<flip(SearchDir), Criteria, /*RangeReuse=*/false>;

  q.sanitize(tt);
  utl::verify(mcraptor_supported(q, rtt),
              "bmrap_profile: query not supported by the mcraptor engine");

  auto const t0 = std::chrono::steady_clock::now();
  s_state.results_.clear();

  auto const is_better = [](auto const a, auto const b) {
    return kFwd ? a < b : a > b;
  };
  auto const fastest_direct = get_fastest_direct(tt, q, SearchDir);
  auto const search_interval = std::visit(
      utl::overloaded{[](interval<unixtime_t> const i) { return i; },
                      [](unixtime_t const t) {
                        return interval<unixtime_t>{t, t};
                      }},
      q.start_time_);
  auto const base_day =
      day_idx_t{std::chrono::duration_cast<date::days>(
                    std::chrono::round<std::chrono::days>(
                        search_interval.from_ +
                        ((search_interval.to_ - search_interval.from_) / 2)) -
                    tt.internal_interval().from_)
                    .count()};

  // ---- destinations + lower bounds, both directions ----
  auto fwd_is_dest = bitvec{};
  auto fwd_dist = std::vector<std::uint16_t>{};
  collect_destinations(tt, q.destination_, q.dest_match_mode_, fwd_is_dest,
                       fwd_dist);
  auto qf = q;
  qf.flip_dir();
  auto bwd_is_dest = bitvec{};
  auto bwd_dist = std::vector<std::uint16_t>{};
  collect_destinations(tt, qf.destination_, qf.dest_match_mode_, bwd_is_dest,
                       bwd_dist);

  auto fwd_lb = std::vector<std::uint16_t>{};
  auto bwd_lb = std::vector<std::uint16_t>{};
  auto const lb_t0 = std::chrono::steady_clock::now();
  dijkstra(tt, q,
           (kFwd ? tt.fwd_search_lb_graph_[q.prf_idx_]
                 : tt.bwd_search_lb_graph_[q.prf_idx_]),
           nullptr, nullptr, fwd_lb);
  dijkstra(tt, qf,
           (kFwd ? tt.bwd_search_lb_graph_[q.prf_idx_]
                 : tt.fwd_search_lb_graph_[q.prf_idx_]),
           nullptr, nullptr, bwd_lb);
  auto const lb_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - lb_t0)
                         .count();

  auto is_via = std::array<bitvec, kMaxVias>{};
  auto no_via = std::vector<via_stop>{};
  auto const mk = [&](auto& algo_state, bitvec& is_dest,
                      std::vector<std::uint16_t>& dist,
                      std::vector<std::uint16_t>& lb, auto const& qq) {
    return std::tuple{std::ref(algo_state), std::ref(is_dest), std::ref(dist),
                      std::ref(lb), std::cref(qq)};
  };
  static_cast<void>(mk);

  // one raptor_state serves the 2-criteria ping, the 2-criteria pong and
  // the slacked pong: within a step they run strictly one after another
  // and each resets what it needs
  auto r2_state = raptor_state{};
  auto mc_pong_state = basic_mcraptor_state<Criteria>{};

  auto ping = ping_t{tt,       rtt,      r2_state,  fwd_is_dest,
                     is_via,   fwd_dist, q.td_dest_, fwd_lb,
                     no_via,   base_day, q.allowed_claszes_,
                     q.require_bike_transport_, q.require_car_transport_,
                     q.prf_idx_ == 2U, q.transfer_time_settings_};
  auto pong = pong_t{tt,       rtt,      r2_state,  bwd_is_dest,
                     is_via,   bwd_dist, qf.td_dest_, bwd_lb,
                     no_via,   base_day, q.allowed_claszes_,
                     q.require_bike_transport_, q.require_car_transport_,
                     q.prf_idx_ == 2U, q.transfer_time_settings_};
  auto mc_ping = mc_ping_t{tt,       rtt,      r_state,   fwd_is_dest,
                           is_via,   fwd_dist, q.td_dest_, fwd_lb,
                           no_via,   base_day, q.allowed_claszes_,
                           q.require_bike_transport_, q.require_car_transport_,
                           q.prf_idx_ == 2U, q.transfer_time_settings_};
  auto mc_pong = mc_pong_t{tt,       rtt,      mc_pong_state, bwd_is_dest,
                           is_via,   bwd_dist, qf.td_dest_,   bwd_lb,
                           no_via,   base_day, q.allowed_claszes_,
                           q.require_bike_transport_, q.require_car_transport_,
                           q.prf_idx_ == 2U, q.transfer_time_settings_};

  auto stats = raptor_stats{};
  auto prune_stats = raptor_stats{};
  auto all_anchors = std::vector<anchor>{};
  auto n_steps = std::uint64_t{0U};
  auto n_bound_builds = std::uint64_t{0U};
  auto ms_ping = std::chrono::steady_clock::duration{};
  auto ms_pong = std::chrono::steady_clock::duration{};
  auto ms_bounds = std::chrono::steady_clock::duration{};
  auto ms_mc = std::chrono::steady_clock::duration{};
  auto ms_mc_pong = std::chrono::steady_clock::duration{};
  auto ms_recon = std::chrono::steady_clock::duration{};
  auto ms_fwd_bounds = std::chrono::steady_clock::duration{};
  auto n_fwd_bound_builds = std::uint64_t{0U};
  auto fwd_bounds = bmrap_bounds{};

  auto start_time =
      kFwd ? search_interval.from_ : search_interval.to_ - duration_t{1};
  auto const end_time =
      kFwd ? search_interval.to_ : search_interval.from_ - duration_t{1};
  auto const is_timeout = [&]() {
    return timeout && (std::chrono::steady_clock::now() - t0) >= *timeout;
  };
  auto const budget_cap = static_cast<std::uint8_t>(
      std::min<unsigned>(q.max_transfers_ + 1U, kMaxTransfers + 1U));

  auto starts = std::vector<start>{};
  auto bounds = bmrap_bounds{};

  // The anchor set - and everything derived from it - is only invalidated
  // when start_time passes the EARLIEST anchor departure. Up to that point
  // every anchor is still available, and no new one can appear: a
  // two-criteria journey only becomes Pareto-optimal once the journey
  // dominating it drops out, which cannot happen before that same
  // breakpoint. So the steps that advance on a MULTICRITERIA breakpoint
  // (which are the majority - the mc profile has more breakpoints than the
  // bicriteria one) re-derive an identical anchor set, and the ping, pong,
  // tau_dep^<- and tau_arr^-> work can all be skipped.
  //
  // Reusing the two matrices is safe in the same direction as everything
  // else here: both were built from an EARLIER departure, which makes them
  // looser, never tighter - tau_arr^->(v,i) can only be earlier, and
  // compute_bounds()' horizon can only be further back. A looser bound
  // under-prunes, which the final restriction filter then cleans up.
  // Forward realizations, kept in the FORWARD convention and spliced in
  // after the results are swapped over (see below).
  auto realized = std::vector<journey>{};
  // A forward search from departure d returns the whole Pareto set at d, so
  // one realization per departure suffices for the entire scan. Without
  // this the pass reruns on every step that rediscovers the same journey.
  auto realized_deps = std::vector<unixtime_t>{};
  auto ms_realize = std::chrono::steady_clock::duration{};
  auto n_realized = std::uint64_t{0U};
  auto const realize_fwd = std::getenv("NIGIRI_BMRAPP_NO_REALIZE") == nullptr;

  auto anchors = std::vector<anchor>{};
  auto budget = std::uint8_t{0U};
  auto anchors_valid_until = std::optional<unixtime_t>{};
  auto n_anchor_recomputes = std::uint64_t{0U};
  auto exit_reason = std::uint64_t{0U};  // 0=cond 1=ping 2=anchors 3=stall

  // same interval extension PONG uses: keep stepping past the nominal
  // window until enough connections have been validated. Without this the
  // driver would only ever scan the raw searchWindow (15 min by default),
  // while every other engine grows it to satisfy numItineraries.
  //
  // What gets counted is selectable via NIGIRI_BMRAPP_COUNT:
  //   "mc"      (default) the multicriteria results, i.e. what this engine
  //             actually returns. They accumulate faster than two-criteria
  //             journeys, so the scan stops sooner than the range driver's.
  //   "anchors" the anchor set, which is exactly what PONG would report.
  //             Reproduces PONG's own stopping point, so both drivers end
  //             up scanning near-identical windows - useful for comparison.
  // meet-in-the-middle bounds for the BACKWARD multicriteria search (mc
  // pong). Costs one extra one-to-all forward RAPTOR per step, so it is
  // opt-in until measured.
  // 0=off, 1=mc pong only, 2=stage 2 (slacked pong) only, 3=both
  auto const fwd_bounds_mode = [] {
    auto const* const e = std::getenv("NIGIRI_BMRAPP_FWD_BOUNDS");
    if (e == nullptr) {
      return 0;
    }
    auto const v = std::string_view{e};
    return v == "mcpong" ? 1 : v == "stage2" ? 2 : 3;
  }();
  auto const fwd_bounds_on = fwd_bounds_mode != 0;

  auto const count_anchors = [] {
    auto const* const e = std::getenv("NIGIRI_BMRAPP_COUNT");
    return e != nullptr && std::string_view{e} == "anchors";
  }();

  auto const anchor_travel = [](anchor const& a) {
    return duration_t{static_cast<duration_t::rep>(
        std::abs((a.found_ - a.anchored_).count()))};
  };
  auto const n_anchors = [&](bool const include_too_slow) {
    return utl::count_if(all_anchors, [&](anchor const& a) {
      if (!is_better(a.anchored_, start_time)) {
        return false;  // not scanned past yet
      }
      if (!include_too_slow && !(anchor_travel(a) < fastest_direct &&
                                 anchor_travel(a) < q.max_travel_time_)) {
        return false;
      }
      // duplicates would dominate each other and cancel out, hence the
      // dedup on insert into all_anchors below
      return !utl::any_of(all_anchors, [&](anchor const& o) {
        return &o != &a && o.trips_ <= a.trips_ &&
               !is_better(o.anchored_, a.anchored_) &&
               !is_better(a.found_, o.found_);
      });
    });
  };

  // The restriction, applied exactly as the final filter does. Counting
  // unrestricted journeys stops the scan on results that are about to be
  // thrown away, so the caller ends up with fewer than numItineraries: on
  // one query BMRAPP returned 6 journeys where 23 exist, because the
  // unrestricted mc pong padded the count on the very first step.
  //
  // It is applied HERE and not at insertion time: A(J) is only final once
  // the scan has passed J's departure. Anchors found at later steps have
  // later-or-equal arrivals and so can never improve A(J), but the anchor
  // that decides a freshly inserted journey may not be discovered yet -
  // dropping it on insertion could discard a journey whose verdict later
  // flips to "keep". Every journey counted below is already is_validated
  // (departure behind start_time), where the verdict cannot change.
  auto const restricted_away = [&](journey const& j) {
    if (get_slack().no_restrict_) {
      return false;
    }
    // NB: inside the scan the journeys still carry mc pong's backward
    // convention - dest_time_ is the DEPARTURE, start_time_ the ARRIVAL.
    // They are only swapped after the loop, which is why the field names
    // here are the mirror of the ones in the final filter.
    auto const dep = j.dest_time_;
    auto const arr = j.start_time_;
    if (utl::any_of(all_anchors, [&](anchor const& a) {
          return a.trips_ == j.transfers_ + 1U && a.anchored_ == dep &&
                 a.found_ == arr;
        })) {
      return false;  // an anchor is its own A(J)
    }
    auto const trips = static_cast<unsigned>(j.transfers_) + 1U;
    auto const* const a = anchor_of<SearchDir>(all_anchors, dep, trips);
    return a == nullptr ||
           trips > trip_budget(a->trips_, std::uint8_t{kMaxTransfers + 1U}) ||
           misses_deadline<SearchDir>(*a, arr);
  };

  auto const n_results = [&](bool const include_too_slow) {
    return utl::count_if(s_state.results_, [&](journey const& j) {
      if (!is_better(j.dest_time_, start_time)) {
        return false;  // dest_time_ is still the departure here
      }
      if (restricted_away(j)) {
        return false;
      }
      if (!include_too_slow && !(j.travel_time() < fastest_direct &&
                                 j.travel_time() < q.max_travel_time_)) {
        return false;
      }
      return !utl::any_of(s_state.results_, [&](journey const& o) {
        return &o != &j && o.tuple_dominates(j);
      });
    });
  };

  auto const n_found = [&](bool const include_too_slow) {
    return count_anchors ? n_anchors(include_too_slow)
                         : n_results(include_too_slow);
  };

  while ((is_better(start_time, end_time) ||
          n_found(true) + n_found(false) <
              2 * static_cast<int>(q.min_connection_count_)) &&
         tt.external_interval().contains(start_time) && !is_timeout()) {
    auto const anchors_stale =
        !anchors_valid_until.has_value() ||
        is_better(*anchors_valid_until, start_time);
    if (anchors_stale) {
      ++n_anchor_recomputes;

    // ---- 1. PING: two-criteria EA from this departure ----
    auto const p0 = std::chrono::steady_clock::now();
    starts.clear();
    get_starts(SearchDir, tt, rtt, start_time, q.start_, q.td_start_,
               q.via_stops_, q.max_start_offset_, q.start_match_mode_,
               q.use_start_footpaths_, starts, false, q.prf_idx_,
               q.transfer_time_settings_);
    ping.reset_arrivals();
    ping.next_start_time();
    if (fwd_bounds_on) {
      // relax target pruning by the arrival slack, so this search's round
      // times are a valid tau_arr^->(v, i) matrix (paper, Sec. 4.3)
      auto const& sc = get_slack();
      ping.set_dest_relax(start_time,
                          sc.arr_fixed_min_ >= 0.0 ? 1.0 : sc.arr_,
                          sc.arr_fixed_min_ >= 0.0
                              ? static_cast<int>(sc.arr_fixed_min_)
                              : 0);
    }
    for (auto const& s : starts) {
      ping.add_start(s.stop_, s.time_at_stop_);
    }
    auto ping_results = pareto_set<journey>{};
    ping.execute(start_time, q.max_transfers_,
                 start_time + (kFwd ? 1 : -1) *
                                  (std::min(fastest_direct, q.max_travel_time_) +
                                   duration_t{1}),
                 q.prf_idx_, ping_results);
    ms_ping += std::chrono::steady_clock::now() - p0;
    utl::sort(ping_results, [&](journey const& a, journey const& b) {
      return is_better(a.dest_time_, b.dest_time_);
    });

    if (fwd_bounds_on) {
      // r2_state still holds the ping's round times here; the pong below
      // reuses the same state, so the matrix has to be taken now - before
      // the anchors exist. Size it from the PING's trip counts: re-anchoring
      // in the pong moves departures, never the number of trips, so this is
      // the same budget the anchors will produce. Sizing to budget_cap
      // instead would make the build loop (rounds x locations) dominate
      // cheap queries.
      auto ping_trips = std::uint8_t{0U};
      for (auto const& j : ping_results) {
        ping_trips = std::max(ping_trips,
                              static_cast<std::uint8_t>(j.transfers_ + 1U));
      }
      auto const f0 = std::chrono::steady_clock::now();
      fwd_bounds = build_reach_matrix<SearchDir>(
          tt, q, r2_state, trip_budget(ping_trips, budget_cap));
      ms_fwd_bounds += std::chrono::steady_clock::now() - f0;
      ++n_fwd_bound_builds;
      if ((fwd_bounds_mode & 1) != 0) {
        mc_pong.set_bounds(&fwd_bounds);
      }
    }

    // ---- 2. PONG: re-anchor each anchor to its LATEST departure ----
    auto const g0 = std::chrono::steady_clock::now();
    auto tight = pareto_set<journey>{};
    pong.reset_arrivals();
    auto g_end = begin(ping_results);
    for (auto pi = begin(ping_results); pi != end(ping_results); ++pi) {
      if (pi != g_end) {
        continue;
      }
      auto const g_arr = pi->dest_time_;
      g_end = std::find_if(pi, end(ping_results), [&](journey const& j) {
        return j.dest_time_ != g_arr;
      });
      auto max_tr = pi->transfers_;
      auto loosest = pi->start_time_;
      for (auto it = std::next(pi); it != g_end; ++it) {
        max_tr = std::max(max_tr, it->transfers_);
        if (is_better(it->start_time_, loosest)) {
          loosest = it->start_time_;
        }
      }
      starts.clear();
      get_starts(flip(SearchDir), tt, rtt, g_arr, qf.start_, qf.td_start_,
                 qf.via_stops_, qf.max_start_offset_, qf.start_match_mode_,
                 qf.start_match_mode_ != location_match_mode::kIntermodal,
                 starts, false, q.prf_idx_, q.transfer_time_settings_);
      pong.next_start_time();
      for (auto const& s : starts) {
        pong.add_start(s.stop_, s.time_at_stop_);
      }
      pong.execute(g_arr, max_tr, loosest - duration_t{kFwd ? 1 : -1},
                   q.prf_idx_, tight);
    }
    ms_pong += std::chrono::steady_clock::now() - g0;

    // tight journeys are (start_time_ = arrival, dest_time_ = departure)
    anchors.clear();
    auto max_trips = std::uint8_t{0U};
    for (auto const& j : tight) {
      auto const trips = static_cast<std::uint8_t>(j.transfers_ + 1U);
      anchors.push_back({j.dest_time_, j.start_time_, trips});
      max_trips = std::max(max_trips, trips);
    }
    if (anchors.empty()) {
      exit_reason = 2U;
      break;
    }
    // Deduplicate: consecutive steps re-discover the same anchor journeys,
    // and two anchors with an identical (dep, arr, trips) tuple dominate one
    // another - so leaving duplicates in would cancel them both out of
    // n_anchors() below and the scan would never reach its stopping point.
    for (auto const& a : anchors) {
      if (!utl::any_of(all_anchors, [&](anchor const& o) {
            return o.anchored_ == a.anchored_ && o.found_ == a.found_ &&
                   o.trips_ == a.trips_;
          })) {
        all_anchors.emplace_back(a);
      }
    }
    budget = trip_budget(max_trips, budget_cap);
    anchors_valid_until = utl::min_element(
        anchors, [&](anchor const& a, anchor const& b) {
          return is_better(a.anchored_, b.anchored_);
        })->anchored_;

    // ---- 3. SLACKED PONG: bounds anchored at THIS departure ----
    if (!get_slack().no_bounds_) {
      auto const b0 = std::chrono::steady_clock::now();
      bounds = compute_bounds<SearchDir, Rt>(
          tt, rtt, q, anchors, base_day,
          /*horizon=*/start_time - duration_t{kFwd ? 1 : -1}, budget, r2_state,
          prune_stats,
          (fwd_bounds_mode & 2) != 0 ? &fwd_bounds : nullptr);
      ms_bounds += std::chrono::steady_clock::now() - b0;
      ++n_bound_builds;
      mc_ping.set_bounds(&bounds);
    }

    }  // anchors_stale
    ++n_steps;

    // ---- 4. MC PING + 5. MC PONG ----
    auto const m0 = std::chrono::steady_clock::now();
    starts.clear();
    get_starts(SearchDir, tt, rtt, start_time, q.start_, q.td_start_,
               q.via_stops_, q.max_start_offset_, q.start_match_mode_,
               q.use_start_footpaths_, starts, false, q.prf_idx_,
               q.transfer_time_settings_);
    mc_ping.reset_arrivals();
    mc_ping.next_start_time();
    for (auto const& s : starts) {
      mc_ping.add_start(s.stop_, s.time_at_stop_);
    }
    auto mc_results = pareto_set<journey>{};
    mc_ping.execute(start_time, static_cast<std::uint8_t>(budget - 1U),
                    start_time + (kFwd ? 1 : -1) *
                                     (std::min(fastest_direct,
                                               q.max_travel_time_) +
                                      duration_t{1}),
                    q.prf_idx_, mc_results);

    ms_mc += std::chrono::steady_clock::now() - m0;

    // ---- 5. MC PONG: re-anchor each mc journey to its latest departure ----
    auto const mp0 = std::chrono::steady_clock::now();
    utl::sort(mc_results, [&](journey const& a, journey const& b) {
      return is_better(a.dest_time_, b.dest_time_);
    });

    mc_pong.reset_arrivals();
    auto step_results = pareto_set<journey>{};
    auto m_end = begin(mc_results);
    for (auto mi = begin(mc_results); mi != end(mc_results); ++mi) {
      if (mi != m_end) {
        continue;
      }
      auto const g_arr = mi->dest_time_;
      m_end = std::find_if(mi, end(mc_results), [&](journey const& j) {
        return j.dest_time_ != g_arr;
      });
      auto max_tr = mi->transfers_;
      for (auto it = std::next(mi); it != m_end; ++it) {
        max_tr = std::max(max_tr, it->transfers_);
      }
      starts.clear();
      get_starts(flip(SearchDir), tt, rtt, g_arr, qf.start_, qf.td_start_,
                 qf.via_stops_, qf.max_start_offset_, qf.start_match_mode_,
                 qf.start_match_mode_ != location_match_mode::kIntermodal,
                 starts, false, q.prf_idx_, q.transfer_time_settings_);
      mc_pong.next_start_time();
      for (auto const& s : starts) {
        mc_pong.add_start(s.stop_, s.time_at_stop_);
      }
      // may not depart before the step we are scanning
      mc_pong.execute(g_arr, max_tr,
                      start_time - duration_t{kFwd ? 1 : -1}, q.prf_idx_,
                      step_results);
    }
    ms_mc_pong += std::chrono::steady_clock::now() - mp0;

    auto const rc0 = std::chrono::steady_clock::now();
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
    for (auto const& j : step_results) {
      if (!j.error_) {
        s_state.results_.add(journey{j});
      }
    }
    ms_recon += std::chrono::steady_clock::now() - rc0;

    // ---- 5b. RE-REALIZE FORWARD ----
    // mc pong reconstructs backwards, so every intermediate leg is the
    // LATEST run that still makes the connection - the traveller gets no
    // slack at any transfer and one delay loses the chain. Re-running
    // forward from the departure mc pong just pinned produces the same
    // tuple with the EARLIEST connections instead. Done here, inside the
    // step, because tau_dep^<- is live: destination pruning alone (even on
    // the exact arrival) leaves too much of the network unpruned.
    if (realize_fwd) {
      auto const rz0 = std::chrono::steady_clock::now();
      auto deps = std::vector<unixtime_t>{};
      for (auto const& j : step_results) {
        if (!j.error_ && j.is_reconstructed_ &&
            utl::find(deps, j.dest_time_) == end(deps) &&
            utl::find(realized_deps, j.dest_time_) == end(realized_deps)) {
          deps.emplace_back(j.dest_time_);  // dest_time_ is the departure
        }
      }
      for (auto const d : deps) {
        auto max_tr = std::uint8_t{0U};
        auto loosest_arr = std::optional<unixtime_t>{};
        for (auto const& j : step_results) {
          if (j.error_ || j.dest_time_ != d) {
            continue;
          }
          max_tr = std::max(max_tr, j.transfers_);
          if (!loosest_arr.has_value() ||
              is_better(*loosest_arr, j.start_time_)) {
            loosest_arr = j.start_time_;  // start_time_ is the arrival
          }
        }
        starts.clear();
        get_starts(SearchDir, tt, rtt, d, q.start_, q.td_start_, q.via_stops_,
                   q.max_start_offset_, q.start_match_mode_,
                   q.use_start_footpaths_, starts, false, q.prf_idx_,
                   q.transfer_time_settings_);
        mc_ping.reset_arrivals();
        mc_ping.next_start_time();
        for (auto const& st : starts) {
          mc_ping.add_start(st.stop_, st.time_at_stop_);
        }
        auto fwd = pareto_set<journey>{};
        mc_ping.execute(d, max_tr,
                        *loosest_arr + duration_t{kFwd ? 1 : -1}, q.prf_idx_,
                        fwd);
        for (auto& f : fwd) {
          if (f.is_reconstructed_ || f.error_) {
            continue;
          }
          try {
            mc_ping.reconstruct(q, f);
          } catch (std::exception const& e) {
            f.error_ = true;
            log(log_lvl::error, "bmrap_profile", "realize failed: {}",
                e.what());
          }
        }
        realized_deps.emplace_back(d);
        for (auto const& f : fwd) {
          if (!f.error_ && f.is_reconstructed_) {
            realized.emplace_back(f);
          }
        }
      }
      ms_realize += std::chrono::steady_clock::now() - rz0;
    }

    // ---- 6. advance ----
    // Advancing on the two-criteria anchors alone is not enough: between
    // two anchor departures the MULTICRITERIA pareto set can still change
    // (a later departure with different walking becomes optimal), and
    // stepping straight past those departures drops those journeys. So the
    // step advances to the loosest departure over the anchors AND this
    // step's own multicriteria journeys.
    // Only departures at or after the current step count: the pong runs
    // with worst_time_at_dest = loosest - 1min, so a validated departure
    // can land exactly on that boundary, one minute BEHIND the step. Those
    // belong to an already-scanned departure, and letting one set the
    // advance would make next == start_time and stall the scan.
    auto loosest_dep = std::optional<unixtime_t>{};
    auto const consider = [&](unixtime_t const d) {
      if (is_better(d, start_time)) {
        return;
      }
      if (!loosest_dep.has_value() || is_better(d, *loosest_dep)) {
        loosest_dep = d;
      }
    };
    for (auto const& a : anchors) {
      consider(a.anchored_);
    }
    // diagnostic: advance like classical PONG (anchors only). Loses
    // multicriteria journeys - it is here to attribute cost, not to use.
    if (std::getenv("NIGIRI_BMRAPP_ADVANCE_ANCHORS") == nullptr) {
      for (auto const& j : step_results) {
        consider(j.dest_time_);  // dest_time_ is the departure here
      }
    }
    if (std::getenv("NIGIRI_BMRAPP_TRACE") != nullptr) {
      std::fprintf(stderr, "STEP start=%lld anchors=[", 
                   static_cast<long long>(start_time.time_since_epoch().count()));
      for (auto const& a : anchors) {
        std::fprintf(stderr, "%lld ",
                     static_cast<long long>(a.anchored_.time_since_epoch().count()));
      }
      std::fprintf(stderr, "] step_results=[");
      for (auto const& j : step_results) {
        std::fprintf(stderr, "%lld ",
                     static_cast<long long>(j.dest_time_.time_since_epoch().count()));
      }
      std::fprintf(stderr, "] loosest=%lld\n",
                   static_cast<long long>(
                       loosest_dep.value_or(start_time).time_since_epoch().count()));
    }
    auto const next = loosest_dep.value_or(start_time) +
                      duration_t{kFwd ? 1 : -1};
    if (!is_better(start_time, next)) {
      exit_reason = 3U;
      break;  // no progress - stop rather than spin
    }
    start_time = next;
  }

  stats = ping.get_stats() + pong.get_stats() + mc_ping.get_stats() +
          mc_pong.get_stats();

  // ---- results: still (arrival, departure); make them journeys ----
  auto const scanned =
      kFwd ? interval<unixtime_t>{search_interval.from_, start_time}
           : interval<unixtime_t>{start_time + duration_t{1},
                                  search_interval.to_};
  utl::erase_if(s_state.results_, [&](journey const& j) {
    return !j.is_reconstructed_ || j.error_ ||
           !is_better(j.dest_time_, start_time) ||
           j.travel_time() >= fastest_direct ||
           j.travel_time() > q.max_travel_time_;
  });
  for (auto& x : s_state.results_) {
    std::swap(x.start_time_, x.dest_time_);
  }

  // Splice in the forward realizations: same journey (identical times,
  // transfers and criteria), better legs. Only legs_ moves - the tuple must
  // come out byte-identical, which is what makes this verifiable.
  if (realize_fwd) {
    for (auto& x : s_state.results_) {
      auto const it = utl::find_if(realized, [&](journey const& f) {
        return f.start_time_ == x.start_time_ && f.dest_time_ == x.dest_time_ &&
               f.transfers_ == x.transfers_ &&
               f.criteria_cost_ == x.criteria_cost_ &&
               f.criteria_air_ == x.criteria_air_ &&
               f.criteria_clasz_ == x.criteria_clasz_;
      });
      if (it != end(realized)) {
        x.legs_ = it->legs_;
        ++n_realized;
      }
    }
  }
  for (auto& j : s_state.results_) {
    auto const swap_st = [](location_idx_t const l) -> location_idx_t {
      switch (to_idx(l)) {
        case to_idx(get_special_station(special_station::kStart)):
          return get_special_station(special_station::kEnd);
        case to_idx(get_special_station(special_station::kEnd)):
          return get_special_station(special_station::kStart);
        default: return l;
      }
    };
    if (!j.legs_.empty()) {
      j.legs_.front().from_ = swap_st(j.legs_.front().from_);
      j.legs_.back().to_ = swap_st(j.legs_.back().to_);
    }
  }

  // ---- restrict to J_R, same definition as the range driver ----
  auto n_restricted = std::uint64_t{0U};
  if (!get_slack().no_restrict_) {
    auto const is_anchor = [&](journey const& j) {
      return utl::any_of(all_anchors, [&](anchor const& a) {
        return a.trips_ == j.transfers_ + 1U && a.anchored_ == j.start_time_ &&
               a.found_ == j.dest_time_;
      });
    };
    utl::erase_if(s_state.results_, [&](journey const& j) {
      if (is_anchor(j)) {
        return false;
      }
      auto const trips = static_cast<unsigned>(j.transfers_) + 1U;
      auto const* a = anchor_of<SearchDir>(all_anchors, j.start_time_, trips);
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
  }

  enrich_with_slow_direct<SearchDir>(tt, rtt, q, scanned, s_state.results_);
  utl::sort(s_state.results_, [](journey const& a, journey const& b) {
    return std::tuple{a.start_time_, a.transfers_, a.dest_time_,
                      a.criteria_cost_} <
           std::tuple{b.start_time_, b.transfers_, b.dest_time_,
                      b.criteria_cost_};
  });

  auto const ms = [](auto const d) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(d).count());
  };
  auto algo_stats = stats.to_map();
  for (auto const& [k, v] : prune_stats.to_map()) {
    algo_stats["prune_" + k] = v;
  }
  algo_stats["bmrapp_steps"] = n_steps;
  algo_stats["bmrapp_anchor_recomputes"] = n_anchor_recomputes;
  algo_stats["bmrapp_ms_realize"] = ms(ms_realize);
  algo_stats["bmrapp_realized"] = n_realized;
  algo_stats["bmrapp_exit"] = exit_reason;
  algo_stats["bmrapp_bound_builds"] = n_bound_builds;
  algo_stats["bmrapp_anchors"] = all_anchors.size();
  algo_stats["bmrapp_restricted_away"] = n_restricted;
  algo_stats["bmrapp_ms_ping"] = ms(ms_ping);
  algo_stats["bmrapp_ms_pong"] = ms(ms_pong);
  algo_stats["bmrapp_ms_bounds"] = ms(ms_bounds);
  algo_stats["bmrapp_ms_mc_ping"] = ms(ms_mc);
  algo_stats["bmrapp_ms_mc_pong"] = ms(ms_mc_pong);
  algo_stats["bmrapp_ms_reconstruct"] = ms(ms_recon);
  algo_stats["bmrapp_ms_fwd_bounds"] = ms(ms_fwd_bounds);
  algo_stats["bmrapp_fwd_bound_builds"] = n_fwd_bound_builds;
  algo_stats["bmrapp_ms_lb"] = static_cast<std::uint64_t>(lb_ms);
  algo_stats["bmrapp_count_anchors"] = count_anchors ? 1U : 0U;

  return routing_result{
      .journeys_ = &s_state.results_,
      .interval_ = scanned,
      .search_stats_ =
          {.lb_time_ = static_cast<std::uint64_t>(lb_ms),
           .execute_time_ = std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - t0)},
      .algo_stats_ = std::move(algo_stats)};
}

}  // namespace

template <typename AlgoState>
routing_result bmrap_profile_search(
    timetable const& tt,
    rt_timetable const* rtt,
    search_state& s_state,
    AlgoState& algo_state,
    query q,
    direction const search_dir,
    std::optional<std::chrono::seconds> const timeout) {
  if (search_dir == direction::kForward) {
    return bmrap_profile<direction::kForward, false>(
        tt, rtt, s_state, algo_state, std::move(q), timeout);
  } else {
    return bmrap_profile<direction::kBackward, false>(
        tt, rtt, s_state, algo_state, std::move(q), timeout);
  }
}

template routing_result bmrap_profile_search(
    timetable const&, rt_timetable const*, search_state&, mcraptor_state&, query,
    direction, std::optional<std::chrono::seconds>);

template routing_result bmrap_profile_search(
    timetable const&, rt_timetable const*, search_state&, mcraptor_cost_state&, query,
    direction, std::optional<std::chrono::seconds>);

template routing_result bmrap_profile_search(
    timetable const&, rt_timetable const*, search_state&, mcraptor_walk_state&, query,
    direction, std::optional<std::chrono::seconds>);

template routing_result bmrap_profile_search(
    timetable const&, rt_timetable const*, search_state&, mcraptor_air_state&, query,
    direction, std::optional<std::chrono::seconds>);

template routing_result bmrap_profile_search(
    timetable const&, rt_timetable const*, search_state&, mcraptor_walk_air_state&, query,
    direction, std::optional<std::chrono::seconds>);

template routing_result bmrap_profile_search(
    timetable const&, rt_timetable const*, search_state&, mcraptor_clasz_state&, query,
    direction, std::optional<std::chrono::seconds>);

template routing_result bmrap_profile_search(
    timetable const&, rt_timetable const*, search_state&, mcraptor_walk_clasz_state&, query,
    direction, std::optional<std::chrono::seconds>);

}  // namespace nigiri::routing
