#include "nigiri/routing/raptor/bmraptor.h"

#include <algorithm>
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
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/special_stations.h"

namespace nigiri::routing {

namespace {

using namespace bmrap_detail;

// PROFILE BM-RAPTOR: a complete single-departure BM-RAPTOR per step of a
// PONG-style scan, so every departure gets bounds anchored at itself (the
// paper's tau_dep^<-(v,i) bottoms out AT tau_dep; a range variant spreading
// one matrix over a window bottoms out at the window start, where near the
// origin the window and not the arrival slack binds).
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
// Step 5 exists because search.h never calls set_tight_start(): a range
// McRAPTOR reports a journey at whichever enumerated start event found it,
// and those slack-departure duplicates survive the result pareto set when
// their walking differs. Re-anchoring makes every reported departure tight.
template <direction SearchDir,
          bool Rt,
          typename Criteria,
          typename AlgoState,
          int GpuMc>
routing_result bmrap_profile(timetable const& tt,
                             rt_timetable const* rtt,
                             search_state& s_state,
                             AlgoState& r_state,
                             query q,
                             std::optional<std::chrono::seconds> const timeout) {
  constexpr auto const kFwd = (SearchDir == direction::kForward);
  using ping_t = typename bmrap_algo_for<SearchDir, Rt, AlgoState>::type;
  using pong_t =
      typename bmrap_algo_for<flip(SearchDir), Rt, AlgoState>::type;
  using mc_ping_for = bmrap_mc_algo_for<SearchDir, Criteria, (GpuMc >= 1)>;
  using mc_pong_for =
      bmrap_mc_algo_for<flip(SearchDir), Criteria, (GpuMc >= 2)>;
  using mc_ping_t = typename mc_ping_for::type;
  using mc_pong_t = typename mc_pong_for::type;

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

  // One state serves the ping, the pong and the slacked pong, as pong.cc does:
  // within a step they run strictly in sequence and each resets what it needs,
  // and the GPU's per-query buffers are direction-indexed so both directions
  // coexist. The multicriteria phases use their own CPU state unless the two
  // device-implemented shapes opt in (see GpuMc); on the device they share one
  // state that outlives the query, the allocation being too large to repeat.
  auto cpu_mc_ping_state = std::conditional_t<
      (GpuMc >= 1), std::monostate, basic_mcraptor_state<Criteria>>{};
  auto cpu_mc_pong_state = std::conditional_t<
      (GpuMc >= 2), std::monostate, basic_mcraptor_state<Criteria>>{};
  auto& mc_ping_state = [&]() -> typename mc_ping_for::state& {
    if constexpr (GpuMc >= 1) {
      return r_state.mc_state();
    } else {
      return cpu_mc_ping_state;
    }
  }();
  auto& mc_pong_state = [&]() -> typename mc_pong_for::state& {
    if constexpr (GpuMc >= 2) {
      return r_state.mc_state();
    } else {
      return cpu_mc_pong_state;
    }
  }();

  // ping_t/pong_t are always plain raptor or gpu_raptor (bmrap_algo_for has
  // no mcraptor specialization), so the ctor always takes the newer
  // no_compulsory_reservation/prf_idx trailing args.
  auto ping = ping_t{tt,       rtt,      r_state,  fwd_is_dest,
                     is_via,   fwd_dist, q.td_dest_, fwd_lb,
                     no_via,   base_day, q.allowed_claszes_,
                     q.require_bike_transport_, q.require_car_transport_,
                     q.prf_idx_ == 2U, q.no_compulsory_reservation_,
                     q.transfer_time_settings_, q.prf_idx_};
  auto pong = pong_t{tt,       rtt,      r_state,  bwd_is_dest,
                     is_via,   bwd_dist, qf.td_dest_, bwd_lb,
                     no_via,   base_day, q.allowed_claszes_,
                     q.require_bike_transport_, q.require_car_transport_,
                     q.prf_idx_ == 2U, q.no_compulsory_reservation_,
                     q.transfer_time_settings_, q.prf_idx_};
  auto mc_ping = mc_ping_t{tt,       rtt,      mc_ping_state, fwd_is_dest,
                           is_via,   fwd_dist, q.td_dest_, fwd_lb,
                           no_via,   base_day, q.allowed_claszes_,
                           q.require_bike_transport_, q.require_car_transport_,
                           q.prf_idx_ == 2U, q.transfer_time_settings_};
  auto mc_pong = mc_pong_t{tt,       rtt,      mc_pong_state, bwd_is_dest,
                           is_via,   bwd_dist, qf.td_dest_,   bwd_lb,
                           no_via,   base_day, q.allowed_claszes_,
                           q.require_bike_transport_, q.require_car_transport_,
                           q.prf_idx_ == 2U, q.transfer_time_settings_};

  if constexpr (GpuMc >= 2) {
    // Phase 5 runs a sequence of departures under ONE reset_arrivals(), and
    // the device engine carries its reuse frontier across them while the CPU
    // engine at RangeReuse=false has no cross-departure reuse at all. Without
    // this the frontier rejects labels no reported journey dominates (it drops
    // a later-departing, higher-transfer variant with an equal arrival).
    // next_start_time() already clears the bags, so within one departure the
    // two engines agree.
    mc_pong.set_reuse_same_dep();
  }

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

  auto const is_timeout = [&]() {
    return timeout && (std::chrono::steady_clock::now() - t0) >= *timeout;
  };
  // upper bound for a search departing at t: nothing slower than the fastest
  // direct connection is of interest
  auto const worst_at_dest = [&](unixtime_t const t) {
    return t + (kFwd ? 1 : -1) *
                   (std::min(fastest_direct, q.max_travel_time_) +
                    duration_t{1});
  };
  // Relaxes the ping's target pruning by the arrival slack, which is what
  // makes its round times a valid tau_arr^->(v, i) matrix (paper, Sec. 4.3).
  // The clamps are relax_arr()'s, so the relaxation matches the restriction
  // it is meant to bound: an unclamped ratio relaxes target pruning by hours
  // on a long-haul query, which made the ping the single most expensive
  // phase (53% of runtime on the EU set).
  auto const relax_ping = [&](auto& algo, unixtime_t const t) {
    auto const& sc = get_slack();
    auto const fixed = sc.arr_fixed_min_ >= 0.0;
    algo.set_dest_relax(t, fixed ? 1.0 : sc.arr_,
                        fixed ? static_cast<int>(sc.arr_fixed_min_) : 0,
                        sc.arr_min_min_, sc.arr_cap_min_);
  };
  auto const budget_cap = static_cast<std::uint8_t>(
      std::min<unsigned>(q.max_transfers_ + 1U, kMaxTransfers + 1U));

  // The scan only GROWS the window on the side it runs towards, which matches
  // the requested extension side in the two "aligned" cases below (the same
  // two PONG covers):
  //
  //   arriveBy | extend_later | scan grows | requested
  //   false    | true         | later      | later      aligned
  //   true     | false        | earlier    | earlier    aligned
  //   false    | false        | later      | earlier    opposed
  //   true     | true         | earlier    | later      opposed
  //
  // The opposed half is reachable through paging (cursor_to_query() takes the
  // extension side from the cursor). There a plain bicriteria range search
  // runs up front: it extends the interval as the query asked and its
  // journeys ARE the anchor set. numItineraries is then satisfied on the
  // bicriteria journeys, so more itineraries can come back than asked for -
  // the harmless direction.
  auto const pretrip =
      std::holds_alternative<interval<unixtime_t>>(q.start_time_);
  auto const aligned =
      !pretrip ||
      ((SearchDir == direction::kBackward) != q.extend_interval_later_);

  auto scan_interval = search_interval;
  auto ms_range = std::chrono::steady_clock::duration{};
  auto range_state = raptor_state{};
  if (!aligned) {
    auto const r0 = std::chrono::steady_clock::now();
    auto range_s_state = search_state{};
    auto const ar = run_anchor_search<SearchDir>(tt, rtt, range_s_state,
                                                 range_state, q, timeout);
    scan_interval = ar.interval_;
    for (auto const& j : *ar.journeys_) {
      all_anchors.push_back({j.start_time_, j.dest_time_,
                             static_cast<std::uint8_t>(j.transfers_ + 1U)});
    }
    // A windowed anchor set is truncated for the steps near its far end;
    // the aligned path's per-step ping never has that problem, since it
    // searches from each step with no far boundary at all.
    if (!all_anchors.empty()) {
      close_anchor_profile<SearchDir>(tt, rtt, q, scan_interval, range_state,
                                      timeout, all_anchors);
    }
    ms_range = std::chrono::steady_clock::now() - r0;
  }

  auto start_time =
      kFwd ? scan_interval.from_ : scan_interval.to_ - duration_t{1};
  auto const end_time =
      kFwd ? scan_interval.to_ : scan_interval.from_ - duration_t{1};

  auto starts = std::vector<start>{};
  auto bounds = bmrap_bounds{};
  // staging for host_round_times() - unused by the CPU engines
  auto rt_buf = std::vector<std::array<delta_t, kVias + 1>>{};

  // Forward realizations (see step 5b), kept in the FORWARD convention and
  // spliced in after the results are swapped over. One realization per
  // departure covers the whole scan, since a forward search from d returns
  // the entire Pareto set at d.
  auto realized = std::vector<journey>{};
  auto realized_deps = std::vector<unixtime_t>{};
  auto ms_realize = std::chrono::steady_clock::duration{};
  auto n_realized = std::uint64_t{0U};

  auto anchors = std::vector<anchor>{};
  auto max_trips = std::uint8_t{0U};
  auto budget = std::uint8_t{0U};
  auto anchors_valid_until = std::optional<unixtime_t>{};
  auto n_anchor_recomputes = std::uint64_t{0U};
  auto exit_reason = std::uint64_t{0U};  // 0=cond 1=ping 2=anchors 3=stall

  // tau_arr^-> pruning on both the mc pong and stage 2 (slacked pong). Where
  // the time goes: without it mc pong is unbounded and dominates everything
  // (15-query set 188.5s -> 83.8s; one Berlin -> Montpellier query 78.7s ->
  // 22.5s, mc pong alone 53.6s -> 3.4s). Costs ~15-25% on trivial queries.

  // The restriction, exactly as the final filter applies it. Counting
  // unrestricted journeys would stop the scan on results about to be thrown
  // away - one query returned 6 journeys where 23 exist, because the
  // unrestricted mc pong padded the count on the very first step.
  //
  // Applied here rather than at insertion because A(J) is only final once
  // the scan has passed J's departure: the deciding anchor may not exist
  // yet, so dropping on insertion could discard a journey whose verdict
  // later flips to "keep". Everything counted below is already validated
  // (departure behind start_time), where it cannot change.
  auto const restricted_away = [&](journey const& j) {
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
        if (&o == &j || !o.tuple_dominates(j)) {
          return false;
        }
        // tuple_dominates() is non-strict, so journeys sharing a tuple (which
        // extra criteria produce) dominate one another and a plain any_of()
        // would drop both, undercounting past min_connection_count_ and
        // overshooting the window. Break the tie by address so each distinct
        // tuple contributes one - same as the all_anchors dedup below.
        return !j.tuple_dominates(o) || &o < &j;
      });
    });
  };

  // Stepping past the far end is how the window grows, so it is only
  // allowed when the scan runs towards the side the query asked for. In the
  // opposed case the range search above already settled the window.
  while ((is_better(start_time, end_time) ||
          (aligned && n_results(true) + n_results(false) <
                          2 * static_cast<int>(q.min_connection_count_))) &&
         tt.external_interval().contains(start_time) && !is_timeout()) {
    // The anchor set, and everything derived from it, only goes stale once
    // start_time passes the EARLIEST anchor departure: until then every
    // anchor is still available and no new one can appear, since a
    // two-criteria journey turns Pareto-optimal only when the journey
    // dominating it drops out - which is that same breakpoint. Steps that
    // advance on a MULTICRITERIA breakpoint (the majority) would re-derive
    // an identical set, so they skip phases 1-3 entirely.
    auto const anchors_stale =
        !anchors_valid_until.has_value() ||
        is_better(*anchors_valid_until, start_time);
    if (anchors_stale) {
      ++n_anchor_recomputes;

      if (!aligned) {
        // The anchor set came from the range search up front, so a step only
        // slices out the anchors still AVAILABLE at it - the very slice
        // anchor_of() resolves A(J) in. No ping, no pong.
        anchors.clear();
        max_trips = 0U;
        for (auto const& a : all_anchors) {
          if (!is_better(a.anchored_, start_time)) {
            anchors.push_back(a);
            max_trips = std::max(max_trips, a.trips_);
          }
        }
        if (anchors.empty()) {
          exit_reason = 2U;
          break;  // nothing left to restrict against
        }
        if (fwd_bounds.empty()) {
          // ONE tau_arr^->(v, i) matrix for the whole window instead of one
          // per step: without a per-step ping there is nothing to take it
          // from, and a matrix built at the window's near end is a valid -
          // merely looser - bound for every step inside it, because travel
          // time is never negative and a later step can only reach v later.
          auto const f0 = std::chrono::steady_clock::now();
          starts.clear();
          get_starts(SearchDir, tt, rtt, start_time, q.start_, q.td_start_,
                     q.via_stops_, q.max_start_offset_, q.start_match_mode_,
                     q.use_start_footpaths_, starts, false, q.prf_idx_,
                     q.transfer_time_settings_);
          ping.reset_arrivals();
          ping.next_start_time();
          relax_ping(ping, start_time);
          for (auto const& st : starts) {
            ping.add_start(st.stop_, st.time_at_stop_);
          }
          auto ping_results = pareto_set<journey>{};
          ping.execute(start_time, q.max_transfers_,
                       worst_at_dest(start_time), ping_results);
          fwd_bounds =
              reach_matrix<SearchDir>(tt, q, r_state, ping, rt_buf,
                                      trip_budget(max_trips, budget_cap),
                                      /*sub_transfer=*/true);
          ms_fwd_bounds += std::chrono::steady_clock::now() - f0;
          ++n_fwd_bound_builds;
          mc_pong.set_bounds(&fwd_bounds);
        }
      } else {

      // ---- 1. PING: two-criteria EA from this departure ----
      auto const p0 = std::chrono::steady_clock::now();
      starts.clear();
      get_starts(SearchDir, tt, rtt, start_time, q.start_, q.td_start_,
                 q.via_stops_, q.max_start_offset_, q.start_match_mode_,
                 q.use_start_footpaths_, starts, false, q.prf_idx_,
                 q.transfer_time_settings_);
      ping.reset_arrivals();
      ping.next_start_time();
      relax_ping(ping, start_time);
      for (auto const& s : starts) {
        ping.add_start(s.stop_, s.time_at_stop_);
      }
      auto ping_results = pareto_set<journey>{};
      ping.execute(start_time, q.max_transfers_,
                   worst_at_dest(start_time), ping_results);
      ms_ping += std::chrono::steady_clock::now() - p0;
      utl::sort(ping_results, [&](journey const& a, journey const& b) {
        return is_better(a.dest_time_, b.dest_time_);
      });

      // r_state still holds the ping's round times here; the pong below reuses
      // the same state, so the matrix has to be taken now - before the anchors
      // exist. Size it from the PING's trip counts: re-anchoring in the pong
      // moves departures, never the number of trips. Sizing to budget_cap
      // instead would make the build loop (rounds x locations) dominate cheap
      // queries.
      {
        auto ping_trips = std::uint8_t{0U};
        for (auto const& j : ping_results) {
          ping_trips = std::max(ping_trips,
                                static_cast<std::uint8_t>(j.transfers_ + 1U));
        }
        auto const f0 = std::chrono::steady_clock::now();
        fwd_bounds =
            reach_matrix<SearchDir>(tt, q, r_state, ping, rt_buf,
                                    trip_budget(ping_trips, budget_cap),
                                    /*sub_transfer=*/true);
        ms_fwd_bounds += std::chrono::steady_clock::now() - f0;
        ++n_fwd_bound_builds;
        mc_pong.set_bounds(&fwd_bounds);
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
                     tight);
      }
      ms_pong += std::chrono::steady_clock::now() - g0;

      // tight journeys are (start_time_ = arrival, dest_time_ = departure)
      anchors.clear();
      max_trips = 0U;
      for (auto const& j : tight) {
        auto const trips = static_cast<std::uint8_t>(j.transfers_ + 1U);
        anchors.push_back({j.dest_time_, j.start_time_, trips});
        max_trips = std::max(max_trips, trips);
      }
      if (anchors.empty()) {
        exit_reason = 2U;
        break;
      }
      // Keep all_anchors a genuine (departure, arrival, trips) Pareto set.
      // Exact-duplicate removal is not enough: a later step can re-anchor a
      // journey to a departure an earlier step already covered with a strictly
      // better arrival, and that dominated entry poisons the restriction twice
      // - anchor_of() hands back a looser deadline, and the is_anchor()
      // early-out waves the dominated journey through as its own A(J). That is
      // how a journey arriving 30 min past its deadline once survived.
      auto const dominates = [&](anchor const& x, anchor const& y) {
        return !is_better(x.anchored_, y.anchored_) &&  // departs no earlier
               !is_better(y.found_, x.found_) &&        // arrives no later
               x.trips_ <= y.trips_;
      };
      for (auto const& a : anchors) {
        if (utl::any_of(all_anchors,
                        [&](anchor const& o) { return dominates(o, a); })) {
          continue;
        }
        utl::erase_if(all_anchors,
                      [&](anchor const& o) { return dominates(a, o); });
        all_anchors.emplace_back(a);
      }
      }  // aligned

      budget = trip_budget(max_trips, budget_cap);
      anchors_valid_until = utl::min_element(
          anchors, [&](anchor const& a, anchor const& b) {
            return is_better(a.anchored_, b.anchored_);
          })->anchored_;

      // ---- 3. SLACKED PONG: bounds anchored at THIS departure ----
      {
        auto const b0 = std::chrono::steady_clock::now();
        bounds = compute_bounds<SearchDir, Rt>(
            tt, rtt, q, anchors, base_day,
            /*horizon=*/start_time - duration_t{kFwd ? 1 : -1}, budget, r_state,
            prune_stats, &fwd_bounds);
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
                    worst_at_dest(start_time), q.prf_idx_, mc_results);

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
    // LATEST run that still makes the connection: no slack at any transfer,
    // and one delay loses the chain. Re-running forward from the departure
    // it just pinned yields the same tuple with the EARLIEST connections.
    // Done inside the step because tau_dep^<- is live here - destination
    // pruning alone leaves too much of the network unpruned.
    {
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
    // Anchor departures alone are not enough: between two of them the
    // MULTICRITERIA pareto set can still change (a later departure with
    // different walking becomes optimal), so the step advances to the
    // loosest departure over the anchors AND this step's own mc journeys.
    // Only departures at or after the current step count - the pong runs
    // with worst_time_at_dest = loosest - 1 min, so a validated departure
    // can land one minute BEHIND the step, and letting one of those set the
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
    for (auto const& j : step_results) {
      consider(j.dest_time_);  // dest_time_ is the departure here
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
  // In the opposed case the range search fixed the window, so report it
  // whole: the scan stops at its far end rather than defining it.
  auto const scanned =
      aligned ? (kFwd ? interval<unixtime_t>{scan_interval.from_, start_time}
                      : interval<unixtime_t>{start_time + duration_t{1},
                                             scan_interval.to_})
              : scan_interval;
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
  for (auto& x : s_state.results_) {
    auto const it = utl::find_if(realized, [&](journey const& f) {
      return f.start_time_ == x.start_time_ && f.dest_time_ == x.dest_time_ &&
             f.transfers_ == x.transfers_ &&
             f.criteria_cost_ == x.criteria_cost_ &&
             f.criteria_mode_filter_ == x.criteria_mode_filter_ &&
             f.criteria_mode_switches_ == x.criteria_mode_switches_;
    });
    if (it != end(realized)) {
      x.legs_ = it->legs_;
      ++n_realized;
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
  {
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
  algo_stats["bmrapp_aligned"] = aligned ? 1U : 0U;
  algo_stats["bmrapp_ms_range"] = ms(ms_range);

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

template <typename Criteria, typename AlgoState>
routing_result bmrap_profile_search(
    timetable const& tt,
    rt_timetable const* rtt,
    search_state& s_state,
    AlgoState& algo_state,
    query q,
    direction const search_dir,
    std::optional<std::chrono::seconds> const timeout) {
  auto const run = [&]<int GpuMc>() {
    return search_dir == direction::kForward
               ? bmrap_profile<direction::kForward, false, Criteria, AlgoState,
                               GpuMc>(tt, rtt, s_state, algo_state,
                                      std::move(q), timeout)
               : bmrap_profile<direction::kBackward, false, Criteria, AlgoState,
                               GpuMc>(tt, rtt, s_state, algo_state,
                                      std::move(q), timeout);
  };
  // Probe the device multicriteria state here rather than inside: it is a
  // large allocation that a big timetable can fail, and the mode is a
  // compile-time choice, so a failure discovered later would take the whole
  // GPU search down with it instead of just the optional phases.
  if constexpr (kGpuMcSupported<Criteria, AlgoState> && kBmrapGpuMcMode != 0) {
    if (algo_state.try_mc_state() != nullptr) {
      return kBmrapGpuMcMode >= 2 ? run.template operator()<2>()
                                  : run.template operator()<1>();
    }
  }
  return run.template operator()<0>();
}

// One line per criteria configuration, for each scalar engine. The
// generalized-cost criterion is deliberately absent: it writes the same
// journey slot as walking, so it is not part of the composable set.
#define NIGIRI_BMRAPP_INSTANTIATE(C, S)                                    \
  template routing_result bmrap_profile_search<C>(                        \
      timetable const&, rt_timetable const*, search_state&, S&, query,     \
      direction, std::optional<std::chrono::seconds>);

NIGIRI_BMRAPP_INSTANTIATE(arr_criteria, raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_non_transit_criteria, raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_mode_filter_criteria, raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_mode_switches_criteria, raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_non_transit_mode_filter_criteria, raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_non_transit_mode_switches_criteria, raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_mode_filter_mode_switches_criteria, raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_non_transit_mode_filter_mode_switches_criteria,
                          raptor_state)

#if defined(NIGIRI_CUDA)
NIGIRI_BMRAPP_INSTANTIATE(arr_criteria, gpu::gpu_raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_non_transit_criteria, gpu::gpu_raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_mode_filter_criteria, gpu::gpu_raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_mode_switches_criteria, gpu::gpu_raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_non_transit_mode_filter_criteria,
                          gpu::gpu_raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_non_transit_mode_switches_criteria,
                          gpu::gpu_raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_mode_filter_mode_switches_criteria,
                          gpu::gpu_raptor_state)
NIGIRI_BMRAPP_INSTANTIATE(arr_non_transit_mode_filter_mode_switches_criteria,
                          gpu::gpu_raptor_state)
#endif

#undef NIGIRI_BMRAPP_INSTANTIATE

}  // namespace nigiri::routing
