#include "nigiri/routing/raptor/bmraptor.h"

#include <algorithm>
#include <optional>
#include <variant>
#include <vector>

#include "utl/erase_if.h"
#include "utl/helpers/algorithm.h"
#include "utl/verify.h"

#include "nigiri/routing/dijkstra.h"
#include "nigiri/routing/direct.h"
#include "nigiri/routing/get_fastest_direct.h"
#include "nigiri/routing/raptor/bmrap_common.h"
#include "nigiri/routing/raptor/bmrap_filters.h"
#include "nigiri/routing/raptor/mcraptor.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/special_stations.h"

namespace nigiri::routing {

namespace {

using namespace bmrap_detail;
using clk = std::chrono::steady_clock;

// adds the time since the last lap to an accumulator
struct stopwatch {
  void lap(clk::duration& acc) {
    auto const now = clk::now();
    acc += now - t_;
    t_ = now;
  }
  clk::time_point t_{clk::now()};
};

// Calls fn(arrival, first, last) for each run of `js` (sorted by dest_time_)
// that shares one dest_time_.
template <typename Journeys, typename Fn>
void for_each_arrival_group(Journeys& js, Fn&& fn) {
  for (auto it = begin(js); it != end(js);) {
    auto const arr = it->dest_time_;
    auto const last = std::find_if(
        it, end(js), [&](journey const& j) { return j.dest_time_ != arr; });
    fn(arr, it, last);
    it = last;
  }
}

// PROFILE BM-RAPTOR: one complete single-departure BM-RAPTOR per step of a
// PONG-style scan, so every departure gets bounds anchored at itself (a range
// variant spreading one matrix over a window bottoms out at the window start).
//
//   1. ping          2-criteria earliest arrival from the step's departure
//   2. pong          backward, re-anchors each anchor to its latest departure
//   3. slacked pong  backward pruning from the slack-relaxed anchor times ->
//                    tau_dep^<-(v, i), horizon = this departure
//   4. mc ping       bounded multicriteria search from this departure
//   5. mc pong       backward, re-anchors each mc journey to its latest
//                    departure: search.h never calls set_tight_start(), so a
//                    range McRAPTOR would report slack-departure duplicates
//   5b. re-realize   forward, for better intermediate legs
template <direction SearchDir,
          bool Rt,
          typename Criteria,
          typename AlgoState,
          int GpuMc>
routing_result bmrap_profile(
    timetable const& tt,
    rt_timetable const* rtt,
    search_state& s_state,
    AlgoState& r_state,
    query q,
    std::optional<std::chrono::seconds> const timeout) {
  constexpr auto const kFwd = (SearchDir == direction::kForward);
  static constexpr auto const kDir = duration_t{kFwd ? 1 : -1};
  using ping_t = typename bmrap_algo_for<SearchDir, Rt, AlgoState>::type;
  using pong_t = typename bmrap_algo_for<flip(SearchDir), Rt, AlgoState>::type;
  using mc_ping_for = bmrap_mc_algo_for<SearchDir, Criteria, (GpuMc >= 1)>;
  using mc_pong_for =
      bmrap_mc_algo_for<flip(SearchDir), Criteria, (GpuMc >= 2)>;
  using mc_ping_t = typename mc_ping_for::type;
  using mc_pong_t = typename mc_pong_for::type;

  q.sanitize(tt);
  utl::verify(mcraptor_supported(q, rtt),
              "bmrap_profile: query not supported by the mcraptor engine");

  auto const t0 = clk::now();
  s_state.results_.clear();

  auto const is_better = [](auto const a, auto const b) {
    return bmrap_detail::is_better<SearchDir>(a, b);
  };
  auto const fastest_direct = get_fastest_direct(tt, q, SearchDir);
  auto const search_interval = std::visit(
      utl::overloaded{
          [](interval<unixtime_t> const i) { return i; },
          [](unixtime_t const t) { return interval<unixtime_t>{t, t}; }},
      q.start_time_);
  auto const base_day =
      day_idx_t{std::chrono::duration_cast<date::days>(
                    std::chrono::round<std::chrono::days>(
                        search_interval.from_ +
                        ((search_interval.to_ - search_interval.from_) / 2)) -
                    tt.internal_interval().from_)
                    .count()};

  // destinations and lower bounds, both directions
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
  auto lb_time = clk::duration{};
  auto sw = stopwatch{};
  if constexpr (ping_t::kUseLowerBounds || bounded_needs_lb<mc_ping_t>()) {
    dijkstra(tt, q,
             (kFwd ? tt.fwd_search_lb_graph_[q.prf_idx_]
                   : tt.bwd_search_lb_graph_[q.prf_idx_]),
             nullptr, nullptr, fwd_lb);
  }
  if constexpr (bounded_needs_lb<pong_t>() || bounded_needs_lb<mc_pong_t>()) {
    dijkstra(tt, qf,
             (kFwd ? tt.bwd_search_lb_graph_[q.prf_idx_]
                   : tt.fwd_search_lb_graph_[q.prf_idx_]),
             nullptr, nullptr, bwd_lb);
  }
  sw.lap(lb_time);

  auto is_via = std::array<bitvec, kMaxVias>{};
  auto no_via = std::vector<via_stop>{};

  // One state serves ping, pong and slacked pong, as in pong.cc: within a step
  // they run in sequence and each resets what it needs, and the GPU's buffers
  // are direction-indexed. The mc phases use their own CPU state unless the
  // device implements their shape (see GpuMc); on the device they share one
  // state that outlives the query, the allocation being too large to repeat.
  auto cpu_mc_ping_state = std::conditional_t<(GpuMc >= 1), std::monostate,
                                              basic_mcraptor_state<Criteria>>{};
  auto cpu_mc_pong_state = std::conditional_t<(GpuMc >= 2), std::monostate,
                                              basic_mcraptor_state<Criteria>>{};
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

  auto const make = [&]<typename Algo>(auto& state, bitvec& is_dest,
                                       std::vector<std::uint16_t>& dist,
                                       auto const& td_dest,
                                       std::vector<std::uint16_t>& lb) {
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
  auto ping = make.template operator()<ping_t>(r_state, fwd_is_dest, fwd_dist,
                                               q.td_dest_, fwd_lb);
  auto pong = make.template operator()<pong_t>(r_state, bwd_is_dest, bwd_dist,
                                               qf.td_dest_, bwd_lb);
  auto mc_ping = make.template operator()<mc_ping_t>(
      mc_ping_state, fwd_is_dest, fwd_dist, q.td_dest_, fwd_lb);
  auto mc_pong = make.template operator()<mc_pong_t>(
      mc_pong_state, bwd_is_dest, bwd_dist, qf.td_dest_, bwd_lb);

  if constexpr (GpuMc >= 2) {
    // Phase 5 runs several departures under ONE reset_arrivals(). The device
    // engine carries its reuse frontier across them, the CPU engine (no range
    // reuse) does not, so without this the frontier rejects labels no reported
    // journey dominates (a later-departing, higher-transfer variant with an
    // equal arrival). Within one departure the two agree.
    mc_pong.set_reuse_same_dep();
  }

  auto stats = raptor_stats{};
  auto prune_stats = raptor_stats{};
  auto all_anchors = std::vector<anchor>{};
  auto n_steps = std::uint64_t{0U};
  auto n_bound_builds = std::uint64_t{0U};
  auto n_fwd_bound_builds = std::uint64_t{0U};
  auto ms_ping = clk::duration{};
  auto ms_pong = clk::duration{};
  auto ms_bounds = clk::duration{};
  auto ms_mc = clk::duration{};
  auto ms_mc_pong = clk::duration{};
  auto ms_recon = clk::duration{};
  auto ms_fwd_bounds = clk::duration{};
  auto ms_range = clk::duration{};
  auto ms_realize = clk::duration{};
  auto fwd_bounds = bmrap_bounds{};

  auto const is_timeout = [&]() {
    return timeout && (clk::now() - t0) >= *timeout;
  };
  // nothing slower than the fastest direct connection is of interest
  auto const worst_at_dest = [&](unixtime_t const t) {
    return t + (kFwd ? 1 : -1) * (std::min(fastest_direct, q.max_travel_time_) +
                                  duration_t{1});
  };
  // Relaxes the ping's target pruning by the arrival slack, which makes its
  // round times a valid tau_arr^->(v, i) matrix (paper, Sec. 4.3). The clamps
  // are relax_arr()'s: an unclamped ratio relaxed by hours on long-haul
  // queries and made the ping 53% of runtime on the EU set.
  auto const relax_ping = [&](auto& algo, unixtime_t const t) {
    auto const& sc = get_slack();
    auto const fixed = sc.arr_fixed_min_ >= 0.0;
    algo.set_dest_relax(t, fixed ? 1.0 : sc.arr_,
                        fixed ? static_cast<int>(sc.arr_fixed_min_) : 0,
                        sc.arr_min_min_, sc.arr_cap_min_);
  };
  auto const budget_cap = static_cast<std::uint8_t>(
      std::min<unsigned>(q.max_transfers_ + 1U, kMaxTransfers + 1U));

  auto starts = std::vector<start>{};
  // seeds `algo` with the starts at time t of the forward query q...
  auto const seed_fwd = [&](auto& algo, unixtime_t const t) {
    starts.clear();
    get_starts(SearchDir, tt, rtt, t, q.start_, q.td_start_, q.via_stops_,
               q.max_start_offset_, q.start_match_mode_, q.use_start_footpaths_,
               starts, false, q.prf_idx_, q.transfer_time_settings_);
    algo.next_start_time();
    for (auto const& s : starts) {
      algo.add_start(s.stop_, s.time_at_stop_);
    }
  };
  // ...or of its flipped counterpart qf
  auto const seed_bwd = [&](auto& algo, unixtime_t const t) {
    starts.clear();
    get_starts(flip(SearchDir), tt, rtt, t, qf.start_, qf.td_start_,
               qf.via_stops_, qf.max_start_offset_, qf.start_match_mode_,
               qf.start_match_mode_ != location_match_mode::kIntermodal, starts,
               false, q.prf_idx_, q.transfer_time_settings_);
    algo.next_start_time();
    for (auto const& s : starts) {
      algo.add_start(s.stop_, s.time_at_stop_);
    }
  };
  auto const run_ping = [&](unixtime_t const t) {
    auto results = pareto_set<journey>{};
    ping.reset_arrivals();
    relax_ping(ping, t);
    seed_fwd(ping, t);
    ping.execute(t, q.max_transfers_, worst_at_dest(t), results);
    return results;
  };
  auto const run_mc_ping =
      [&](unixtime_t const t, std::uint8_t const max_transfers,
          unixtime_t const worst, pareto_set<journey>& results) {
        mc_ping.reset_arrivals();
        seed_fwd(mc_ping, t);
        mc_ping.execute(t, max_transfers, worst, results);
      };

  // The scan only GROWS the window on the side it runs towards, which matches
  // the requested extension side in two "aligned" cases (the two PONG covers):
  //
  //   arriveBy | extend_later | scan grows | requested
  //   false    | true         | later      | later      aligned
  //   true     | false        | earlier    | earlier    aligned
  //   false    | false        | later      | earlier    opposed
  //   true     | true         | earlier    | later      opposed
  //
  // The opposed half is reachable through paging (cursor_to_query() takes the
  // extension side from the cursor). There a plain bicriteria range search
  // runs up front: it extends the interval as asked and its journeys ARE the
  // anchor set. numItineraries is then satisfied on the bicriteria journeys, so
  // more itineraries than asked may come back - the harmless direction.
  auto const pretrip =
      std::holds_alternative<interval<unixtime_t>>(q.start_time_);
  auto const aligned = !pretrip || ((SearchDir == direction::kBackward) !=
                                    q.extend_interval_later_);

  auto scan_interval = search_interval;
  auto range_state = raptor_state{};
  if (!aligned) {
    sw = stopwatch{};
    auto range_s_state = search_state{};
    auto const ar = run_anchor_search<SearchDir>(tt, rtt, range_s_state,
                                                 range_state, q, timeout);
    scan_interval = ar.interval_;
    for (auto const& j : *ar.journeys_) {
      all_anchors.push_back({j.start_time_, j.dest_time_,
                             static_cast<std::uint8_t>(j.transfers_ + 1U)});
    }
    // A windowed anchor set is truncated near its far end; the aligned path's
    // per-step ping searches with no far boundary and never is.
    if (!all_anchors.empty()) {
      close_anchor_profile<SearchDir>(tt, rtt, q, scan_interval, range_state,
                                      timeout, all_anchors);
    }
    sw.lap(ms_range);
  }

  auto start_time =
      kFwd ? scan_interval.from_ : scan_interval.to_ - duration_t{1};
  auto const end_time =
      kFwd ? scan_interval.to_ : scan_interval.from_ - duration_t{1};

  auto bounds = bmrap_bounds{};

  // Forward realizations (step 5b), in the forward convention, spliced in after
  // the results are swapped over. One per departure covers the whole scan: a
  // forward search from d returns the entire Pareto set at d.
  auto realized = std::vector<journey>{};
  auto realized_deps = std::vector<unixtime_t>{};
  auto n_realized = std::uint64_t{0U};

  auto anchors = std::vector<anchor>{};
  auto max_trips = std::uint8_t{0U};
  auto budget = std::uint8_t{0U};
  auto anchors_valid_until = std::optional<unixtime_t>{};
  auto n_anchor_recomputes = std::uint64_t{0U};
  auto exit_reason = std::uint64_t{0U};  // 0=cond 1=ping 2=anchors 3=stall

  // Inside the scan, journeys carry mc pong's backward convention: dest_time_
  // is the DEPARTURE and start_time_ the ARRIVAL. They are swapped after it.
  //
  // The restriction is applied when counting, not on insertion, because A(J)
  // is only final once the scan has passed J's departure: dropping earlier
  // could discard a journey whose verdict later flips to "keep". Counting
  // unrestricted journeys would stop the scan on results about to be thrown
  // away (6 returned where 23 exist).
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
        // Count as the response shows them: every Pareto-optimal journey, also
        // the several an extra criterion produces on one (departure, arrival,
        // transfers) tuple. Counting tuples kept the scan stepping long past
        // numItineraries. dominates() is non-strict, so break ties between
        // exact duplicates by address.
        return !j.dominates(o) || &o < &j;
      });
    });
  };

  // Stepping past the far end is how the window grows, so it is only allowed
  // when the scan runs towards the requested side; in the opposed case the
  // range search above already settled the window.
  while ((is_better(start_time, end_time) ||
          (aligned && n_results(true) + n_results(false) <
                          2 * static_cast<int>(q.min_connection_count_))) &&
         tt.external_interval().contains(start_time) && !is_timeout()) {
    // The anchor set, and everything derived from it, only goes stale once
    // start_time passes the earliest anchor departure: until then every anchor
    // is still available, and a 2-criteria journey only turns Pareto-optimal
    // when the one dominating it drops out - the same breakpoint. Steps on a
    // multicriteria breakpoint (the majority) skip phases 1-3.
    auto const anchors_stale = !anchors_valid_until.has_value() ||
                               is_better(*anchors_valid_until, start_time);
    if (anchors_stale) {
      ++n_anchor_recomputes;

      if (!aligned) {
        // the anchor set comes from the range search: slice out the anchors
        // still available at this step
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
          break;
        }
        if (fwd_bounds.empty()) {
          // One tau_arr^->(v, i) matrix for the whole window, without a
          // per-step ping to take it from. Built at the window's near end it is
          // a valid, merely looser, bound for every later step.
          sw = stopwatch{};
          run_ping(start_time);
          fwd_bounds = reach_matrix<SearchDir>(
              tt, q, r_state, ping, trip_budget(max_trips, budget_cap),
              /*sub_transfer=*/true);
          sw.lap(ms_fwd_bounds);
          ++n_fwd_bound_builds;
          mc_pong.set_bounds(&fwd_bounds);
        }
      } else {
        // ---- 1. PING ----
        sw = stopwatch{};
        auto ping_results = run_ping(start_time);
        sw.lap(ms_ping);
        utl::sort(ping_results, [&](journey const& a, journey const& b) {
          return is_better(a.dest_time_, b.dest_time_);
        });

        // r_state still holds the ping's round times, which the pong reuses,
        // so take the matrix now. Size it by the ping's trip counts (the pong
        // moves departures, never trip counts): budget_cap would make the
        // rounds x locations loop dominate cheap queries.
        auto ping_trips = std::uint8_t{0U};
        for (auto const& j : ping_results) {
          ping_trips = std::max(ping_trips,
                                static_cast<std::uint8_t>(j.transfers_ + 1U));
        }
        fwd_bounds = reach_matrix<SearchDir>(
            tt, q, r_state, ping, trip_budget(ping_trips, budget_cap),
            /*sub_transfer=*/true);
        sw.lap(ms_fwd_bounds);
        ++n_fwd_bound_builds;
        mc_pong.set_bounds(&fwd_bounds);
        pong.set_bounds(&fwd_bounds);

        // ---- 2. PONG: re-anchor each anchor to its latest departure ----
        auto tight = pareto_set<journey>{};
        pong.reset_arrivals();
        for_each_arrival_group(ping_results, [&](unixtime_t const arr,
                                                 auto const first,
                                                 auto const last) {
          auto const max_tr =
              std::max_element(first, last,
                               [](journey const& a, journey const& b) {
                                 return a.transfers_ < b.transfers_;
                               })
                  ->transfers_;
          auto const loosest =
              std::min_element(first, last,
                               [&](journey const& a, journey const& b) {
                                 return is_better(a.start_time_, b.start_time_);
                               })
                  ->start_time_;
          seed_bwd(pong, arr);
          pong.execute(arr, max_tr, loosest - kDir, tight);
        });
        sw.lap(ms_pong);

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
        // Keep all_anchors a genuine (departure, arrival, trips) Pareto set. A
        // later step can re-anchor a journey to a departure an earlier step
        // covered with a strictly better arrival; that dominated entry would
        // poison the restriction twice (a looser anchor_of() deadline, and
        // outside_restriction() waving the dominated journey through as its own
        // A(J)), once letting a journey arrive 30 min past its deadline.
        auto const dominates = [&](anchor const& x, anchor const& y) {
          return !is_better(x.anchored_, y.anchored_) &&  // departs no earlier
                 !is_better(y.found_, x.found_) &&  // arrives no later
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
      }

      budget = trip_budget(max_trips, budget_cap);
      anchors_valid_until =
          utl::min_element(anchors, [&](anchor const& a, anchor const& b) {
            return is_better(a.anchored_, b.anchored_);
          })->anchored_;

      // ---- 3. SLACKED PONG: bounds anchored at THIS departure ----
      sw = stopwatch{};
      bounds = compute_bounds<SearchDir, Rt>(
          tt, rtt, q, anchors, base_day, /*horizon=*/start_time - kDir, budget,
          r_state, prune_stats, &fwd_bounds);
      sw.lap(ms_bounds);
      ++n_bound_builds;
      mc_ping.set_bounds(&bounds);
    }
    ++n_steps;

    // ---- 4. MC PING ----
    sw = stopwatch{};
    auto mc_results = pareto_set<journey>{};
    run_mc_ping(start_time, static_cast<std::uint8_t>(budget - 1U),
                worst_at_dest(start_time), mc_results);
    sw.lap(ms_mc);

    // ---- 5. MC PONG: re-anchor each mc journey to its latest departure ----
    // skip journeys an earlier step already validated
    utl::erase_if(mc_results, [&](journey const& x) {
      return s_state.results_.is_dominated(x);
    });
    utl::sort(mc_results, [&](journey const& a, journey const& b) {
      return is_better(a.dest_time_, b.dest_time_);
    });

    mc_pong.reset_arrivals();
    auto step_results = pareto_set<journey>{};
    for_each_arrival_group(mc_results, [&](unixtime_t const arr,
                                           auto const first, auto const last) {
      seed_bwd(mc_pong, arr);
      // may not depart before the step being scanned
      mc_pong.execute(arr,
                      std::max_element(first, last,
                                       [](journey const& a, journey const& b) {
                                         return a.transfers_ < b.transfers_;
                                       })
                          ->transfers_,
                      start_time - kDir, step_results);
    });
    sw.lap(ms_mc_pong);

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
    sw.lap(ms_recon);

    // ---- 5b. RE-REALIZE FORWARD ----
    // mc pong reconstructs backwards, so every intermediate leg is the LATEST
    // run that still connects: no slack at any transfer, and one delay loses
    // the chain. Re-running forward from the departure it just pinned yields
    // the same tuple with the EARLIEST connections. Done inside the step
    // because tau_dep^<- is live here; destination pruning alone leaves too
    // much of the network unpruned.
    auto deps = std::vector<unixtime_t>{};
    for (auto const& j : step_results) {
      if (!j.error_ && j.is_reconstructed_ &&
          utl::find(deps, j.dest_time_) == end(deps) &&
          utl::find(realized_deps, j.dest_time_) == end(realized_deps)) {
        deps.emplace_back(j.dest_time_);
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
          loosest_arr = j.start_time_;
        }
      }
      auto fwd = pareto_set<journey>{};
      run_mc_ping(d, max_tr, *loosest_arr + kDir, fwd);
      for (auto& f : fwd) {
        if (f.is_reconstructed_ || f.error_) {
          continue;
        }
        try {
          mc_ping.reconstruct(q, f);
        } catch (std::exception const& e) {
          f.error_ = true;
          log(log_lvl::error, "bmrap_profile", "realize failed: {}", e.what());
        }
      }
      realized_deps.emplace_back(d);
      for (auto const& f : fwd) {
        if (!f.error_ && f.is_reconstructed_) {
          realized.emplace_back(f);
        }
      }
    }
    sw.lap(ms_realize);

    // ---- 6. advance ----
    // The multicriteria set can still change between two anchor departures (a
    // later departure with different walking becomes optimal), so advance to
    // the loosest departure over the anchors AND this step's mc journeys. Only
    // departures at or after this step count: the pong's worst_time_at_dest is
    // loosest - 1 min, so a validated departure can land one minute behind the
    // step, and letting it set the advance would stall the scan.
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
    auto const next = loosest_dep.value_or(start_time) + kDir;
    if (!is_better(start_time, next)) {
      exit_reason = 3U;
      break;
    }
    start_time = next;
  }

  stats = ping.get_stats() + pong.get_stats() + mc_ping.get_stats() +
          mc_pong.get_stats();

  // ---- results: still (arrival, departure); make them journeys ----
  // In the opposed case the range search fixed the window: report it whole.
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

  // The legs so far come from mc_pong.reconstruct(qf, ...), so their front/back
  // special stations are mirrored. Fix that BEFORE splicing: the realized legs
  // come from mc_ping.reconstruct(q, ...) and are already in the final
  // convention.
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

  // Splice in the forward realizations: same journey, better legs. Only legs_
  // moves; the tuple must come out identical.
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

  // restrict to J_R
  auto n_restricted = std::uint64_t{0U};
  utl::erase_if(s_state.results_, [&](journey const& j) {
    auto const drop = outside_restriction<SearchDir>(
        all_anchors, j.start_time_, j.dest_time_,
        static_cast<unsigned>(j.transfers_) + 1U);
    n_restricted += drop ? 1U : 0U;
    return drop;
  });

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
  algo_stats["bmrapp_ms_lb"] = ms(lb_time);
  algo_stats["bmrapp_aligned"] = aligned ? 1U : 0U;
  algo_stats["bmrapp_ms_range"] = ms(ms_range);

  return routing_result{
      .journeys_ = &s_state.results_,
      .interval_ = scanned,
      .search_stats_ =
          {.lb_time_ = ms(lb_time),
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
    query q,
    direction const search_dir,
    std::optional<std::chrono::seconds> const timeout,
    int const gpu_mc_mode) {
  auto const run = [&]<int GpuMc>() {
    return search_dir == direction::kForward
               ? bmrap_profile<direction::kForward, false, Criteria, AlgoState,
                               GpuMc>(tt, rtt, s_state, algo_state,
                                      std::move(q), timeout)
               : bmrap_profile<direction::kBackward, false, Criteria, AlgoState,
                               GpuMc>(tt, rtt, s_state, algo_state,
                                      std::move(q), timeout);
  };
  // Probe the device mc state here: it is a large allocation a big timetable
  // can fail, and a later failure would take the whole GPU search down instead
  // of just the optional phases.
  if constexpr (kGpuMcSupported<Criteria, AlgoState>) {
    if (gpu_mc_mode != 0 && algo_state.try_mc_state() != nullptr) {
      return gpu_mc_mode >= 2 ? run.template operator()<2>()
                              : run.template operator()<1>();
    }
  }
  return run.template operator()<0>();
}

// One line per criteria configuration and scalar engine. Generalized cost is
// absent: it writes the same journey slot as non_transit.
#define NIGIRI_BMRAPP_INSTANTIATE(C, S)                                \
  template routing_result bmrap_profile_search<C>(                     \
      timetable const&, rt_timetable const*, search_state&, S&, query, \
      direction, std::optional<std::chrono::seconds>, int);

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
