#include "nigiri/routing/raptor/pong.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <map>
#include <ranges>
#include <type_traits>

#include "utl/helpers/algorithm.h"
#include "utl/sorted_diff.h"
#include "utl/timing.h"

#include "nigiri/location_match_mode.h"
#include "nigiri/routing/direct.h"
#include "nigiri/routing/get_earliest_transport.h"
#include "nigiri/routing/gpu/mcraptor.h"
#include "nigiri/routing/gpu/raptor.h"
#include "nigiri/routing/leg_alternatives.h"
#include "nigiri/routing/raptor/mcraptor.h"
#include "nigiri/routing/transfer_time_settings.h"
#include "nigiri/rt/frun.h"
#include "nigiri/types.h"

#define trace_pong(...)
// #define trace_pong fmt::println

namespace nigiri::routing {

template <direction SearchDir, via_offset_t Vias, bool Rt, typename AlgoState,
          bool RangeReuse>
struct pong_algo_for {
  using type = raptor<SearchDir, Rt, Vias, search_mode::kOneToOne>;
};

template <direction SearchDir, via_offset_t Vias, bool Rt, typename Criteria,
          bool RangeReuse>
struct pong_algo_for<SearchDir, Vias, Rt, basic_mcraptor_state<Criteria>,
                     RangeReuse> {
  using type = basic_mcraptor<SearchDir, Criteria, RangeReuse>;
};

#if defined(NIGIRI_CUDA)
template <direction SearchDir, via_offset_t Vias, bool Rt, bool RangeReuse>
struct pong_algo_for<SearchDir, Vias, Rt, gpu::gpu_raptor_state, RangeReuse> {
  using type = gpu::gpu_raptor<SearchDir>;
};

// GPU mcraptor pong: the state's frontiers are per-direction, so the
// forward ping and backward pong coexist; the GPU's reuse frontier
// replaces the CPU's RangeReuse template switch (result-neutral).
template <direction SearchDir, via_offset_t Vias, bool Rt, bool RangeReuse>
struct pong_algo_for<SearchDir, Vias, Rt, gpu::gpu_mcraptor_state,
                     RangeReuse> {
  using type = gpu::gpu_mcraptor<SearchDir, /*WithCost=*/false>;
};

template <direction SearchDir, via_offset_t Vias, bool Rt, bool RangeReuse>
struct pong_algo_for<SearchDir, Vias, Rt, gpu::gpu_mcraptor_cost_state,
                     RangeReuse> {
  using type = gpu::gpu_mcraptor<SearchDir, /*WithCost=*/true>;
};
#endif

auto to_tuple(journey const& j) {
  return std::tuple{j.departure_time(), j.arrival_time(), j.transfers_};
}

std::optional<std::array<journey::leg, 3U>> get_earliest_alternative(
    timetable const& tt,
    rt_timetable const* rtt,
    query const& q,
    location_idx_t const from,
    location_idx_t const to,
    unixtime_t const from_arr,
    unixtime_t const to_dep) {
  auto const direct_query = make_alternative_query(tt, rtt, q, from, to);
  auto cursor =
      get_direct_journeys<direction::kForward>(tt, rtt, direct_query, from_arr);
  if (!cursor) {
    return std::nullopt;
  }
  auto legs = cursor();
  if (legs.back().arr_time_ > to_dep) {
    return std::nullopt;
  }
  return std::array{std::move(legs[0]), std::move(legs[1]), std::move(legs[2])};
}

template <direction SearchDir, bool Rt, via_offset_t Vias, typename AlgoState>
routing_result pong(timetable const& tt,
                    rt_timetable const* rtt,
                    search_state& s_state,
                    AlgoState& r_state,
                    query q,
                    std::optional<std::chrono::seconds> timeout) {
  constexpr auto kFwd = (SearchDir == direction::kForward);

  // forward ping = plain EA over the window (no bag reuse for mcraptor);
  // backward pong = validation toward fixed anchors (rRAPTOR bag reuse ON)
  using ping_algo_t =
      typename pong_algo_for<SearchDir, Vias, Rt, AlgoState, false>::type;
  using pong_algo_t =
      typename pong_algo_for<flip(SearchDir), Vias, Rt, AlgoState, true>::type;

  s_state.results_.clear();
  q.sanitize(tt);

  auto const processing_start_time = std::chrono::steady_clock::now();

  auto const fastest_direct = get_fastest_direct(tt, q, SearchDir);
  auto const search_interval = std::visit(
      utl::overloaded{[](interval<unixtime_t> const start_interval) {
                        return start_interval;
                      },
                      [](unixtime_t const start_time) {
                        return interval<unixtime_t>{start_time, start_time};
                      }},
      q.start_time_);
  auto const base_day =
      day_idx_t{std::chrono::duration_cast<date::days>(
                    std::chrono::round<std::chrono::days>(
                        search_interval.from_ +
                        ((search_interval.to_ - search_interval.from_) / 2)) -
                    tt.internal_interval().from_)
                    .count()};

  // ====
  // PING
  // ----
  constexpr auto const kGpu = std::is_same_v<AlgoState, gpu::gpu_raptor_state>;

  auto ping_dist_to_dest = std::vector<std::uint16_t>{};
  auto ping_is_dest = bitvec{};
  auto ping_is_via = std::array<bitvec, kMaxVias>{};
  collect_destinations(tt, q.destination_, q.dest_match_mode_, ping_is_dest,
                       ping_dist_to_dest);
  for (auto const [i, via] : utl::enumerate(q.via_stops_)) {
    collect_via_destinations(tt, via.location_, ping_is_via[i]);
  }

  auto lb_time = std::chrono::steady_clock::duration{};
  auto ping_lb = std::vector<std::uint16_t>{};
  auto pong_lb = std::vector<std::uint16_t>{};
  // the two lb dijkstras are independent (ping: destination-rooted on the
  // search-direction graph; pong: start-rooted on the flipped graph), so
  // overlap them - they dominate the per-query CPU critical path
  auto pong_lb_time = std::chrono::steady_clock::duration{};
  auto pong_lb_fut = std::future<void>{};
  if constexpr (pong_algo_t::kUseLowerBounds) {
    pong_lb_fut = std::async(std::launch::async, [&tt, rtt, q, &pong_lb,
                                                  &pong_lb_time] {
      auto const t0 = std::chrono::steady_clock::now();
      auto qf = q;
      qf.flip_dir();
      dijkstra(tt, qf,
               (kFwd ? tt.bwd_search_lb_graph_[qf.prf_idx_]
                     : tt.fwd_search_lb_graph_[qf.prf_idx_]),
               ((rtt == nullptr || kGpu)
                    ? nullptr
                    : &(kFwd ? rtt->bwd_search_lb_graph_has_edges_
                             : rtt->fwd_search_lb_graph_has_edges_)),
               ((rtt == nullptr || kGpu)
                    ? nullptr
                    : &(kFwd ? rtt->bwd_search_lb_graph_
                             : rtt->fwd_search_lb_graph_)),
               pong_lb);
      pong_lb_time = std::chrono::steady_clock::now() - t0;
    });
  }
  auto const ping_lb_start = std::chrono::steady_clock::now();
  if constexpr (ping_algo_t::kUseLowerBounds) {
    dijkstra(tt, q,
             (kFwd ? tt.fwd_search_lb_graph_[q.prf_idx_]
                   : tt.bwd_search_lb_graph_[q.prf_idx_]),
             ((rtt == nullptr || kGpu)
                  ? nullptr
                  : &(kFwd ? rtt->fwd_search_lb_graph_has_edges_
                           : rtt->bwd_search_lb_graph_has_edges_)),
             ((rtt == nullptr || kGpu) ? nullptr
                                       : &(kFwd ? rtt->fwd_search_lb_graph_
                                                : rtt->bwd_search_lb_graph_)),
             ping_lb);
  }
  lb_time += std::chrono::steady_clock::now() - ping_lb_start;

  auto ping = ping_algo_t{tt,
                          rtt,
                          r_state,
                          ping_is_dest,
                          ping_is_via,
                          ping_dist_to_dest,
                          q.td_dest_,
                          ping_lb,
                          q.via_stops_,
                          base_day,
                          q.allowed_claszes_,
                          q.require_bike_transport_,
                          q.require_car_transport_,
                          q.prf_idx_ == 2U,
                          q.transfer_time_settings_};
  if constexpr (requires { ping.set_tight_start(); }) {
    // the ping sweeps its whole window in one step: without tight starts
    // its journeys are priced from the step start and the contained
    // phantom waiting collapses dep-normalized cost-pareto variants
    // before they can become pong anchors (the dual of the phantom-wait
    // DROP case below). Tight starts re-anchor every ping journey at its
    // latest feasible departure - search.h semantics.
    ping.set_tight_start();
  }

  // ====
  // PONG
  // ----
  q.flip_dir();

  auto pong_dist_to_dest = std::vector<std::uint16_t>{};
  auto pong_is_dest = bitvec{};
  collect_destinations(tt, q.destination_, q.dest_match_mode_, pong_is_dest,
                       pong_dist_to_dest);

  auto pong_is_via = std::array<bitvec, kMaxVias>{};
  for (auto const [i, via] : utl::enumerate(q.via_stops_)) {
    collect_via_destinations(tt, via.location_, pong_is_via[i]);
  }

  if constexpr (pong_algo_t::kUseLowerBounds) {
    pong_lb_fut.get();  // overlapped with the ping dijkstra + ping setup
    lb_time += pong_lb_time;
  }
  auto pong = pong_algo_t{tt,
                          rtt,
                          r_state,
                          pong_is_dest,
                          pong_is_via,
                          pong_dist_to_dest,
                          q.td_dest_,
                          pong_lb,
                          q.via_stops_,
                          base_day,
                          q.allowed_claszes_,
                          q.require_bike_transport_,
                          q.require_car_transport_,
                          q.prf_idx_ == 2U,
                          q.transfer_time_settings_};
  // pong-side engines: the persistent reuse frontier may only reject
  // against SAME-departure entries (= the same merged anchor run, plain
  // dominance semantics). Cross-anchor rejections were observed to fire
  // without a real dominating journey behind them (see the q#45 trace:
  // anchor (13:35, +1d 12:56, 8) unreproducible though no dominator can
  // exist, or the ping pareto would not contain the anchor).
  if constexpr (requires { pong.set_reuse_same_dep(); }) {
    pong.set_reuse_same_dep();
  }

  q.flip_dir();

  // ========
  // >> PLAY!
  // --------
  auto starts = std::vector<start>{};
  auto result = routing_result{
      .journeys_ = &s_state.results_,
      .interval_ = search_interval,
      .search_stats_ =
          {.lb_time_ = static_cast<std::uint64_t>(
               std::chrono::duration_cast<std::chrono::milliseconds>(lb_time)
                   .count())},
      .algo_stats_ = {}};
  auto start_time =
      kFwd ? search_interval.from_ : search_interval.to_ - duration_t{1};
  auto const end_time =
      kFwd ? search_interval.to_ : search_interval.from_ - duration_t{1};
  auto const is_better = [](auto a, auto b) { return kFwd ? a < b : a > b; };
  auto const is_validated = [&](journey const& j) {
    return is_better(j.dest_time_, start_time);
  };
  auto const is_tuple_dominated = [&](journey const& j) {
    return utl::any_of(*result.journeys_, [&](journey const& o) {
      return &o != &j && o.tuple_dominates(j);
    });
  };
  auto const get_result_count = [&](bool const include_too_slow) {
    return utl::count_if(*result.journeys_, [&](journey const& j) {
      return is_validated(j) &&
             (include_too_slow || (j.travel_time() < fastest_direct &&
                                   j.travel_time() < q.max_travel_time_)) &&
             !is_tuple_dominated(j);
    });
  };
  auto const is_timeout_reached = [&]() {
    if (timeout) {
      return (std::chrono::steady_clock::now() - processing_start_time) >=
             *timeout;
    }
    return false;
  };
  // NIGIRI_PONG_STATS: how often is the identical pong anchor re-executed
  // within one query? (feasibility data for a per-query result cache)
  static bool const pong_stats = std::getenv("NIGIRI_PONG_STATS") != nullptr;
  auto pong_exec_total = 0U;
  auto pong_exec_keys = std::map<std::pair<unixtime_t, std::uint8_t>, unsigned>{};

  while ((is_better(start_time, end_time) ||
          get_result_count(true) + get_result_count(false) <
              2 * q.min_connection_count_) &&
         tt.external_interval().contains(start_time) && !is_timeout_reached()) {
    // ----
    // PING
    // ----

    trace_pong("START_TIME={}", start_time);

    starts.clear();
    get_starts(SearchDir, tt, rtt, start_time, q.start_, q.td_start_,
               q.via_stops_, q.max_start_offset_, q.start_match_mode_,
               q.use_start_footpaths_, starts, false, q.prf_idx_,
               q.transfer_time_settings_);
    ping.reset_arrivals();
    ping.next_start_time();
    for (auto const& s : starts) {
      trace_pong("--- PING START: {} at time_at_start={} time_at_stop={}",
                 loc{tt, s.stop_}, s.time_at_start_, s.time_at_stop_);
      ping.add_start(s.stop_, s.time_at_stop_);
    }
    auto const worst_time_at_dest =
        start_time + (kFwd ? 1 : -1) * (q.max_travel_time_ + duration_t{1});
    auto ping_results = pareto_set<journey>{};
    ping.execute(start_time, q.max_transfers_, worst_time_at_dest, q.prf_idx_,
                 ping_results);
    kFwd ? ++result.search_stats_.n_execute_fwd_
         : ++result.search_stats_.n_execute_bwd_;
    if (ping_results.empty()) {
      trace_pong(
          "EMPTY PING RESULTS -> QUIT (max_transfers={}, "
          "worst_time_at_dest={})",
          q.max_transfers_, worst_time_at_dest);
      break;
    }
    utl::erase_if(ping_results, [&](journey const& x) {
      auto const dominated = result.journeys_->is_dominated(x);
      if (dominated) {
        trace_pong("DELETE DOMINATED {}", to_tuple(x));
      }
      return dominated;
    });
    // validation anchors must be processed in dominance order (best
    // arrival first): the pong search's destination frontier persists
    // across anchors, so entries may only stem from anchors whose
    // journeys dominate on the anchor dimension - then a pruned label
    // always corresponds to a journey dominated by an already-validated
    // one (see the no-match handling below)
    utl::sort(ping_results, [&](journey const& a, journey const& b) {
      return is_better(a.dest_time_, b.dest_time_);
    });

    // ----
    // PONG
    // ----
    q.flip_dir();
    // one pong search per distinct anchor ARRIVAL time: all same-arrival
    // pareto anchors (transfers x cost trade-offs) are answered by a
    // single merged run with the group-maximal attributes (max
    // transfers, loosest departure bound). The per-anchor matching
    // below stays exact - a looser bound can only find equal-or-better
    // departures for each tuple.
    // reset per STEP, reuse across groups: anchor groups are processed
    // in ascending arrival order, so a persisted completed journey that
    // prunes a later group's label fully dominates every journey that
    // label could produce (arr ordering closes the frame gap); partial-
    // label reuse rejections are same-departure-gated by the engines.
    pong.reset_arrivals();
    auto g_end = begin(ping_results);
    for (auto pi = begin(ping_results); pi != end(ping_results); ++pi) {
      auto& ping_j = *pi;
      if (pi == g_end) {  // new arrival group -> one merged search
        auto const g_arr = ping_j.dest_time_;
        g_end = std::find_if(pi, end(ping_results), [&](journey const& j) {
          return j.dest_time_ != g_arr;
        });
        auto max_transfers = ping_j.transfers_;
        auto loosest_start = ping_j.start_time_;
        for (auto it = std::next(pi); it != g_end; ++it) {
          max_transfers = std::max(max_transfers, it->transfers_);
          if (is_better(it->start_time_, loosest_start)) {
            loosest_start = it->start_time_;
          }
        }

        starts.clear();
        get_starts(flip(SearchDir), tt, rtt, g_arr, q.start_, q.td_start_,
                   q.via_stops_, q.max_start_offset_, q.start_match_mode_,
                   q.start_match_mode_ != location_match_mode::kIntermodal,
                   starts, false, q.prf_idx_, q.transfer_time_settings_);
        pong.next_start_time();
        for (auto const& s : starts) {
          trace_pong("---- PONG START: {} at time_at_start={} time_at_stop={}",
                     loc{tt, s.stop_}, s.time_at_start_, s.time_at_stop_);
          pong.add_start(s.stop_, s.time_at_stop_);
        }
        if (pong_stats) {
          ++pong_exec_total;
          ++pong_exec_keys[{g_arr, max_transfers}];
        }
        pong.execute(g_arr, max_transfers,
                     loosest_start - duration_t{kFwd ? 1 : -1}, q.prf_idx_,
                     s_state.results_);
        kFwd ? ++result.search_stats_.n_execute_bwd_
             : ++result.search_stats_.n_execute_fwd_;
      }
      trace_pong("-- PING RESULT: {}", to_tuple(ping_j));

      // multi-criteria configurations can hold several pareto journeys
      // with the same (transfers, start_time) - e.g. walking trade-offs:
      // reconstruct all of them, the tuple-optimal departure updates ping
      auto match = end(s_state.results_);
      for (auto it = begin(s_state.results_); it != end(s_state.results_);
           ++it) {
        if (it->transfers_ == ping_j.transfers_ &&
            it->start_time_ == ping_j.dest_time_) {
          if (!it->is_reconstructed_ && !it->error_) {
            pong.reconstruct(q, *it);
          }
          if (match == end(s_state.results_) ||
              is_better(match->dest_time_, it->dest_time_)) {
            match = it;
          }
        }
      }

      if (match == end(s_state.results_)) {
        // No journey for this anchor: the anchor's journey is dominated by
        // an already-validated journey (its reverse labels lost the result
        // pareto set / the destination frontier - both only ever remove
        // dominated journeys). Advance the ping journey to the dominating
        // journey's validated departure: the dominance implies the
        // anchor's own latest departure cannot be later, so its slot is
        // fully covered and the progression is not held back (otherwise
        // the search crawls minute-by-minute and the count-based
        // termination can stop the frontier before later journeys already
        // found validate).
        auto superseding = end(s_state.results_);
        for (auto it = begin(s_state.results_); it != end(s_state.results_);
             ++it) {
          // a TRUE dominator departs no earlier than the anchor's proven
          // departure - without this clause a non-dominating journey can
          // drag the progression backwards into a silent stall
          if (it->transfers_ <= ping_j.transfers_ &&
              !is_better(ping_j.dest_time_, it->start_time_) &&
              !is_better(it->dest_time_, ping_j.start_time_) &&
              (superseding == end(s_state.results_) ||
               is_better(superseding->dest_time_, it->dest_time_))) {
            superseding = it;
          }
        }
        if (superseding == end(s_state.results_)) {
          throw utl::fail(
              "no pong for transfers={}, start_time={} found, journeys={}",
              ping_j.transfers_, ping_j.dest_time_,
              s_state.results_.els_ | std::views::transform(to_tuple));
        }
        trace_pong("---- DROP superseded ping {} [start -> {}]",
                   to_tuple(ping_j), superseding->dest_time_);
        ping_j.start_time_ = superseding->dest_time_;
        continue;
      }

      // If the pong's best journey for this (arrival, transfers) tuple
      // departs worse than the ping journey's own proven departure, the
      // ping journey is not on the pong's pareto frontier - it is
      // dominated. This can only happen with a departure-dependent
      // generalized cost (elapsed time charges waiting): the forward ping
      // prices a journey at an early departure where a competing journey
      // sits idle, inflating that competitor's cost; the backward pong
      // prices every journey at its latest, minimum-wait departure, where a
      // fewer-transfer/cheaper journey dominates the ping journey outright.
      // The dominating journey is already validated into the results, so
      // this anchor must neither be emitted nor hold the progression back
      // to an earlier departure. Drop it. (For a non-cost search the pong
      // can always at least reproduce the ping journey, so this never
      // triggers there and behaviour is unchanged.)
      if (is_better(match->dest_time_, ping_j.start_time_)) {
        trace_pong("---- DROP dominated ping {} (pong best {} worse than dep)",
                   to_tuple(ping_j), match->dest_time_);
        ping_j.error_ = true;
        continue;
      }
      trace_pong("---- HIT [updating ping start time {} -> {}]\n",
                 ping_j.start_time_, match->dest_time_);
      ping_j.start_time_ = match->dest_time_;
    }
    q.flip_dir();

    utl::erase_if(ping_results, [](journey const& j) { return j.error_; });

    // NEXT
    if (ping_results.empty()) {
      // every ping journey at this start time was a dominated phantom-wait
      // artifact; advance minimally so the interval scan keeps progressing
      start_time += duration_t{kFwd ? 1 : -1};
      continue;
    }
    auto const first_it =
        utl::min_element(ping_results, [&](journey const& a, journey const& b) {
          return is_better(a.start_time_, b.start_time_);
        });
    auto const next = first_it->start_time_ + duration_t{kFwd ? 1 : -1};

    if (!is_better(start_time, next)) {
      throw utl::fail("no pong progress: start_time={}, next={}", start_time,
                      next);
    }

    trace_pong(
        "AFTER {} [next={}]:\n\t{}", start_time, next,
        fmt::join(s_state.results_.els_ | std::views::transform(to_tuple),
                  "\n\t"));

    start_time = next;
  }

  if (pong_stats && pong_exec_total != 0U) {
    auto repeated = 0U;
    for (auto const& [k, c] : pong_exec_keys) {
      repeated += c - 1U;
    }
    std::fprintf(stderr, "PONG_EXEC total=%u unique=%zu repeated=%u\n",
                 pong_exec_total, pong_exec_keys.size(), repeated);
  }

  utl::erase_if(s_state.results_, [&](journey const& j) {
    auto const erase = !j.is_reconstructed_ || !is_validated(j) ||
                       j.travel_time() >= fastest_direct ||
                       j.travel_time() >= q.max_travel_time_;
    if (erase) {
      trace_pong(
          "ERASE not_reconstructed={}, not_validated={}, "
          "slower_than_direct={}, slower_than_query_max_travel_time={} {}",
          j.legs_.empty(), !is_validated(j), j.travel_time() >= fastest_direct,
          j.travel_time() >= q.max_travel_time_, to_tuple(j));
    }
    return erase;
  });

  for (auto& x : s_state.results_) {
    std::swap(x.start_time_, x.dest_time_);
  }

  result.interval_ = {kFwd ? search_interval.from_ : start_time + duration_t{1},
                      kFwd ? start_time : search_interval.to_};
  result.algo_stats_ = (ping.get_stats() + pong.get_stats()).to_map();
  result.search_stats_.execute_time_ =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          (std::chrono::steady_clock::now() - processing_start_time));

  for (auto& j : s_state.results_) {
    auto const swap = [](location_idx_t const l) -> location_idx_t {
      switch (to_idx(l)) {
        case to_idx(get_special_station(special_station::kStart)):
          return get_special_station(special_station::kEnd);
        case to_idx(get_special_station(special_station::kEnd)):
          return get_special_station(special_station::kStart);
        default: return l;
      }
    };
    j.legs_.front().from_ = swap(j.legs_.front().from_);
    j.legs_.back().to_ = swap(j.legs_.back().to_);
  }

  auto const iv = result.interval_;
  enrich_with_slow_direct<SearchDir>(tt, rtt, q, iv, s_state.results_);

  utl::sort(s_state.results_, [](journey const& a, journey const& b) {
    return std::tuple{a.start_time_, a.transfers_, a.dest_time_,
                      a.criteria_cost_} <
           std::tuple{b.start_time_, b.transfers_, b.dest_time_,
                      b.criteria_cost_};
  });

  trace_pong("RESULT:\n\t{}",
             fmt::join(s_state.results_.els_ | std::views::transform(to_tuple),
                       "\n\t"));

  if constexpr (!kFwd) {
    return result;
  }

  if constexpr (Vias != 0U) {
    if (utl::any_of(q.via_stops_, [](via_stop const& v) {
          return v.stay_ == duration_t{0};
        })) {
      // Stay duration == 0 means via-stop doesn't require a transfer.
      // => The via stop could be "optimized away" by get_earliest_alternative!
      return result;
    }
  }

  for (auto& j : s_state.results_) {
    auto v = via_offset_t{0};
    for (auto const [transit_1, transfer_1, transit_2, transfer_2, transit_3] :
         utl::nwise<5>(j.legs_)) {
      if (!std::holds_alternative<journey::run_enter_exit>(transit_1.uses_) ||
          !std::holds_alternative<journey::run_enter_exit>(transit_2.uses_) ||
          !std::holds_alternative<journey::run_enter_exit>(transit_3.uses_)) {
        continue;
      }

      auto const& front = std::get<journey::run_enter_exit>(transit_1.uses_);
      auto const& back = std::get<journey::run_enter_exit>(transit_3.uses_);

      auto const front_r = rt::frun{tt, rtt, front.r_};
      auto const from = front_r[front.stop_range_.to_ - 1U];

      auto arr_time = from.time(event_type::kArr);
      if (v < q.via_stops_.size() &&
          matches(tt, location_match_mode::kEquivalent,
                  q.via_stops_[v].location_, from.get_location_idx())) {
        arr_time += q.via_stops_[v++].stay_;
      }

      auto const back_r = rt::frun{tt, rtt, back.r_};
      auto const to = back_r[back.stop_range_.from_];

      auto dep_time = to.time(event_type::kDep);
      if (v < q.via_stops_.size() &&
          matches(tt, location_match_mode::kEquivalent,
                  q.via_stops_[v].location_, to.get_location_idx())) {
        // do not increment v, via may be used in next iteration
        dep_time -= q.via_stops_[v].stay_;
      }

      auto const earlier =
          get_earliest_alternative(tt, rtt, q, from.get_location_idx(),
                                   to.get_location_idx(), arr_time, dep_time);

      if (earlier.has_value()) {
        transfer_1 = earlier->at(0);
        transit_2 = earlier->at(1);
        transfer_2 = earlier->at(2);
      }
    }
  }

  return result;
}

template <direction SearchDir, via_offset_t Vias, typename AlgoState>
routing_result pong_with_vias(timetable const& tt,
                              rt_timetable const* rtt,
                              search_state& s_state,
                              AlgoState& r_state,
                              query q,
                              std::optional<std::chrono::seconds> timeout) {
  if (rtt == nullptr) {
    return pong<SearchDir, false, Vias>(tt, rtt, s_state, r_state, std::move(q),
                                        timeout);
  } else {
    return pong<SearchDir, true, Vias>(tt, rtt, s_state, r_state, std::move(q),
                                       timeout);
  }
}

template <direction SearchDir, typename AlgoState>
routing_result pong_search_with_dir(
    timetable const& tt,
    rt_timetable const* rtt,
    search_state& s_state,
    AlgoState& r_state,
    query q,
    std::optional<std::chrono::seconds> timeout) {
  switch (q.via_stops_.size()) {
    case 0:
      return pong_with_vias<SearchDir, 0>(tt, rtt, s_state, r_state,
                                          std::move(q), timeout);
    case 1:
      return pong_with_vias<SearchDir, 1>(tt, rtt, s_state, r_state,
                                          std::move(q), timeout);
    case 2:
      return pong_with_vias<SearchDir, 2>(tt, rtt, s_state, r_state,
                                          std::move(q), timeout);
  }
  throw utl::fail("{} vias not supported (max={})", kMaxVias);
}

template <typename AlgoState>
routing_result pong_search(timetable const& tt,
                           rt_timetable const* rtt,
                           search_state& s_state,
                           AlgoState& r_state,
                           query q,
                           direction search_dir,
                           std::optional<std::chrono::seconds> timeout) {
  if (search_dir == direction::kForward) {
    return pong_search_with_dir<direction::kForward>(tt, rtt, s_state, r_state,
                                                     std::move(q), timeout);
  } else {
    return pong_search_with_dir<direction::kBackward>(tt, rtt, s_state, r_state,
                                                      std::move(q), timeout);
  }
}

template routing_result pong_search(timetable const&,
                                    rt_timetable const*,
                                    search_state&,
                                    raptor_state&,
                                    query,
                                    direction,
                                    std::optional<std::chrono::seconds>);

template routing_result pong_search(timetable const&,
                                    rt_timetable const*,
                                    search_state&,
                                    mcraptor_state&,
                                    query,
                                    direction,
                                    std::optional<std::chrono::seconds>);

template routing_result pong_search(timetable const&,
                                    rt_timetable const*,
                                    search_state&,
                                    mcraptor_cost_state&,
                                    query,
                                    direction,
                                    std::optional<std::chrono::seconds>);

#if defined(NIGIRI_CUDA)
template routing_result pong_search(timetable const&,
                                    rt_timetable const*,
                                    search_state&,
                                    gpu::gpu_raptor_state&,
                                    query,
                                    direction,
                                    std::optional<std::chrono::seconds>);

template routing_result pong_search(timetable const&,
                                    rt_timetable const*,
                                    search_state&,
                                    gpu::gpu_mcraptor_state&,
                                    query,
                                    direction,
                                    std::optional<std::chrono::seconds>);

template routing_result pong_search(timetable const&,
                                    rt_timetable const*,
                                    search_state&,
                                    gpu::gpu_mcraptor_cost_state&,
                                    query,
                                    direction,
                                    std::optional<std::chrono::seconds>);
#endif

}  // namespace nigiri::routing
