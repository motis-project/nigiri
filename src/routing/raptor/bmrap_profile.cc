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
#include "nigiri/routing/raptor/pong.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/special_stations.h"

namespace nigiri::routing {

namespace {

// Keeps a label at l iff a label of the other search at l is catchable from it
// (stored post-transfer: one buffer of slack) and the two join into a target.
template <direction SearchDir, typename Criteria>
struct join_oracle final : mc_oracle<Criteria> {
  join_oracle(timetable const& tt,
              transfer_time_settings const& settings,
              mc_label_snapshot<Criteria> const& labels,
              std::vector<journey const*> targets)
      : tt_{&tt},
        settings_{&settings},
        labels_{&labels},
        targets_{std::move(targets)} {}

  bool feasible(std::uint32_t const l,
                unsigned const k,
                delta_t const t,
                Criteria const& label) const override {
    auto const slack = adjusted_transfer_time(
        *settings_, tt_->locations_.transfer_time_[location_idx_t{l}].count());
    auto const at = static_cast<int>(t);
    for (auto const& e : labels_->at(l)) {
      auto const other_at = static_cast<int>(e.crit_.arr_);
      if (SearchDir == direction::kForward ? at > other_at + slack
                                           : at < other_at - slack) {
        continue;  // not catchable
      }
      for (auto const* j : targets_) {
        if (k + e.round_ <= j->transfers_ + 1U &&
            label.join_fits(e.crit_, *j)) {
          return true;
        }
      }
    }
    return false;
  }

  timetable const* tt_;
  transfer_time_settings const* settings_;
  mc_label_snapshot<Criteria> const* labels_;
  std::vector<journey const*> targets_;
};

using namespace bmrap_detail;

bool same_criteria(journey const& a, journey const& b) {
  return a.transfers_ == b.transfers_ && a.criteria_cost_ == b.criteria_cost_ &&
         a.criteria_mode_filter_ == b.criteria_mode_filter_ &&
         a.criteria_mode_switches_ == b.criteria_mode_switches_;
}
using clk = std::chrono::steady_clock;

struct stopwatch {
  void lap(clk::duration& acc) {
    auto const now = clk::now();
    acc += now - t_;
    t_ = now;
  }
  clk::time_point t_{clk::now()};
};

// fn(arrival, first, last) per run of equal dest_time_ (js sorted by it)
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
    basic_mcraptor_state<Criteria>& cpu_mc_state,
    query q,
    std::optional<std::chrono::seconds> const timeout) {
  constexpr auto const kFwd = (SearchDir == direction::kForward);
  static constexpr auto const kDir = duration_t{kFwd ? 1 : -1};
  using ping_t = typename bmrap_algo_for<SearchDir, Rt, AlgoState>::type;
  using pong_t = typename bmrap_algo_for<flip(SearchDir), Rt, AlgoState>::type;
  using mc_ping_for = bmrap_mc_algo_for<SearchDir, Criteria, GpuMc>;
  using mc_pong_for = bmrap_mc_algo_for<flip(SearchDir), Criteria, GpuMc>;
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

  auto fwd_lb = std::vector<std::uint16_t>{};
  auto bwd_lb = std::vector<std::uint16_t>{};
  auto lb_time = clk::duration{};
  auto sw = stopwatch{};
  if constexpr (ping_t::kUseLowerBounds || bounded_needs_lb<mc_ping_t>()) {
    dijkstra(tt, nullptr, q, SearchDir, fwd_lb);
  }
  if constexpr (bounded_needs_lb<pong_t>() || bounded_needs_lb<mc_pong_t>()) {
    dijkstra(tt, nullptr, qf, flip(SearchDir), bwd_lb);
  }
  sw.lap(lb_time);
  auto ms_setup = clk::duration{};

  auto is_via = std::array<bitvec, kMaxVias>{};
  auto no_via = std::vector<via_stop>{};

  // One scalar state serves ping, pong and slacked pong, one mc state all mc
  // phases: they run in sequence, and a phase guided by another's labels reads
  // a copy of them.
  static_assert(
      std::is_same_v<typename mc_ping_for::state, typename mc_pong_for::state>);
  auto& mc_state = [&]() -> typename mc_ping_for::state& {
    if constexpr (GpuMc) {
      return r_state.mc_state();
    } else {
      return cpu_mc_state;
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
      mc_state, fwd_is_dest, fwd_dist, q.td_dest_, fwd_lb);
  auto mc_pong = make.template operator()<mc_pong_t>(
      mc_state, bwd_is_dest, bwd_dist, qf.td_dest_, bwd_lb);
  // the starts' walks, as search.h tells the plain engines: td offsets may
  // wait, locations of a start stop are reached by a transfer
  mc_ping.set_td_start(q.td_start_);
  mc_ping.set_start_stops(q.start_, q.start_match_mode_);
  mc_pong.set_td_start(qf.td_start_);
  mc_pong.set_start_stops(qf.start_, qf.start_match_mode_);

  if constexpr (GpuMc) {
    // phase 5 runs several departures under one reset_arrivals(): the device
    // reuse frontier must not carry over between them, as on the CPU
    mc_pong.set_reuse_same_dep();
  }
  sw.lap(ms_setup);

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
  // Bounds the arrival relative to the step, so it looks past max_travel_time
  // (the next useful departure can be hours away); travel time is filtered on
  // the results.
  auto const worst_at_dest = [&](unixtime_t const t) {
    return t + (kFwd ? 1 : -1) *
                   std::min(q.max_travel_time_ + kMinLookAhead, kMaxTravelTime);
  };
  // the arrival slack on the ping's target pruning makes its round times a
  // valid tau_arr^->(v, i) matrix (paper, Sec. 4.3); clamped as relax_arr()
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
    ping.execute(t, q.max_transfers_, all_rounds(worst_at_dest(t)), results);
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
    auto const ar = run_anchor_search<SearchDir>(tt, rtt, range_s_state,
                                                 r_state, q, timeout);
    scan_interval = ar.interval_;
    for (auto const& j : *ar.journeys_) {
      all_anchors.push_back({j.start_time_, j.dest_time_,
                             static_cast<std::uint8_t>(j.transfers_ + 1U)});
    }
    // a windowed anchor set is truncated at its far end
    if (!all_anchors.empty()) {
      close_anchor_profile<SearchDir>(tt, rtt, q, scan_interval, r_state,
                                      all_anchors);
    }
    sw.lap(ms_range);
  }

  auto start_time =
      kFwd ? scan_interval.from_ : scan_interval.to_ - duration_t{1};
  auto const end_time =
      kFwd ? scan_interval.to_ : scan_interval.from_ - duration_t{1};

  auto bounds = bmrap_bounds{};

  // Guided searches keep only the labels that join one of the other
  // direction's labels (copied right after it ran: mc_label_snapshot, on the
  // device take_guide()) into one of the targets. Complete as long as the mc
  // phases project with a zero lower bound: destination pruning then never
  // cuts a label that can still complete to a target.
  static_assert(!bounded_needs_lb<mc_ping_t>() &&
                !bounded_needs_lb<mc_pong_t>());
  using labels_t =
      std::conditional_t<GpuMc, std::monostate, mc_label_snapshot<Criteria>>;
  auto ping_labels = labels_t{};
  auto pong_labels = labels_t{};
  auto const guided = [&]<direction Dir>(
                          auto& engine, [[maybe_unused]] labels_t const& labels,
                          std::vector<journey const*> targets, auto&& run) {
    if constexpr (GpuMc) {
      engine.set_guide(targets);
      run();
      engine.set_guide({});
    } else {
      auto const oracle = join_oracle<Dir, Criteria>{
          tt, q.transfer_time_settings_, labels, std::move(targets)};
      engine.set_oracle(&oracle);
      run();
      engine.set_oracle(nullptr);
    }
  };

  // ---- 5b. REALIZE ----
  // The mc pong's legs are the latest connections (no slack at any transfer).
  // Forward queries rerun each departure forward, guided by the mc pong's
  // labels, and splice in the earliest connections; for arriveBy the mc pong
  // runs forward and already has them.
  auto realized = std::vector<journey>{};
  auto n_realized = std::uint64_t{0U};
  auto n_unrealized = std::uint64_t{0U};
  auto max_guide_labels = std::size_t{0U};
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
      guided.template operator()<SearchDir>(
          mc_ping, pong_labels, std::move(targets),
          [&]() { run_mc_ping(d, max_tr, arr + kDir, fwd); });
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
  auto n_mc_pong_skipped = std::uint64_t{0U};

  auto anchors = std::vector<anchor>{};
  auto max_trips = std::uint8_t{0U};
  auto budget = std::uint8_t{0U};
  auto anchors_valid_until = std::optional<unixtime_t>{};
  auto n_anchor_recomputes = std::uint64_t{0U};
  auto exit_reason = std::uint64_t{0U};  // 0=cond 2=anchors 3=stall

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
         tt.external_interval().contains(start_time) && !is_timeout()) {
    // The anchors (and phases 1-3) only go stale once the step passes the
    // earliest anchor departure; steps in between skip them.
    auto const anchors_stale = !anchors_valid_until.has_value() ||
                               is_better(*anchors_valid_until, start_time);
    if (anchors_stale) {
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
        ++n_fwd_bound_builds;
        mc_pong.set_bounds(&fwd_bounds);
      } else {
        // ---- 1. PING ----
        sw = stopwatch{};
        auto ping_results = run_ping(start_time);
        sw.lap(ms_ping);
        utl::sort(ping_results, [&](journey const& a, journey const& b) {
          return is_better(a.dest_time_, b.dest_time_);
        });

        // before the pong overwrites the round times; sized by the ping's
        // trips (the pong moves departures, not trip counts)
        auto ping_trips = std::uint8_t{0U};
        for (auto const& j : ping_results) {
          ping_trips = std::max(ping_trips,
                                static_cast<std::uint8_t>(j.transfers_ + 1U));
        }
        fwd_bounds = reach_matrix<SearchDir>(
            tt, q, r_state, ping, trip_budget(ping_trips, budget_cap),
            /*sub_transfer=*/true, /*host_copy=*/!GpuMc);
        sw.lap(ms_fwd_bounds);
        ++n_fwd_bound_builds;
        mc_pong.set_bounds(&fwd_bounds);
        pong.set_bounds(&fwd_bounds);

        // ---- 2. PONG ----
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
          pong.execute(arr, max_tr, all_rounds(loosest - kDir), tight);
        });
        sw.lap(ms_pong);

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
        // A later step can re-anchor a journey to a departure an earlier step
        // covered with a better arrival: keep all_anchors a (departure,
        // arrival, trips) Pareto set, a dominated anchor must not be an A(J).
        auto const dominates = [&](anchor const& x, anchor const& y) {
          return !is_better(x.anchored_, y.anchored_) &&
                 !is_better(y.found_, x.found_) && x.trips_ <= y.trips_;
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

      // ---- 3. SLACKED PONG ----
      sw = stopwatch{};
      bounds = compute_bounds<SearchDir, Rt>(
          tt, rtt, q, anchors, base_day, /*horizon=*/start_time - kDir, budget,
          r_state, prune_stats, &fwd_bounds, /*host_copy=*/!GpuMc);
      sw.lap(ms_bounds);
      ++n_bound_builds;
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
    if constexpr (GpuMc) {
      mc_ping.take_guide();
    } else {
      ping_labels.take(mc_state);
    }
    sw.lap(ms_mc);

    // ---- 5. MC PONG ----
    // An mc journey an earlier result dominates is still re-anchored: its
    // latest departure can lie after the dominating one, where no later step
    // finds it. An arrival group is skipped when all its journeys already are
    // re-anchored in the results (the mc ping finds most of the previous
    // step's again). Results have the scan's convention, mc journeys not.
    utl::sort(mc_results, [&](journey const& a, journey const& b) {
      return is_better(a.dest_time_, b.dest_time_);
    });
    auto const re_anchored = [&](journey const& x) {
      return utl::any_of(s_state.results_, [&](journey const& y) {
        return y.start_time_ == x.dest_time_ &&
               !is_better(y.dest_time_, x.start_time_) && same_criteria(x, y);
      });
    };

    mc_pong.reset_arrivals();
    auto step_results = pareto_set<journey>{};
    auto skipped_deps = std::vector<unixtime_t>{};
    for_each_arrival_group(mc_results, [&](unixtime_t const arr,
                                           auto const first, auto const last) {
      auto const group_tr =
          std::max_element(first, last, [](journey const& a, journey const& b) {
            return a.transfers_ < b.transfers_;
          })->transfers_;
      if (std::all_of(first, last, re_anchored)) {
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
      for (auto it = first; it != last; ++it) {
        targets.push_back(&*it);
      }
      seed_bwd(mc_pong, arr);
      guided.template operator()<flip(SearchDir)>(
          mc_pong, ping_labels, std::move(targets), [&]() {
            mc_pong.execute(arr, group_tr, all_rounds(start_time - kDir),
                            step_results);
          });
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
        if constexpr (GpuMc) {
          mc_pong.take_guide();
        } else {
          pong_labels.take(mc_state);
          max_guide_labels =
              std::max(max_guide_labels, pong_labels.labels_.size());
        }
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
    sw.lap(ms_recon);

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
      exit_reason = 3U;
      break;
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
  algo_stats["bmrapp_ms_setup"] = ms(ms_setup);
  algo_stats["bmrapp_max_guide_labels"] = max_guide_labels;
  algo_stats["bmrapp_aligned"] = aligned ? 1U : 0U;
  algo_stats["bmrapp_ms_range"] = ms(ms_range);
  algo_stats["bmrapp_mc_pong_skipped"] = n_mc_pong_skipped;

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
    basic_mcraptor_state<Criteria>& mc_state,
    query q,
    direction const search_dir,
    std::optional<std::chrono::seconds> const timeout,
    bool const gpu_mc) {
  // A fixed start time has no profile to restrict (and no window to scan):
  // the mcraptor Pareto set of that departure.
  if (std::holds_alternative<unixtime_t>(q.start_time_)) {
    return raptor_search(tt, rtt, s_state, mc_state, std::move(q), search_dir,
                         timeout);
  }
  auto const run = [&]<bool GpuMc>() {
    return search_dir == direction::kForward
               ? bmrap_profile<direction::kForward, false, Criteria, AlgoState,
                               GpuMc>(tt, rtt, s_state, algo_state, mc_state,
                                      std::move(q), timeout)
               : bmrap_profile<direction::kBackward, false, Criteria, AlgoState,
                               GpuMc>(tt, rtt, s_state, algo_state, mc_state,
                                      std::move(q), timeout);
  };
  // the device mc state is a large allocation that can fail: fall back to the
  // CPU mc phases then
  if constexpr (kGpuMcSupported<Criteria, AlgoState>) {
    if (gpu_mc && algo_state.try_mc_state() != nullptr) {
      return run.template operator()<true>();
    }
  }
  return run.template operator()<false>();
}

#define NIGIRI_BMRAPP_INSTANTIATE(C, S)                         \
  template routing_result bmrap_profile_search<C>(              \
      timetable const&, rt_timetable const*, search_state&, S&, \
      basic_mcraptor_state<C>&, query, direction,               \
      std::optional<std::chrono::seconds>, bool);

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
