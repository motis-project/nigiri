#include "nigiri/routing/raptor/pong.h"

#include <ranges>
#include <type_traits>

#include "utl/helpers/algorithm.h"
#include "utl/sorted_diff.h"
#include "utl/timing.h"

#include "nigiri/location_match_mode.h"
#include "nigiri/routing/direct.h"
#include "nigiri/routing/gpu/raptor.h"
#include "nigiri/routing/get_earliest_transport.h"
#include "nigiri/routing/leg_alternatives.h"
#include "nigiri/routing/transfer_time_settings.h"
#include "nigiri/rt/frun.h"
#include "nigiri/types.h"

#define trace_pong(...)
// #define trace_pong fmt::println

namespace nigiri::routing {

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

  // ping searches in SearchDir, pong searches in the flipped direction; both
  // share the algo state. Select CPU raptor or gpu_raptor based on the state.
  using ping_algo_t = std::conditional_t<
      std::is_same_v<AlgoState, gpu::gpu_raptor_state>,
      gpu::gpu_raptor<SearchDir, Rt, Vias>,
      raptor<SearchDir, Rt, Vias, search_mode::kOneToOne>>;
  using pong_algo_t = std::conditional_t<
      std::is_same_v<AlgoState, gpu::gpu_raptor_state>,
      gpu::gpu_raptor<flip(SearchDir), Rt, Vias>,
      raptor<flip(SearchDir), Rt, Vias, search_mode::kOneToOne>>;

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
  auto ping_lb = std::vector<std::uint16_t>{};
  // The GPU searches the scheduled timetable only (rtt=nullptr in the kernel).
  // The scheduled lb is the TIGHTEST admissible bound for a scheduled search.
  // (The rt-aware bound would also be admissible -- nigiri's rt lb_graph only
  // ADDS edges that are faster than scheduled and the dijkstra keeps the min, so
  // rt-aware lb <= scheduled lb, never longer -- but it's weaker pruning, so we
  // use the scheduled one for the GPU.)
  constexpr auto const kGpu = std::is_same_v<AlgoState, gpu::gpu_raptor_state>;
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

  auto ping_dist_to_dest = std::vector<std::uint16_t>{};
  auto ping_is_dest = bitvec{};
  auto ping_is_via = std::array<bitvec, kMaxVias>{};
  collect_destinations(tt, q.destination_, q.dest_match_mode_, ping_is_dest,
                       ping_dist_to_dest);
  for (auto const [i, via] : utl::enumerate(q.via_stops_)) {
    collect_via_destinations(tt, via.location_, ping_is_via[i]);
  }

  auto ping = ping_algo_t{
      tt,
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

  // ====
  // PONG
  // ----
  q.flip_dir();

  auto pong_lb = std::vector<std::uint16_t>{};
  // scheduled lower bound for the GPU (see ping_lb above)
  dijkstra(tt, q,
           (kFwd ? tt.bwd_search_lb_graph_[q.prf_idx_]
                 : tt.fwd_search_lb_graph_[q.prf_idx_]),
           ((rtt == nullptr || kGpu)
                ? nullptr
                : &(kFwd ? rtt->bwd_search_lb_graph_has_edges_
                         : rtt->fwd_search_lb_graph_has_edges_)),
           ((rtt == nullptr || kGpu) ? nullptr
                                     : &(kFwd ? rtt->bwd_search_lb_graph_
                                              : rtt->fwd_search_lb_graph_)),
           pong_lb);

  auto pong_dist_to_dest = std::vector<std::uint16_t>{};
  auto pong_is_dest = bitvec{};
  collect_destinations(tt, q.destination_, q.dest_match_mode_, pong_is_dest,
                       pong_dist_to_dest);

  auto pong_is_via = std::array<bitvec, kMaxVias>{};
  for (auto const [i, via] : utl::enumerate(q.via_stops_)) {
    collect_via_destinations(tt, via.location_, pong_is_via[i]);
  }

  auto pong = pong_algo_t{
      tt,
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

  q.flip_dir();

  // ========
  // >> PLAY!
  // --------
  auto starts = std::vector<start>{};
  auto result = routing_result{
      .journeys_ = &s_state.results_,
      .interval_ = search_interval,
      .search_stats_ = {.lb_time_ = 0U},  // lower bounds disabled
      .algo_stats_ = {}};
  auto start_time =
      kFwd ? search_interval.from_ : search_interval.to_ - duration_t{1};
  auto const end_time =
      kFwd ? search_interval.to_ : search_interval.from_ - duration_t{1};
  auto const is_better = [](auto a, auto b) { return kFwd ? a < b : a > b; };
  auto const is_validated = [&](journey const& j) {
    return is_better(j.dest_time_, start_time);
  };
  auto const get_result_count = [&](bool const include_too_slow) {
    return utl::count_if(*result.journeys_, [&](journey const& j) {
      return is_validated(j) &&
             (include_too_slow || (j.travel_time() < fastest_direct &&
                                   j.travel_time() < q.max_travel_time_));
    });
  };
  auto const is_timeout_reached = [&]() {
    if (timeout) {
      return (std::chrono::steady_clock::now() - processing_start_time) >=
             *timeout;
    }
    return false;
  };
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
    utl::sort(ping_results, [](journey const& a, journey const& b) {
      return a.transfers_ > b.transfers_;
    });

    // ----
    // PONG
    // ----
    q.flip_dir();
    pong.reset_arrivals();
    for (auto& ping_j : ping_results) {
      trace_pong("-- PING RESULT: {}", to_tuple(ping_j));

      starts.clear();
      get_starts(flip(SearchDir), tt, rtt, ping_j.dest_time_, q.start_,
                 q.td_start_, q.via_stops_, q.max_start_offset_,
                 q.start_match_mode_,
                 q.start_match_mode_ != location_match_mode::kIntermodal,
                 starts, false, q.prf_idx_, q.transfer_time_settings_);
      pong.next_start_time();
      for (auto const& s : starts) {
        trace_pong("---- PONG START: {} at time_at_start={} time_at_stop={}",
                   loc{tt, s.stop_}, s.time_at_start_, s.time_at_stop_);
        pong.add_start(s.stop_, s.time_at_stop_);
      }
      pong.execute(ping_j.dest_time_, ping_j.transfers_,
                   ping_j.start_time_ - duration_t{kFwd ? 1 : -1}, q.prf_idx_,
                   s_state.results_);
      kFwd ? ++result.search_stats_.n_execute_bwd_
           : ++result.search_stats_.n_execute_fwd_;

      auto const match =
          utl::find_if(s_state.results_, [&](journey const& pong_j) {
            return pong_j.transfers_ == ping_j.transfers_ &&
                   pong_j.start_time_ == ping_j.dest_time_;
          });

      if (match == end(s_state.results_)) {
        throw utl::fail(
            "no pong for transfers={}, start_time={} found, journeys={}",
            ping_j.transfers_, ping_j.dest_time_,
            s_state.results_.els_ | std::views::transform(to_tuple));
      }

      trace_pong("---- HIT [updating ping start time {} -> {}]\n",
                 ping_j.start_time_, match->dest_time_);
      if (!match->is_reconstructed_ && !match->error_) {
        pong.reconstruct(q, *match);
      }
      ping_j.start_time_ = match->dest_time_;
    }
    q.flip_dir();

    // NEXT
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
    return std::tuple{a.start_time_, a.transfers_} <
           std::tuple{b.start_time_, b.transfers_};
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

#if defined(NIGIRI_CUDA)
template routing_result pong_search(timetable const&,
                                    rt_timetable const*,
                                    search_state&,
                                    gpu::gpu_raptor_state&,
                                    query,
                                    direction,
                                    std::optional<std::chrono::seconds>);
#endif

}  // namespace nigiri::routing
