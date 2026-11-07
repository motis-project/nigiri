#include "nigiri/routing/raptor/pong.h"

#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <limits>
#include <ranges>
#include <span>
#include <string_view>
#include <type_traits>

#include "utl/helpers/algorithm.h"
#include "utl/sorted_diff.h"
#include "utl/timing.h"

#include "nigiri/location_match_mode.h"
#include "nigiri/routing/direct.h"
#include "nigiri/routing/get_earliest_transport.h"
#include "nigiri/routing/gpu/raptor.h"
#include "nigiri/routing/leg_alternatives.h"
#include "nigiri/routing/search_interval.h"
#include "nigiri/routing/transfer_time_settings.h"
#include "nigiri/rt/frun.h"
#include "nigiri/types.h"

#define trace_pong(...)
// #define trace_pong fmt::println

namespace nigiri::routing {

interval<unixtime_t> pong_search_interval(direction const search_dir,
                                          timetable const& tt,
                                          query const& q,
                                          duration_t const min_look_ahead) {
  auto const query_iv = start_time_interval(q.start_time_);
  auto const tt_iv = tt.external_interval();
  auto const max_search_iv = search_dir == direction::kForward
                                 ? interval{query_iv.from_, tt_iv.to_}
                                 : interval{tt_iv.from_, query_iv.to_};

  return reachable_events(
      search_dir, max_search_iv,
      std::min(q.max_travel_time_ + min_look_ahead, kMaxTravelTime));
}

constexpr auto const kPruneWithPingBounds = true;

// Pong traverses huge parts of the timetable to collect the requested number
// of journeys. Journeys departing (kFwd) / arriving (!kFwd) weeks or months
// after / before the queried start time are most likely not interesting for
// the user, so the search is capped at this distance from the query start
// time. Six days (instead of a full week) prevents a query on a Monday from
// reporting journeys on the following / previous Monday.
constexpr auto const kMaxLookAhead = 6_days;

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
  // the generator anchors the boarding walk at the transit departure
  // (latest start) -> shift to the interior transfer convention:
  // the walk starts at the previous leg's arrival.
  // A time-dependent footpath may only be usable later (e.g. an elevator out
  // of service): keep the wait before it, as in the reconstruction.
  auto const is_td = rtt != nullptr && has_profile_td_footpaths(q.prf_idx_) &&
                     rtt->has_td_footpaths_out_[q.prf_idx_].test(legs[0].from_);
  if (!is_td) {
    auto const walk_duration = legs[0].arr_time_ - legs[0].dep_time_;
    legs[0].dep_time_ = from_arr;
    legs[0].arr_time_ = from_arr + walk_duration;
  }
  return std::array{std::move(legs[0]), std::move(legs[1]), std::move(legs[2])};
}

void swap_special_stations(journey& j) {
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

template <direction SearchDir, via_offset_t Vias>
void finish_journeys(timetable const& tt,
                     rt_timetable const* rtt,
                     query const& q,
                     interval<unixtime_t> const iv,
                     pareto_set<journey>& results) {
  enrich_with_slow_direct<SearchDir>(tt, rtt, q, iv, results);

  utl::sort(results, [](journey const& a, journey const& b) {
    return std::tuple{a.start_time_, a.transfers_} <
           std::tuple{b.start_time_, b.transfers_};
  });

  trace_pong("RESULT:\n\t{}",
             fmt::join(results.els_ | std::views::transform(to_tuple), "\n\t"));

  if constexpr (SearchDir == direction::kBackward || Vias != 0U) {
    // via requirements (order, stays) are invisible to
    // get_earliest_alternative
    return;
  }

  for (auto& j : results) {
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

      auto const arr_time = from.time(event_type::kArr);

      auto const back_r = rt::frun{tt, rtt, back.r_};
      auto const to = back_r[back.stop_range_.from_];

      auto const dep_time = to.time(event_type::kDep);

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
}

template <direction SearchDir, bool Rt, via_offset_t Vias, typename AlgoState>
routing_result pong(timetable const& tt,
                    rt_timetable const* rtt,
                    search_state& s_state,
                    AlgoState& r_state,
                    query q,
                    std::optional<std::chrono::seconds> timeout) {
  constexpr auto kFwd = (SearchDir == direction::kForward);

  using ping_algo_t =
      std::conditional_t<std::is_same_v<AlgoState, gpu::gpu_raptor_state>,
                         gpu::gpu_raptor<SearchDir, false>,
                         raptor<SearchDir, Rt, Vias, search_mode::kOneToOne>>;
  using pong_algo_t = std::conditional_t<
      std::is_same_v<AlgoState, gpu::gpu_raptor_state>,
      gpu::gpu_raptor<flip(SearchDir), kPruneWithPingBounds>,
      raptor<flip(SearchDir), Rt, Vias, search_mode::kOneToOne>>;

  s_state.results_.clear();

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

  auto const lookahead_end = kFwd ? search_interval.from_ + kMaxLookAhead
                                  : search_interval.to_ - kMaxLookAhead;

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
  auto const ping_lb_start = std::chrono::steady_clock::now();
  auto ping_lb = std::vector<std::uint16_t>{};
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
                          q.no_compulsory_reservation_,
                          q.transfer_time_settings_,
                          q.prf_idx_};

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

  auto const pong_lb_start = std::chrono::steady_clock::now();
  auto pong_lb = std::vector<std::uint16_t>{};
  if constexpr (pong_algo_t::kUseLowerBounds) {
    if (kPruneWithPingBounds) {
      pong_lb.assign(tt.n_locations(), 0U);
    } else {
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
    }
  }
  lb_time += std::chrono::steady_clock::now() - pong_lb_start;

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
                          q.no_compulsory_reservation_,
                          q.transfer_time_settings_,
                          q.prf_idx_};

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
  auto const get_result_count = [&](bool const include_too_slow) {
    return utl::count_if(*result.journeys_, [&](journey const& j) {
      return is_validated(j) &&
             (include_too_slow || (j.travel_time() < fastest_direct &&
                                   j.travel_time() <= q.max_travel_time_));
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
         tt.external_interval().contains(start_time) &&
         is_better(start_time, lookahead_end) && !is_timeout_reached()) {
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
      ping.add_start(s.stop_, s.time_at_stop_, s.v_);
    }
    auto worst_time_at_dest = std::array<unixtime_t, kMaxTransfers + 2U>{};
    worst_time_at_dest.fill(
        start_time +
        (kFwd ? 1 : -1) *
            std::min(q.max_travel_time_ + kMinLookAhead, kMaxTravelTime));
    for (auto const& j : s_state.results_) {
      if (!j.error_ && !is_better(j.dest_time_, start_time)) {
        for (auto k = j.transfers_ + 1U; k != worst_time_at_dest.size(); ++k) {
          if (is_better(j.start_time_, worst_time_at_dest[k])) {
            worst_time_at_dest[k] = j.start_time_;
          }
        }
      }
    }
    auto ping_results = pareto_set<journey>{};
    ping.execute(start_time, q.max_transfers_, worst_time_at_dest,
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
    utl::sort(ping_results, [](journey const& a, journey const& b) {
      return a.transfers_ > b.transfers_;
    });
    if (!result.cursor_bounds_.has_value()) {
      auto bounds = std::array<unixtime_t, kMaxTransfers + 2U>{};
      bounds.fill(kFwd ? unixtime_t::max() : unixtime_t::min());
      for (auto const& j : ping_results) {
        for (auto k = j.transfers_ + 1U; k != bounds.size(); ++k) {
          if (is_better(j.dest_time_, bounds[k])) {
            bounds[k] = j.dest_time_;
          }
        }
      }
      result.cursor_bounds_ = bounds;
    }

    // ----
    // PONG
    // ----
    if constexpr (kPruneWithPingBounds) {
      // Has to happen before pong.reset_arrivals() wipes the shared
      // round_times the ping search just filled.
      ping.fill_bounds(ping_results.begin()->transfers_ + std::size_t{1U});
    }
    q.flip_dir();
    pong.reset_arrivals();
    auto const run_pong = [&](auto& po, journey& ping_j) {
      starts.clear();
      get_starts(flip(SearchDir), tt, rtt, ping_j.dest_time_, q.start_,
                 q.td_start_, q.via_stops_, q.max_start_offset_,
                 q.start_match_mode_,
                 q.start_match_mode_ != location_match_mode::kIntermodal,
                 starts, false, q.prf_idx_, q.transfer_time_settings_);
      po.next_start_time();
      for (auto const& s : starts) {
        trace_pong("---- PONG START: {} at time_at_start={} time_at_stop={}",
                   loc{tt, s.stop_}, s.time_at_start_, s.time_at_stop_);
        po.add_start(s.stop_, s.time_at_stop_, s.v_);
      }
      auto pong_worst_time_at_dest =
          std::array<unixtime_t, kMaxTransfers + 2U>{};
      pong_worst_time_at_dest.fill(ping_j.start_time_);
      po.execute(ping_j.dest_time_, ping_j.transfers_, pong_worst_time_at_dest,
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
        po.reconstruct(q, *match);
      }
      ping_j.start_time_ = match->dest_time_;
    };
    for (auto& ping_j : ping_results) {
      trace_pong("-- PING RESULT: {}", to_tuple(ping_j));

      // A previous pong already found the latest departure for this arrival
      // time and number of transfers: the ping found the same journey again,
      // so the pong would, too.
      auto const known =
          utl::find_if(s_state.results_, [&](journey const& pong_j) {
            return pong_j.transfers_ == ping_j.transfers_ &&
                   pong_j.start_time_ == ping_j.dest_time_;
          });
      if (known != end(s_state.results_)) {
        trace_pong("---- KNOWN [updating ping start time {} -> {}]\n",
                   ping_j.start_time_, known->dest_time_);
        ping_j.start_time_ = known->dest_time_;
        continue;
      }

      if constexpr (kPruneWithPingBounds) {
        pong.set_bounds(ping_j.transfers_ + 1U);
      }
      run_pong(pong, ping_j);
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
    auto const j_start_time = j.dest_time_;
    auto const is_out_of_interval =
        kFwd ? !q.extend_interval_later_ && j_start_time >= search_interval.to_
             : !q.extend_interval_earlier_ &&
                   j_start_time < search_interval.from_;
    auto const is_out_of_lookahead = !is_better(j_start_time, lookahead_end);
    auto const erase = !j.is_reconstructed_ || !is_validated(j) ||
                       is_out_of_interval || is_out_of_lookahead ||
                       j.travel_time() >= fastest_direct ||
                       j.travel_time() > q.max_travel_time_;
    if (erase) {
      trace_pong(
          "ERASE not_reconstructed={}, not_validated={}, "
          "slower_than_direct={}, slower_than_query_max_travel_time={}, "
          "beyond_max_lookahead={} {}",
          j.legs_.empty(), !is_validated(j), j.travel_time() >= fastest_direct,
          j.travel_time() > q.max_travel_time_, is_out_of_lookahead,
          to_tuple(j));
    }
    return erase;
  });

  for (auto& x : s_state.results_) {
    std::swap(x.start_time_, x.dest_time_);
  }

  auto const searched_until =
      is_better(lookahead_end, start_time) ? lookahead_end : start_time;
  result.interval_ = {
      kFwd ? search_interval.from_ : searched_until + duration_t{1},
      kFwd ? searched_until : search_interval.to_};
  result.algo_stats_ = (ping.get_stats() + pong.get_stats()).to_map();
  result.search_stats_.execute_time_ =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          (std::chrono::steady_clock::now() - processing_start_time));

  for (auto& j : s_state.results_) {
    swap_special_stations(j);
  }

  finish_journeys<SearchDir, Vias>(tt, rtt, q, result.interval_,
                                   s_state.results_);
  return result;
}

// Paging against the search direction: fwd + extend earlier (the page before
// a page that starts at T), bwd + extend later. The page holds every journey
// departing in it that is pareto-optimal among all journeys departing after
// it (the standard pong's result semantics). The page is enumerated from the
// arrival side (reverse ping = latest departures arriving before a deadline,
// forward pong = their exact arrival): first until it holds enough journeys,
// then what arrives at or after the adjacent page's earliest arrival, per
// transfer count below the adjacent page's arrival for that count, and finally
// the rest, all bounded by the page start that follows the journeys found.
//
// Cursor: q.cursor_bounds_ = per round the earliest arrival (fwd) / latest
// departure (bwd) of the adjacent page's first ping: every journey arriving
// before the best of them departs before T, and a journey with k transfers
// arriving at or after the bound of round k+1 is dominated by the adjacent
// page. Computed with one ping at T if not given.
template <direction SearchDir, bool Rt, via_offset_t Vias, typename AlgoState>
routing_result pong_reverse(timetable const& tt,
                            rt_timetable const* rtt,
                            search_state& s_state,
                            AlgoState& r_state,
                            query q,
                            std::optional<std::chrono::seconds> timeout) {
  constexpr auto const kFwd = (SearchDir == direction::kForward);
  constexpr auto const kRev = flip(SearchDir);
  constexpr auto const kGpu = std::is_same_v<AlgoState, gpu::gpu_raptor_state>;

  using ping_t =
      std::conditional_t<kGpu, gpu::gpu_raptor<SearchDir, false>,
                         raptor<SearchDir, Rt, Vias, search_mode::kOneToOne>>;
  using rev_ping_t =
      std::conditional_t<kGpu, gpu::gpu_raptor<kRev, false>,
                         raptor<kRev, Rt, Vias, search_mode::kOneToOne>>;
  using rev_pong_t =
      std::conditional_t<kGpu, gpu::gpu_raptor<SearchDir, kPruneWithPingBounds>,
                         raptor<SearchDir, Rt, Vias, search_mode::kOneToOne>>;

  s_state.results_.clear();

  auto const processing_start_time = std::chrono::steady_clock::now();
  auto const is_timeout_reached = [&]() {
    return timeout.has_value() && (std::chrono::steady_clock::now() -
                                   processing_start_time) >= *timeout;
  };

  auto const fastest_direct = get_fastest_direct(tt, q, SearchDir);
  auto const search_interval = start_time_interval(q.start_time_);
  auto const base_day =
      day_idx_t{std::chrono::duration_cast<date::days>(
                    std::chrono::round<std::chrono::days>(
                        search_interval.from_ +
                        ((search_interval.to_ - search_interval.from_) / 2)) -
                    tt.internal_interval().from_)
                    .count()};
  auto const is_better = [](auto a, auto b) { return kFwd ? a < b : a > b; };
  auto const dir = [](duration_t const d) { return kFwd ? d : -d; };
  auto const step = dir(duration_t{1});
  auto const max_look =
      dir(std::min(q.max_travel_time_ + kMinLookAhead, kMaxTravelTime));

  // t0: the last minute of this page next to the cursor T
  auto const t0 = kFwd ? search_interval.to_ - step : search_interval.from_;
  auto const target =
      kFwd ? search_interval.from_ : search_interval.to_ - duration_t{1};
  auto const lookahead_end = t0 - dir(kMaxLookAhead);

  auto qr = q;
  qr.flip_dir();

  struct side {
    std::vector<std::uint16_t> dist_to_dest_;
    bitvec is_dest_;
    std::array<bitvec, kMaxVias> is_via_;
  };
  auto const make_side = [&](query const& x) {
    auto s = side{};
    collect_destinations(tt, x.destination_, x.dest_match_mode_, s.is_dest_,
                         s.dist_to_dest_);
    for (auto const [i, via] : utl::enumerate(x.via_stops_)) {
      collect_via_destinations(tt, via.location_, s.is_via_[i]);
    }
    return s;
  };
  auto fwd_side = make_side(q);
  auto rev_side = make_side(qr);

  auto lb_time = std::chrono::steady_clock::duration{};
  auto const compute_lb = [&]<direction Dir>(query const& x,
                                             std::vector<std::uint16_t>& lb) {
    constexpr auto const kDirFwd = Dir == direction::kForward;
    auto const start = std::chrono::steady_clock::now();
    dijkstra(
        tt, x,
        (kDirFwd ? tt.fwd_search_lb_graph_[x.prf_idx_]
                 : tt.bwd_search_lb_graph_[x.prf_idx_]),
        ((rtt == nullptr || kGpu)
             ? nullptr
             : &(kDirFwd ? rtt->fwd_search_lb_graph_has_edges_
                         : rtt->bwd_search_lb_graph_has_edges_)),
        ((rtt == nullptr || kGpu) ? nullptr
                                  : &(kDirFwd ? rtt->fwd_search_lb_graph_
                                              : rtt->bwd_search_lb_graph_)),
        lb);
    lb_time += std::chrono::steady_clock::now() - start;
  };
  auto ping_lb = std::vector<std::uint16_t>{};
  auto rev_lb = std::vector<std::uint16_t>{};
  auto bounded_lb = std::vector<std::uint16_t>{};
  if constexpr (ping_t::kUseLowerBounds) {
    compute_lb.template operator()<SearchDir>(q, ping_lb);
    compute_lb.template operator()<kRev>(qr, rev_lb);
    bounded_lb.assign(tt.n_locations(), 0U);
  }

  auto const make = [&]<typename Algo>(side& s, query const& x,
                                       std::vector<std::uint16_t>& lb) {
    return Algo{tt,
                rtt,
                r_state,
                s.is_dest_,
                s.is_via_,
                s.dist_to_dest_,
                x.td_dest_,
                lb,
                x.via_stops_,
                base_day,
                x.allowed_claszes_,
                x.require_bike_transport_,
                x.require_car_transport_,
                x.prf_idx_ == 2U,
                x.no_compulsory_reservation_,
                x.transfer_time_settings_,
                x.prf_idx_};
  };
  auto ping = make.template operator()<ping_t>(fwd_side, q, ping_lb);
  auto rev_ping = make.template operator()<rev_ping_t>(rev_side, qr, rev_lb);
  auto rev_pong = make.template operator()<rev_pong_t>(fwd_side, q, bounded_lb);

  auto result = routing_result{.journeys_ = &s_state.results_,
                               .interval_ = search_interval,
                               .search_stats_ = {},
                               .algo_stats_ = {}};
  auto starts = std::vector<start>{};
  auto const worst_array = [](unixtime_t const t) {
    auto a = std::array<unixtime_t, kMaxTransfers + 2U>{};
    a.fill(t);
    return a;
  };
  auto const best_of = [&](pareto_set<journey> const& set) {
    return std::min_element(begin(set), end(set),
                            [&](journey const& a, journey const& b) {
                              return is_better(a.dest_time_, b.dest_time_);
                            })
        ->dest_time_;
  };

  auto page_edge = t0 + step;  // validated: page_edge..t0 (both inclusive)
  auto const is_validated = [&](journey const& j) {
    return !is_better(j.start_time_, page_edge) &&
           !is_better(t0, j.start_time_);
  };
  auto const is_fast = [&](journey const& j) {
    return j.travel_time() < fastest_direct &&
           j.travel_time() <= q.max_travel_time_;
  };
  auto const is_known = [&](journey const& j) {
    return j.is_reconstructed_ && !j.error_ &&
           !is_better(j.start_time_, lookahead_end);
  };

  // per round: the arrivals of the adjacent page's first ping
  auto adjacent_worst = worst_array(t0 + step + max_look);

  // ping from the start side: earliest arrivals for departures >= t with at
  // most max_transfers, bounded by the known journeys departing >= t
  auto const run_ping = [&](unixtime_t const t,
                            std::uint8_t const max_transfers,
                            pareto_set<journey>& out) {
    starts.clear();
    get_starts(SearchDir, tt, rtt, t, q.start_, q.td_start_, q.via_stops_,
               q.max_start_offset_, q.start_match_mode_, q.use_start_footpaths_,
               starts, false, q.prf_idx_, q.transfer_time_settings_);
    ping.reset_arrivals();
    ping.next_start_time();
    for (auto const& s : starts) {
      ping.add_start(s.stop_, s.time_at_stop_, s.v_);
    }
    auto worst_time_at_dest = adjacent_worst;
    for (auto k = 0U; k != worst_time_at_dest.size(); ++k) {
      if (is_better(t + max_look, worst_time_at_dest[k])) {
        worst_time_at_dest[k] = t + max_look;
      }
    }
    for (auto const& j : s_state.results_) {
      if (!j.error_ && !is_better(j.start_time_, t)) {
        for (auto k = j.transfers_ + 1U; k != worst_time_at_dest.size(); ++k) {
          if (is_better(j.dest_time_, worst_time_at_dest[k])) {
            worst_time_at_dest[k] = j.dest_time_;
          }
        }
      }
    }
    ping.execute(t, max_transfers, worst_time_at_dest, out);
    kFwd ? ++result.search_stats_.n_execute_fwd_
         : ++result.search_stats_.n_execute_bwd_;
  };

  // deadline for the reverse ping: before the earliest arrival of everything
  // departing >= T, so that it only finds departures of this page
  auto const no_bound = kFwd ? unixtime_t::max() : unixtime_t::min();
  auto deadline = t0;
  if (q.cursor_bounds_.has_value()) {
    for (auto k = 0U; k != adjacent_worst.size(); ++k) {
      auto const b = (*q.cursor_bounds_)[k];
      if (b != no_bound) {
        adjacent_worst[k] = b;
        if (deadline == t0 || is_better(b - step, deadline)) {
          deadline = b - step;
        }
      }
    }
  } else {
    auto first = pareto_set<journey>{};
    run_ping(t0 + step, q.max_transfers_, first);
    for (auto const& j : first) {
      for (auto k = j.transfers_ + 1U; k != adjacent_worst.size(); ++k) {
        if (is_better(j.dest_time_, adjacent_worst[k])) {
          adjacent_worst[k] = j.dest_time_;
        }
      }
    }
    if (!first.empty()) {
      deadline = best_of(first) - step;
    }
  }

  // one step from the arrival side: the latest departures (>= from) arriving
  // before the deadline, with their exact arrival; then the deadline moves
  // before the latest of these arrivals
  auto rev_exhausted = false;
  auto const rev_step = [&](unixtime_t const from,
                            std::uint8_t const max_transfers) {
    starts.clear();
    get_starts(kRev, tt, rtt, deadline, qr.start_, qr.td_start_, qr.via_stops_,
               qr.max_start_offset_, qr.start_match_mode_,
               qr.use_start_footpaths_, starts, false, qr.prf_idx_,
               qr.transfer_time_settings_);
    rev_ping.reset_arrivals();
    rev_ping.next_start_time();
    for (auto const& s : starts) {
      rev_ping.add_start(s.stop_, s.time_at_stop_, s.v_);
    }
    auto rev_results = pareto_set<journey>{};
    rev_ping.execute(deadline, max_transfers, worst_array(from), rev_results);
    kFwd ? ++result.search_stats_.n_execute_bwd_
         : ++result.search_stats_.n_execute_fwd_;
    if (rev_results.empty()) {
      rev_exhausted = true;
      return;
    }
    utl::sort(rev_results, [](journey const& a, journey const& b) {
      return a.transfers_ > b.transfers_;
    });

    if constexpr (kPruneWithPingBounds) {
      rev_ping.fill_bounds(rev_results.begin()->transfers_ + std::size_t{1U});
    }
    rev_pong.reset_arrivals();
    for (auto& pj : rev_results) {
      auto const known = utl::find_if(s_state.results_, [&](journey const& j) {
        return j.transfers_ == pj.transfers_ && j.start_time_ == pj.dest_time_;
      });
      if (known != end(s_state.results_)) {
        pj.start_time_ = known->dest_time_;
        continue;
      }
      if constexpr (kPruneWithPingBounds) {
        rev_pong.set_bounds(pj.transfers_ + 1U);
      }
      starts.clear();
      get_starts(SearchDir, tt, rtt, pj.dest_time_, q.start_, q.td_start_,
                 q.via_stops_, q.max_start_offset_, q.start_match_mode_,
                 q.start_match_mode_ != location_match_mode::kIntermodal,
                 starts, false, q.prf_idx_, q.transfer_time_settings_);
      rev_pong.next_start_time();
      for (auto const& x : starts) {
        rev_pong.add_start(x.stop_, x.time_at_stop_, x.v_);
      }
      auto pong_results = pareto_set<journey>{};
      rev_pong.execute(pj.dest_time_, pj.transfers_,
                       worst_array(pj.start_time_), pong_results);
      kFwd ? ++result.search_stats_.n_execute_fwd_
           : ++result.search_stats_.n_execute_bwd_;
      auto const match = utl::find_if(pong_results, [&](journey const& j) {
        return j.transfers_ == pj.transfers_ && j.start_time_ == pj.dest_time_;
      });
      if (match == end(pong_results)) {
        throw utl::fail("pong_reverse: no pong for transfers={}, time={}",
                        pj.transfers_, pj.dest_time_);
      }
      rev_pong.reconstruct(q, *match);
      pj.start_time_ = match->dest_time_;
      s_state.results_.add(std::move(*match));
    }

    auto const next =
        std::max_element(begin(rev_results), end(rev_results),
                         [&](journey const& a, journey const& b) {
                           return is_better(a.start_time_, b.start_time_);
                         })
            ->start_time_ -
        step;
    if (!is_better(next, deadline)) {
      throw utl::fail("pong_reverse: no progress: deadline={}, next={}",
                      deadline, next);
    }
    deadline = next;
  };
  auto const can_step = [&]() {
    return !rev_exhausted && tt.external_interval().contains(deadline) &&
           is_better(lookahead_end, deadline) && !is_timeout_reached();
  };

  auto const first_deadline = deadline;

  // the page starts at the departure that makes it hold enough journeys
  // (no such departure: everything known is in the page)
  auto const choose_page_edge = [&]() {
    auto known = std::vector<journey const*>{};
    for (auto const& j : s_state.results_) {
      if (is_known(j) && !is_better(t0, j.start_time_)) {
        known.push_back(&j);
      }
    }
    utl::sort(known, [&](journey const* a, journey const* b) {
      return is_better(b->start_time_, a->start_time_);
    });
    auto n = 0U;
    auto edge = t0 + step;
    for (auto const* j : known) {
      edge = j->start_time_;
      n += is_fast(*j) ? 2U : 1U;
      if (n >= 2 * q.min_connection_count_) {
        return std::pair{edge, true};
      }
    }
    return std::pair{edge, false};
  };
  auto const clamp_page_edge = [&](unixtime_t edge) {
    if (is_better(target, edge)) {
      edge = target;
    }
    if (is_better(edge, lookahead_end)) {
      edge = lookahead_end;
    }
    return edge;
  };

  // enumerate from the arrival side until the page holds enough journeys
  auto enough = false;
  while (can_step()) {
    auto const [edge, is_enough] = choose_page_edge();
    enough = is_enough;
    if (enough) {
      page_edge = clamp_page_edge(edge);
      break;
    }
    rev_step(deadline - max_look, q.max_transfers_);
  }
  if (!enough) {
    page_edge = clamp_page_edge(choose_page_edge().first);
  }
  auto const main_deadline = deadline;
  auto const main_exhausted = rev_exhausted;

  // slow journeys (arriving at or after the adjacent page's earliest arrival)
  // are pareto-optimal only with fewer transfers than every adjacent-page
  // journey arriving before them: with k transfers they arrive before the
  // adjacent page's best arrival with <= k transfers (round k+1). They are
  // enumerated from the arrival side per transfer count, from the most
  // transfers down over disjoint arrival ranges, bounded by the page start,
  // which follows the journeys found
  auto covered = first_deadline + step;
  for (auto k = q.max_transfers_ + 1U; k-- != 0U;) {
    auto const upper = adjacent_worst[k + 1U];
    if (!is_better(covered, upper)) {
      continue;
    }
    deadline = upper - step;
    if (!tt.external_interval().contains(deadline)) {
      deadline = kFwd ? tt.external_interval().to_ - duration_t{1}
                      : tt.external_interval().from_;
    }
    rev_exhausted = false;
    while (!rev_exhausted && !is_better(deadline, covered) &&
           tt.external_interval().contains(deadline) && !is_timeout_reached()) {
      auto const [edge, is_enough] = choose_page_edge();
      enough = is_enough;
      if (enough) {
        page_edge = clamp_page_edge(edge);
      }
      rev_step(enough ? page_edge : deadline - max_look,
               static_cast<std::uint8_t>(k));
    }
    covered = upper;
  }

  // the rest of the page from the arrival side, bounded by the page start,
  // which follows the journeys found
  deadline = main_deadline;
  rev_exhausted = main_exhausted;
  while (can_step()) {
    auto const [edge, is_enough] = choose_page_edge();
    enough = is_enough;
    if (enough) {
      page_edge = clamp_page_edge(edge);
      if (is_better(deadline, page_edge)) {
        break;
      }
      rev_step(page_edge, q.max_transfers_);
    } else {
      rev_step(deadline - max_look, q.max_transfers_);
    }
  }
  page_edge = clamp_page_edge(choose_page_edge().first);

  // per round the best dest time of everything departing in or after the page
  auto bounds = std::array<unixtime_t, kMaxTransfers + 2U>{};
  for (auto k = 0U; k != bounds.size(); ++k) {
    bounds[k] = adjacent_worst[k] == t0 + step + max_look ? no_bound
                                                          : adjacent_worst[k];
  }
  for (auto const& j : s_state.results_) {
    if (is_known(j) && is_validated(j)) {
      for (auto k = j.transfers_ + 1U; k != bounds.size(); ++k) {
        if (is_better(j.dest_time_, bounds[k])) {
          bounds[k] = j.dest_time_;
        }
      }
    }
  }
  result.cursor_bounds_ = bounds;

  utl::erase_if(s_state.results_, [&](journey const& j) {
    return !j.is_reconstructed_ || !is_validated(j) ||
           j.travel_time() >= fastest_direct ||
           j.travel_time() > q.max_travel_time_;
  });

  result.interval_ =
      kFwd ? interval{page_edge, t0 + step} : interval{t0, page_edge - step};
  result.search_stats_.lb_time_ = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(lb_time).count());
  result.algo_stats_ =
      (ping.get_stats() + rev_ping.get_stats() + rev_pong.get_stats()).to_map();
  result.search_stats_.execute_time_ =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          (std::chrono::steady_clock::now() - processing_start_time));

  finish_journeys<SearchDir, Vias>(tt, rtt, q, result.interval_,
                                   s_state.results_);
  return result;
}

template <direction SearchDir, via_offset_t Vias, typename AlgoState>
routing_result pong_with_vias(timetable const& tt,
                              rt_timetable const* rtt,
                              search_state& s_state,
                              AlgoState& r_state,
                              query q,
                              std::optional<std::chrono::seconds> timeout) {
  auto const reverse =
      SearchDir == direction::kForward
          ? q.extend_interval_earlier_ && !q.extend_interval_later_
          : q.extend_interval_later_ && !q.extend_interval_earlier_;
  if (reverse) {
    return rtt == nullptr
               ? pong_reverse<SearchDir, false, Vias>(tt, rtt, s_state, r_state,
                                                      std::move(q), timeout)
               : pong_reverse<SearchDir, true, Vias>(tt, rtt, s_state, r_state,
                                                     std::move(q), timeout);
  }
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
  q.sanitize(tt);

  if (rtt != nullptr) {
    auto const a = pong_search_interval(SearchDir, tt, q, kMinLookAhead);
    auto const b = pong_search_interval(flip(SearchDir), tt, q, kMinLookAhead);
    if (!rtt->affects(
            interval{std::min(a.from_, b.from_), std::max(a.to_, b.to_)},
            q.prf_idx_)) {
      rtt = nullptr;
    }
  }

  if constexpr (std::is_same_v<AlgoState, gpu::gpu_raptor_state>) {
    utl::verify(q.via_stops_.empty(), "GPU raptor does not support vias");
    return pong_with_vias<SearchDir, 0>(tt, rtt, s_state, r_state, std::move(q),
                                        timeout);
  } else {
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
