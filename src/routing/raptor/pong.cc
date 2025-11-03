#include "nigiri/routing/raptor/pong.h"

#include <ranges>

#include "nigiri/routing/direct.h"
#include "nigiri/rt/frun.h"

#define trace_pong(...)
// #define trace_pong fmt::println

namespace nigiri::routing {

auto to_tuple(journey const& j) {
  return std::tuple{j.departure_time(), j.arrival_time(), j.transfers_};
}

template <direction SearchDir, bool Rt, via_offset_t Vias>
routing_result pong(timetable const& tt,
                    rt_timetable const* rtt,
                    search_state& s_state,
                    raptor_state& r_state,
                    query q,
                    std::optional<std::chrono::seconds> timeout) {
  constexpr auto const kFwd = (SearchDir == direction::kForward);

  q.sanitize(tt);

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
  dijkstra(tt, q,
           kFwd ? tt.fwd_search_lb_graph_[q.prf_idx_]
                : tt.bwd_search_lb_graph_[q.prf_idx_],
           ping_lb);
  for (auto const [l, lb] : utl::enumerate(ping_lb)) {
    if (lb != std::numeric_limits<std::decay_t<decltype(lb)>>::max()) {
      trace_pong("ping lb {}: {}", location{tt, location_idx_t{l}}, lb);
    }
  }
  trace_pong("\n");

  auto ping_dist_to_dest = std::vector<std::uint16_t>{};
  auto ping_is_dest = bitvec{};
  auto ping_is_via = std::array<bitvec, kMaxVias>{};
  collect_destinations(tt, q.destination_, q.dest_match_mode_, ping_is_dest,
                       ping_dist_to_dest);
  for (auto const [i, via] : utl::enumerate(q.via_stops_)) {
    collect_via_destinations(tt, via.location_, ping_is_via[i]);
  }

  auto ping = raptor<SearchDir, Rt, Vias, search_mode::kOneToOne>{
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
  dijkstra(tt, q,
           kFwd ? tt.bwd_search_lb_graph_[q.prf_idx_]
                : tt.fwd_search_lb_graph_[q.prf_idx_],
           pong_lb);
  for (auto const [l, lb] : utl::enumerate(pong_lb)) {
    if (lb != std::numeric_limits<std::decay_t<decltype(lb)>>::max()) {
      trace_pong("pong lb {}: {}", location{tt, location_idx_t{l}}, lb);
    }
  }
  trace_pong("\n");
  q.flip_dir();

  auto pong_dist_to_dest = std::vector<std::uint16_t>{};
  auto pong_is_dest = bitvec{};
  collect_destinations(tt, q.start_, q.start_match_mode_, pong_is_dest,
                       pong_dist_to_dest);

  auto pong_is_via = std::array<bitvec, kMaxVias>{};
  auto reverse_via = q.via_stops_;
  std::reverse(begin(reverse_via), end(reverse_via));
  for (auto const [i, via] : utl::enumerate(reverse_via)) {
    collect_via_destinations(tt, via.location_, pong_is_via[i]);
  }

  auto pong = raptor<flip(SearchDir), Rt, Vias, search_mode::kOneToOne>{
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

  // ========
  // >> PLAY!
  // --------
  auto starts = std::vector<start>{};
  auto result = routing_result{.journeys_ = &s_state.results_,
                               .interval_ = search_interval,
                               .search_stats_ = {},
                               .algo_stats_ = {}};
  auto start_time =
      kFwd ? search_interval.from_ : search_interval.to_ - duration_t{1};
  auto const is_better = [](auto a, auto b) { return kFwd ? a < b : a > b; };
  auto const is_validated = [&](journey const& j) {
    return is_better(j.dest_time_, start_time);
  };
  auto const get_result_count = [&]() {
    return utl::count_if(*result.journeys_, [&](journey const& j) {
      return is_validated(j) &&  //
             j.travel_time() < fastest_direct &&
             j.travel_time() < q.max_travel_time_;
    });
  };
  auto const processing_start_time = std::chrono::steady_clock::now();
  auto const is_timeout_reached = [&]() {
    if (timeout) {
      return (std::chrono::steady_clock::now() - processing_start_time) >=
             *timeout;
    }
    return false;
  };
  while (get_result_count() < q.min_connection_count_ &&
         tt.external_interval().contains(start_time) && !is_timeout_reached()) {
    // ----
    // PING
    // ----

    trace_pong("START_TIME={}", start_time);

    starts.clear();
    get_starts(SearchDir, tt, rtt, start_time, q.start_, q.td_start_,
               q.max_start_offset_, q.start_match_mode_, q.use_start_footpaths_,
               starts, false, q.prf_idx_, q.transfer_time_settings_);
    ping.reset_arrivals();
    ping.next_start_time();
    for (auto const& s : starts) {
      trace_pong("--- PING START: {} at time_at_start={} time_at_stop={}",
                 location{tt, s.stop_}, s.time_at_start_, s.time_at_stop_);
      ping.add_start(s.stop_, s.time_at_stop_);
    }
    auto const worst_time_at_dest =
        start_time + (kFwd ? 1 : -1) * (q.max_travel_time_ + duration_t{1});
    auto ping_results = pareto_set<journey>{};
    ping.execute(start_time, q.max_transfers_, worst_time_at_dest, q.prf_idx_,
                 ping_results);
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
                 q.td_start_, q.max_start_offset_, q.start_match_mode_,
                 q.start_match_mode_ != location_match_mode::kIntermodal,
                 starts, false, q.prf_idx_, q.transfer_time_settings_);
      pong.next_start_time();
      for (auto const& s : starts) {
        trace_pong("---- PONG START: {} at time_at_start={} time_at_stop={}",
                   location{tt, s.stop_}, s.time_at_start_, s.time_at_stop_);
        ping.add_start(s.stop_, s.time_at_stop_);
      }
      pong.execute(ping_j.dest_time_, ping_j.transfers_,
                   ping_j.start_time_ - duration_t{kFwd ? 1 : -1}, q.prf_idx_,
                   s_state.results_);

      auto const match =
          utl::find_if(s_state.results_, [&](journey const& pong_j) {
            return pong_j.transfers_ == ping_j.transfers_ &&
                   pong_j.start_time_ == ping_j.dest_time_;
          });

      if (match == end(s_state.results_)) {
        fmt::println(
            "no pong for transfers={}, start_time={} found, journeys={}",
            ping_j.transfers_, ping_j.dest_time_,
            s_state.results_.els_ | std::views::transform(to_tuple));
        continue;
      }

      trace_pong("---- HIT [updating ping start time {} -> {}]\n",
                 ping_j.start_time_, match->dest_time_);
      if (match->legs_.empty() && !match->error_) {
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

    trace_pong(
        "AFTER {} [next={}]:\n\t{}", start_time, next,
        fmt::join(s_state.results_.els_ | std::views::transform(to_tuple),
                  "\n\t"));

    start_time = next;
  }

  utl::erase_if(s_state.results_, [&](journey const& j) {
    auto const erase = j.legs_.empty() || !is_validated(j) ||
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

  utl::sort(s_state.results_, [](journey const& a, journey const& b) {
    return std::tuple{a.start_time_, a.transfers_} <
           std::tuple{b.start_time_, b.transfers_};
  });

  trace_pong("RESULT:\n\t{}",
             fmt::join(s_state.results_.els_ | std::views::transform(to_tuple),
                       "\n\t"));

  result.interval_ = {kFwd ? search_interval.from_ : start_time + duration_t{1},
                      kFwd ? start_time : search_interval.to_};
  result.algo_stats_ = (ping.get_stats() + pong.get_stats()).to_map();

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

  if constexpr (!kFwd) {
    return result;
  }

  auto no_td_dest = td_offsets_t{};
  auto no_vias = std::vector<via_stop>{};
  utl::fill(ping_lb, 0U);
  for (auto& v : ping_is_via) {
    v.zero_out();
  }

  auto results = pareto_set<journey>{};
  for (auto& j : s_state.results_) {
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

      auto const back_r = rt::frun{tt, rtt, back.r_};
      auto const to = back_r[back.stop_range_.from_];

      results.clear();

      auto const dest =
          std::vector<offset>{{{to.get_location_idx(), 0_minutes, 0U}}};
      auto start =
          std::vector<offset>{{{from.get_location_idx(), 0_minutes, 0U}}};
      collect_destinations(tt, dest, location_match_mode::kExact, ping_is_dest,
                           ping_dist_to_dest);
      auto earlier = raptor<SearchDir, Rt, 0U, search_mode::kOneToOne>{
          tt,
          rtt,
          r_state,
          ping_is_dest,
          ping_is_via,
          ping_dist_to_dest,
          no_td_dest,
          ping_lb,
          no_vias,
          base_day,
          q.allowed_claszes_,
          q.require_bike_transport_,
          q.require_car_transport_,
          q.prf_idx_ == 2U,
          q.transfer_time_settings_};
      earlier.reset_arrivals();
      earlier.next_start_time();

      earlier.add_start(
          from.get_location_idx(),
          from.time(event_type::kArr) +
              tt.locations_.transfer_time_[from.get_location_idx()]);
      for (auto const& fp :
           tt.locations_.footpaths_out_[q.prf_idx_][from.get_location_idx()]) {
        earlier.add_start(
            fp.target(), from.time(event_type::kArr) +
                             (kFwd ? 1 : -1) *
                                 adjusted_transfer_time(
                                     q.transfer_time_settings_, fp.duration()));
      }

      earlier.execute(start_time, 1U, to.time(event_type::kDep), q.prf_idx_,
                      results);

      auto q1 = q;
      q1.start_time_ = from.time(event_type::kArr);
      q1.start_ = std::move(start);
      q1.start_match_mode_ = location_match_mode::kExact;
      q1.dest_match_mode_ = location_match_mode::kExact;
      for (auto& x : results) {
        earlier.reconstruct(q1, x);

        utl::verify(x.legs_.size() == 3U, "expected 3 legs, got {}",
                    x.legs_.size());

        transfer_1 = x.legs_[0];
        transit_2 = x.legs_[1];
        transfer_2 = x.legs_[2];
      }
    }
  }

  return result;
}

template <direction SearchDir, via_offset_t Vias>
routing_result pong_with_vias(timetable const& tt,
                              rt_timetable const* rtt,
                              search_state& s_state,
                              raptor_state& r_state,
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

template <direction SearchDir>
routing_result pong_search_with_dir(
    timetable const& tt,
    rt_timetable const* rtt,
    search_state& s_state,
    raptor_state& r_state,
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

routing_result pong_search(timetable const& tt,
                           rt_timetable const* rtt,
                           search_state& s_state,
                           raptor_state& r_state,
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

}  // namespace nigiri::routing