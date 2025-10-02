#include "nigiri/routing/raptor/pong.h"

namespace nigiri::routing {

template <direction SearchDir, bool Rt, via_offset_t Vias>
routing_result pong(timetable const& tt,
                    rt_timetable const* rtt,
                    search_state& s_state,
                    raptor_state& r_state,
                    query q) {
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
  auto ping_travel_time_lb = std::vector<std::uint16_t>{};
  dijkstra(tt, q,
           SearchDir == direction::kForward
               ? tt.fwd_search_lb_graph_[q.prf_idx_]
               : tt.bwd_search_lb_graph_[q.prf_idx_],
           ping_travel_time_lb);

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
      ping_travel_time_lb,
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
  auto pong_travel_time_lb = std::vector<std::uint16_t>{};
  pong_travel_time_lb.resize(tt.n_locations());
  utl::fill(pong_travel_time_lb, 0U);

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
      pong_travel_time_lb,
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
  auto start_time = SearchDir == direction::kForward
                        ? search_interval.from_
                        : search_interval.to_ - duration_t{1};
  while (result.journeys_->size() < q.min_connection_count_ &&
         tt.external_interval().contains(start_time)) {
    // ----
    // PING
    // ----

    starts.clear();
    get_starts(SearchDir, tt, rtt, start_time, q.start_, q.td_start_,
               q.max_start_offset_, q.start_match_mode_, q.use_start_footpaths_,
               starts, false, q.prf_idx_, q.transfer_time_settings_);
    ping.reset_arrivals();
    ping.next_start_time();
    for (auto const& s : starts) {
      ping.add_start(s.stop_, s.time_at_stop_);
    }
    auto const worst_time_at_dest =
        start_time +
        (kFwd ? 1 : -1) *
            (std::min(fastest_direct, q.max_travel_time_) + duration_t{1});
    auto ping_results = pareto_set<journey>{};
    ping.execute(start_time, q.max_transfers_, worst_time_at_dest, q.prf_idx_,
                 ping_results);

    // --
    // UB
    // --

    for (auto& i : r_state.round_times_storage_) {
      if constexpr (kFwd) {
        i -= 1;
      } else {
        i += 1;
      }
    }
    for (auto& i : r_state.best_storage_) {
      if constexpr (kFwd) {
        i -= 1;
      } else {
        i += 1;
      }
    }

    // ----
    // PONG
    // ----

    assert(utl::is_sorted(ping_results,
                          [](journey const& a, journey const& b) {
                            // Journeys are found sorted by transfers:
                            // -> more transfers = shorter travel time
                            // -> longest travel time will be found first
                            return a.travel_time() > b.travel_time();
                          }) &&
           "ping results not sorted");

    q.flip_dir();
    pong.reset_arrivals();
    for (auto const& j : ping_results) {
      pong.next_start_time();
      pong.add_start(j.dest_, j.dest_time_);
      pong.execute(j.dest_time_, j.transfers_,
                   j.start_time_ - duration_t{kFwd ? 1 : -1}, q.prf_idx_,
                   s_state.results_);
      for (auto& r : s_state.results_) {
        if (r.transfers_ == j.transfers_ && r.start_time_ == j.dest_time_) {
          pong.reconstruct(q, r);
        }
      }
    }
    q.flip_dir();

    // ----
    // PONG
    // ----
  }
  return result;
}

template <direction SearchDir, via_offset_t Vias>
routing_result pong_with_vias(timetable const& tt,
                              rt_timetable const* rtt,
                              search_state& s_state,
                              raptor_state& r_state,
                              query q) {
  if (rtt == nullptr) {
    return pong<SearchDir, true, Vias>(tt, rtt, s_state, r_state, std::move(q));
  } else {
    return pong<SearchDir, true, Vias>(tt, rtt, s_state, r_state, std::move(q));
  }
}

template <direction SearchDir>
routing_result pong_search_with_dir(timetable const& tt,
                                    rt_timetable const* rtt,
                                    search_state& s_state,
                                    raptor_state& r_state,
                                    query q) {
  switch (q.via_stops_.size()) {
    case 0:
      return pong_with_vias<SearchDir, 0>(tt, rtt, s_state, r_state,
                                          std::move(q));
    case 1:
      return pong_with_vias<SearchDir, 1>(tt, rtt, s_state, r_state,
                                          std::move(q));
    case 2:
      return pong_with_vias<SearchDir, 2>(tt, rtt, s_state, r_state,
                                          std::move(q));
  }
  throw utl::fail("{} vias not supported (max={})", kMaxVias);
}

routing_result pong_search(
    timetable const& tt,
    rt_timetable const* rtt,
    search_state& s_state,
    raptor_state& r_state,
    query q,
    direction search_dir,
    std::optional<std::chrono::seconds>  // TODO(felix) maybe reuse?
) {
  if (search_dir == direction::kForward) {
    return pong_search_with_dir<direction::kForward>(tt, rtt, s_state, r_state,
                                                     std::move(q));
  } else {
    return pong_search_with_dir<direction::kBackward>(tt, rtt, s_state, r_state,
                                                      std::move(q));
  }
}

}  // namespace nigiri::routing