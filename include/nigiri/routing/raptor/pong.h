#pragma once

#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/raptor.h"
#include "nigiri/routing/search.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

template <direction SearchDir, bool Rt, via_offset_t Vias>
routing_result ping_pong(timetable const& tt,
                         rt_timetable const* rtt,
                         query q) {
  constexpr auto const kFwd = (SearchDir == direction::kForward);
  constexpr auto const kBwd = (SearchDir == direction::kBackward);

  q.sanitize(tt);

  auto state = raptor_state{};
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
      state,
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

  auto pong = raptor<SearchDir, Rt, Vias, search_mode::kOneToOne>{
      tt,
      rtt,
      state,
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

  // ====
  // PLAY
  // ----
  auto starts = std::vector<start>{};
  auto result = routing_result{};
  auto start_time = SearchDir == direction::kForward
                        ? search_interval.from_
                        : search_interval.to_ - duration_t{1};
  while (result.journeys_->size() < q.min_connection_count_ &&
         tt.external_interval().contains(start_time)) {
    starts.clear();
    get_starts(SearchDir, tt, rtt, start_time, q.start_, q.td_start_,
               q.max_start_offset_, q.start_match_mode_, q.use_start_footpaths_,
               starts, false, q.prf_idx_, q.transfer_time_settings_);
    ping.next_start_time();
    for (auto const& s : start_time) {
      ping.add_start(s.stop_, s.time_at_stop_);
    }
    auto const worst_time_at_dest =
        start_time +
        (kFwd ? 1 : -1) *
            (std::min(fastest_direct, q.max_travel_time_) + duration_t{1});
    auto ping_results = pareto_set<journey>{};
    ping.execute(start_time, q.max_transfers_, worst_time_at_dest, q.prf_idx_,
                 ping_results);
  }
  return result;
}

}  // namespace nigiri::routing