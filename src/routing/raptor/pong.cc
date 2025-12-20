#include "nigiri/routing/raptor/pong.h"

#include <ranges>

#include "utl/sorted_diff.h"

#include "nigiri/routing/get_earliest_transport.h"
#include "nigiri/rt/frun.h"

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
    unixtime_t const to_dep,
    bitvec& is_src,
    bitvec& is_dst) {
  auto const merge_sorted = [](auto& dst, auto const& src) {
    auto const original_size = static_cast<int>(dst.size());
    dst.resize(dst.size() + src.size());
    std::copy(begin(src), end(src), begin(dst) + original_size);
    std::inplace_merge(begin(dst), begin(dst) + original_size, end(dst));
    dst.erase(std::unique(begin(dst), end(dst)), end(dst));
  };

  auto const add = [&](auto& dst, auto& marker, auto& copy_from,
                       location_idx_t const l, direction const dir) {
    marker.set(to_idx(l), true);
    merge_sorted(dst, copy_from[l]);

    for (auto const& fp : (dir == direction::kForward
                               ? tt.locations_.footpaths_out_
                               : tt.locations_.footpaths_in_)[q.prf_idx_][l]) {
      marker.set(to_idx(fp.target()), true);
      merge_sorted(dst, copy_from[fp.target()]);
    }
  };

  is_src.resize(tt.n_locations());
  is_dst.resize(tt.n_locations());
  is_src.zero_out();
  is_dst.zero_out();

  // Determine earliest departure + adjusted transfers at ingress stops.
  auto ingress = hash_map<location_idx_t, std::pair<footpath, unixtime_t>>{};
  {
    auto transfer_time = adjusted_transfer_time(
        q.transfer_time_settings_, tt.locations_.transfer_time_[from]);
    ingress.emplace(from, std::pair{footpath{from, transfer_time},
                                    from_arr + transfer_time});
    for (auto const fp : tt.locations_.footpaths_out_[q.prf_idx_][from]) {
      transfer_time =
          adjusted_transfer_time(q.transfer_time_settings_, fp.duration());
      ingress.emplace(fp.target(),
                      std::pair{footpath{fp.target(), transfer_time},
                                from_arr + transfer_time});
    }
  }

  // Join all relevent routes.
  auto from_routes = std::vector<route_idx_t>{},
       to_routes = std::vector<route_idx_t>{};
  add(from_routes, is_src, tt.location_routes_, from, direction::kForward);
  add(to_routes, is_dst, tt.location_routes_, to, direction::kBackward);

  // Join all relevent rt transports.
  auto from_rt_transports = std::vector<rt_transport_idx_t>{},
       to_rt_transports = std::vector<rt_transport_idx_t>{};
  if (rtt != nullptr) {
    add(from_rt_transports, is_src, rtt->location_rt_transports_, from,
        direction::kForward);
    add(to_rt_transports, is_dst, rtt->location_rt_transports_, to,
        direction::kBackward);
  }

  // Visit route in RAPTOR without updating intermediate stops that are not the
  // destination  because there's no next round (no dynamic programming).
  auto const get_earliest =
      [&]<typename T>(T const x,  // route_idx_t | rt_transport_idx_t
                      stop_idx_t const stop_idx,
                      unixtime_t const time) -> std::optional<rt::frun> {
    if constexpr (std::is_same_v<T, rt_transport_idx_t>) {
      auto const dep = rtt->unix_event_time(x, stop_idx, event_type::kDep);
      return time <= dep ? std::optional{rt::frun::from_rt(tt, rtt, x)}
                         : std::nullopt;
    } else {
      auto const [day, mam] = tt.day_idx_mam(time);
      if (rtt == nullptr) {
        auto const t = get_earliest_transport(
            tt, tt, 0U, x, stop_idx, day, mam, location_idx_t::invalid(),
            [](day_idx_t, std::int16_t) { return false; });
        return t.is_valid() ? std::optional{rt::frun::from_t(tt, rtt, t)}
                            : std::nullopt;
      } else {
        auto const t = get_earliest_transport(
            tt, *rtt, 0U, x, stop_idx, day, mam, location_idx_t::invalid(),
            [](day_idx_t, std::int16_t) { return false; });
        return t.is_valid() ? std::optional{rt::frun::from_t(tt, rtt, t)}
                            : std::nullopt;
      }
    }
  };

  auto earliest_arr = to_dep;
  auto best = std::optional<std::array<journey::leg, 3U>>{};
  auto const update_earliest = [&](auto&& loc_seq, auto&& r) {
    struct enter_info {
      journey::leg ingress_leg_;
      rt::frun fr_;
      stop_idx_t enter_stop_idx_;
      location_idx_t enter_location_;
      unixtime_t enter_time_;
    };

    auto et = std::optional<enter_info>{};
    for (auto i = stop_idx_t{0U}; i != loc_seq.size(); ++i) {
      auto stp = stop{loc_seq[i]};

      if (et.has_value() && ((q.require_bike_transport_ &&
                              !et->fr_[i].bikes_allowed(event_type::kArr)) ||
                             (q.require_car_transport_ &&
                              !et->fr_[i].cars_allowed(event_type::kArr)))) {
        et = std::nullopt;
      }

      // Check for earlier arrival at destination.
      // -> update arrival + legs
      if (et.has_value() && is_dst[to_idx(stp.location_idx())] &&
          stp.out_allowed()) {
        auto const trip_arr = et->fr_[i].time(event_type::kArr);

        auto const check_fp = [&](footpath const& fp) {
          if (fp.target() != to) {
            return;
          }

          auto const adjusted_fp_time =
              adjusted_transfer_time(q.transfer_time_settings_, fp.duration());
          auto const dst_arr = trip_arr + adjusted_fp_time;
          if (dst_arr > earliest_arr) {
            return;
          }

          earliest_arr = dst_arr;
          best = std::array<journey::leg, 3U>{
              et->ingress_leg_,
              journey::leg{
                  direction::kForward,
                  et->enter_location_,
                  stp.location_idx(),
                  et->enter_time_,
                  trip_arr,
                  journey::run_enter_exit{et->fr_, et->enter_stop_idx_, i},
              },
              journey::leg{
                  direction::kForward,
                  stp.location_idx(),
                  to,
                  trip_arr,
                  dst_arr,
                  footpath{fp.target(), adjusted_fp_time},
              }};
        };

        check_fp({stp.location_idx(),
                  adjusted_transfer_time(
                      q.transfer_time_settings_,
                      tt.locations_.transfer_time_[stp.location_idx()])});

        if (q.prf_idx_ != 0U &&
            rtt->has_td_footpaths_out_[q.prf_idx_][stp.location_idx()]) {
          for_each_footpath<direction::kForward>(
              rtt->td_footpaths_out_[q.prf_idx_][stp.location_idx()], trip_arr,
              [&](footpath const fp) { check_fp(fp); });
        } else {
          for (auto const& fp :
               tt.locations_.footpaths_out_[q.prf_idx_][stp.location_idx()]) {
            check_fp(fp);
          }
        }
      }

      // Check for earlier trip.
      if (stp.in_allowed(q.prf_idx_) && is_src[to_idx(stp.location_idx())]) {
        auto const [fp, location_arr] = ingress.at(stp.location_idx());
        auto const candidate = get_earliest(r, i, location_arr);
        if (candidate.has_value() &&
            (!et.has_value() || (*candidate)[i].time(event_type::kDep) <
                                    et->fr_[i].time(event_type::kDep))) {
          et =
              enter_info{.ingress_leg_ =
                             journey::leg{
                                 direction::kForward,
                                 from,
                                 stp.location_idx(),
                                 from_arr,
                                 location_arr,
                                 fp,
                             },
                         .fr_ = *candidate,
                         .enter_stop_idx_ = i,
                         .enter_location_ = stp.location_idx(),
                         .enter_time_ = (*candidate)[i].time(event_type::kDep)};
        }
      }
    }
  };

  utl::sorted_diff(
      from_routes, to_routes, std::less<route_idx_t>{},
      [](auto&&, auto&&) { return false; },
      utl::overloaded{
          [](utl::op, route_idx_t) {},
          [&](route_idx_t const r, route_idx_t) {
            if (is_allowed(q.allowed_claszes_, tt.route_clasz_[r]) &&
                (!q.require_bike_transport_ || tt.has_bike_transport(r)) &&
                (!q.require_car_transport_ || tt.has_car_transport(r))) {
              update_earliest(tt.route_location_seq_[r], r);
            }
          }});

  utl::sorted_diff(
      from_rt_transports, to_rt_transports, std::less<rt_transport_idx_t>{},
      [](auto&&, auto&&) { return false; },
      utl::overloaded{
          [](utl::op, rt_transport_idx_t) {},
          [&](rt_transport_idx_t const rt_t, rt_transport_idx_t) {
            if (is_allowed(q.allowed_claszes_,
                           rtt->rt_transport_section_clasz_[rt_t].front()) &&
                (!q.require_bike_transport_ || rtt->has_bike_transport(rt_t)) &&
                (!q.require_car_transport_ || rtt->has_car_transport(rt_t))) {
              update_earliest(rtt->rt_transport_location_seq_[rt_t], rt_t);
            }
          }});

  return best;
}

template <direction SearchDir, bool Rt, via_offset_t Vias>
routing_result pong(timetable const& tt,
                    rt_timetable const* rtt,
                    search_state& s_state,
                    raptor_state& r_state,
                    query q,
                    std::optional<std::chrono::seconds> timeout) {
  constexpr auto kFwd = (SearchDir == direction::kForward);

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
  dijkstra(tt, q,
           rtt == nullptr ? (kFwd ? tt.fwd_search_lb_graph_[q.prf_idx_]
                                  : tt.bwd_search_lb_graph_[q.prf_idx_])
                          : (kFwd ? rtt->fwd_search_lb_graph_[q.prf_idx_]
                                  : rtt->bwd_search_lb_graph_[q.prf_idx_]),
           ping_lb);

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
           rtt == nullptr ? (kFwd ? tt.bwd_search_lb_graph_[q.prf_idx_]
                                  : tt.fwd_search_lb_graph_[q.prf_idx_])
                          : (kFwd ? rtt->bwd_search_lb_graph_[q.prf_idx_]
                                  : rtt->fwd_search_lb_graph_[q.prf_idx_]),
           pong_lb);

  auto pong_dist_to_dest = std::vector<std::uint16_t>{};
  auto pong_is_dest = bitvec{};
  collect_destinations(tt, q.destination_, q.dest_match_mode_, pong_is_dest,
                       pong_dist_to_dest);

  auto pong_is_via = std::array<bitvec, kMaxVias>{};
  for (auto const [i, via] : utl::enumerate(q.via_stops_)) {
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

  q.flip_dir();

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

      auto const earlier = get_earliest_alternative(
          tt, rtt, q, from.get_location_idx(), to.get_location_idx(),
          from.time(event_type::kArr), to.time(event_type::kDep),
          r_state.prev_station_mark_, r_state.station_mark_);

      if (earlier.has_value()) {
        transfer_1 = earlier->at(0);
        transit_2 = earlier->at(1);
        transfer_2 = earlier->at(2);
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