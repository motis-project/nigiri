#include "nigiri/routing/direct.h"

#include <queue>

#include "utl/concat.h"
#include "utl/erase_duplicates.h"
#include "utl/erase_if.h"
#include "utl/overloaded.h"
#include "utl/sorted_diff.h"

#include "nigiri/rt/frun.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/special_stations.h"
#include "nigiri/td_footpath.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

namespace {

template <direction Dir>
std::optional<offset> lookup_duration(std::vector<offset> const& static_offs,
                                      td_offsets_t const& td_offs,
                                      location_idx_t const loc,
                                      unixtime_t const t) {
  if (auto const it = td_offs.find(loc); it != end(td_offs)) {
    auto const td = get_td_duration<Dir>(it->second, t);
    if (!td.has_value() || td->first >= footpath::kMaxDuration) {
      return std::nullopt;
    }
    return offset{loc, td->first, td->second.transport_mode_id_};
  }
  for (auto const& o : static_offs) {
    if (o.target() == loc && o.duration_ < footpath::kMaxDuration) {
      return o;
    }
  }
  return std::nullopt;
}

bool sections_violate_constraints(rt::frun const& fr_probe,
                                  unsigned const from_section_idx,
                                  unsigned const to_section_idx,
                                  bool const require_bike,
                                  bool const require_car) {
  if (!require_bike && !require_car) {
    return false;
  }
  for (auto i = from_section_idx; i != to_section_idx; ++i) {
    auto const section_start = static_cast<stop_idx_t>(i);
    if (require_bike &&
        !fr_probe[section_start].bikes_allowed(event_type::kDep)) {
      return true;
    }
    if (require_car &&
        !fr_probe[section_start].cars_allowed(event_type::kDep)) {
      return true;
    }
  }
  return false;
}

template <direction Dir>
utl::generator<std::array<journey::leg, 3>> route_gen(
    timetable const& tt,
    rt_timetable const* rtt,
    route_idx_t const r,
    stop_idx_t const start_stop_idx,
    stop_idx_t const end_stop_idx,
    query const q,
    unixtime_t const time) {
  constexpr auto kFwd = Dir == direction::kForward;

  auto day_int = static_cast<int>(to_idx(tt.day_idx_mam(time).first));
  auto const start_events = tt.event_times_at_stop(
      r, start_stop_idx, kFwd ? event_type::kDep : event_type::kArr);
  auto const n_transports = tt.route_transport_ranges_[r].size();
  auto const loc_seq = tt.route_location_seq_[r];
  auto const start_loc = stop{loc_seq[start_stop_idx]}.location_idx();
  auto const end_loc = stop{loc_seq[end_stop_idx]}.location_idx();

  auto const day_lo =
      static_cast<int>(to_idx(tt.day_idx(tt.internal_interval_days().from_)));
  auto const day_hi =
      static_cast<int>(to_idx(tt.day_idx(tt.internal_interval_days().to_)));

  while (kFwd ? day_int <= day_hi : day_int >= day_lo) {
    auto const day = day_idx_t{static_cast<day_idx_t::value_t>(day_int)};
    for (auto t_offset = std::size_t{0}; t_offset < n_transports; ++t_offset) {
      auto const idx = kFwd ? t_offset : (n_transports - 1U - t_offset);
      auto const ev = start_events[idx];
      auto const t = tt.route_transport_ranges_[r][idx];
      auto const ev_day_offset = ev.days();
      auto const day_off = static_cast<int>(to_idx(day)) - ev_day_offset;
      if (day_off < 0) {
        continue;
      }
      auto const start_day = static_cast<std::size_t>(day_off);
      auto const& bitfields = rtt != nullptr ? rtt->bitfields_ : tt.bitfields_;
      auto const traffic_idx = rtt != nullptr ? rtt->transport_traffic_days_[t]
                                              : tt.transport_traffic_days_[t];
      if (start_day >= bitfields[traffic_idx].size() ||
          !bitfields[traffic_idx].test(start_day)) {
        continue;
      }

      auto const tr =
          transport{t, day_idx_t{static_cast<day_idx_t::value_t>(start_day)}};
      auto const start_time = tt.event_time(
          tr, start_stop_idx, kFwd ? event_type::kDep : event_type::kArr);
      auto const end_time = tt.event_time(
          tr, end_stop_idx, kFwd ? event_type::kArr : event_type::kDep);

      auto const start_off = lookup_duration<flip(Dir)>(q.start_, q.td_start_,
                                                        start_loc, start_time);
      if (!start_off.has_value()) {
        continue;
      }

      auto const end_off =
          lookup_duration<Dir>(q.destination_, q.td_dest_, end_loc, end_time);
      if (!end_off.has_value()) {
        continue;
      }

      auto transit = journey::leg{
          Dir,
          start_loc,
          end_loc,
          start_time,
          end_time,
          journey::run_enter_exit{
              rt::frun{tt, rtt,
                       rt::run{.t_ = tr,
                               .stop_range_ = {0U, static_cast<stop_idx_t>(
                                                       loc_seq.size())}}},
              start_stop_idx, end_stop_idx}};

      auto const& orig_off = kFwd ? *start_off : *end_off;
      auto const& dest_off = kFwd ? *end_off : *start_off;
      auto const boarding_loc = kFwd ? start_loc : end_loc;
      auto const alighting_loc = kFwd ? end_loc : start_loc;
      auto const boarding_time = kFwd ? start_time : end_time;
      auto const alighting_time = kFwd ? end_time : start_time;

      auto orig_leg = journey::leg{
          direction::kForward, get_special_station(special_station::kStart),
          boarding_loc,        boarding_time - orig_off.duration_,
          boarding_time,       orig_off};
      auto dest_leg = journey::leg{direction::kForward,
                                   alighting_loc,
                                   get_special_station(special_station::kEnd),
                                   alighting_time,
                                   alighting_time + dest_off.duration_,
                                   dest_off};

      // fwd: skip if origin departure is before the lower-bound `time`.
      // bwd: skip if dest arrival is after the upper-bound `time`.
      if (kFwd ? orig_leg.dep_time_ < time : dest_leg.arr_time_ > time) {
        continue;
      }

      co_yield std::array<journey::leg, 3>{
          std::move(orig_leg), std::move(transit), std::move(dest_leg)};
    }

    if constexpr (kFwd) {
      ++day_int;
    } else {
      --day_int;
    }
  }
}

template <direction Dir>
utl::generator<std::array<journey::leg, 3>> rt_gen(
    timetable const& tt,
    rt_timetable const& rtt,
    rt_transport_idx_t const rt_idx,
    stop_idx_t const start_stop_idx,
    stop_idx_t const end_stop_idx,
    query const q,
    unixtime_t const time) {
  constexpr auto kFwd = Dir == direction::kForward;
  auto const start_time = rtt.unix_event_time(
      rt_idx, start_stop_idx, kFwd ? event_type::kDep : event_type::kArr);
  auto const end_time = rtt.unix_event_time(
      rt_idx, end_stop_idx, kFwd ? event_type::kArr : event_type::kDep);
  auto const loc_seq = rtt.rt_transport_location_seq_[rt_idx];
  auto const start_loc = stop{loc_seq[start_stop_idx]}.location_idx();
  auto const end_loc = stop{loc_seq[end_stop_idx]}.location_idx();

  auto const start_off =
      lookup_duration<flip(Dir)>(q.start_, q.td_start_, start_loc, start_time);
  if (!start_off.has_value()) {
    co_return;
  }
  auto const end_off =
      lookup_duration<Dir>(q.destination_, q.td_dest_, end_loc, end_time);
  if (!end_off.has_value()) {
    co_return;
  }

  auto transit = journey::leg{
      Dir,
      start_loc,
      end_loc,
      start_time,
      end_time,
      journey::run_enter_exit{
          rt::frun{tt, &rtt,
                   rt::run{.stop_range_ = {0U, static_cast<stop_idx_t>(
                                                   loc_seq.size())},
                           .rt_ = rt_idx}},
          start_stop_idx, end_stop_idx}};

  auto const& orig_off = kFwd ? *start_off : *end_off;
  auto const& dest_off = kFwd ? *end_off : *start_off;
  auto const boarding_loc = kFwd ? start_loc : end_loc;
  auto const alighting_loc = kFwd ? end_loc : start_loc;
  auto const boarding_time = kFwd ? start_time : end_time;
  auto const alighting_time = kFwd ? end_time : start_time;

  auto orig_leg = journey::leg{
      direction::kForward, get_special_station(special_station::kStart),
      boarding_loc,        boarding_time - orig_off.duration_,
      boarding_time,       orig_off};
  auto dest_leg = journey::leg{direction::kForward,
                               alighting_loc,
                               get_special_station(special_station::kEnd),
                               alighting_time,
                               alighting_time + dest_off.duration_,
                               dest_off};

  if (kFwd ? orig_leg.dep_time_ < time : dest_leg.arr_time_ > time) {
    co_return;
  }

  co_yield std::array<journey::leg, 3>{std::move(orig_leg), std::move(transit),
                                       std::move(dest_leg)};
}

template <direction Dir, typename LocSeq, typename Fn>
void for_each_pair(LocSeq const& loc_seq,
                   hash_set<location_idx_t> const& start_locs,
                   hash_set<location_idx_t> const& end_locs,
                   profile_idx_t const prf_idx,
                   Fn&& fn) {
  constexpr auto kFwd = Dir == direction::kForward;

  auto const is_wheelchair = prf_idx == kWheelchairProfile;
  auto first = std::optional<stop_idx_t>{};
  for (auto i = 0U; i != loc_seq.size(); ++i) {
    auto const stop_idx =
        static_cast<stop_idx_t>(kFwd ? i : loc_seq.size() - i - 1U);
    auto const stp = stop{loc_seq[stop_idx]};
    auto const loc = stp.location_idx();

    if (first.has_value()) {
      if (end_locs.contains(loc) && stp.can_finish<Dir>(is_wheelchair)) {
        fn(*first, stop_idx);
        first = std::nullopt;
      }
    } else if (start_locs.contains(loc) && stp.can_start<Dir>(is_wheelchair)) {
      first = stop_idx;
    }
  }
}

}  // namespace

template <direction Dir>
utl::generator<std::array<journey::leg, 3>> get_direct_journeys(
    timetable const& tt,
    rt_timetable const* rtt,
    query const& q_in,
    unixtime_t const time) {
  auto const q = q_in;
  constexpr auto kFwd = Dir == direction::kForward;

  auto const merge_sorted = [](auto& dst, auto const& src) {
    auto const original_size = static_cast<int>(dst.size());
    dst.resize(dst.size() + src.size());
    std::copy(begin(src), end(src), begin(dst) + original_size);
    std::inplace_merge(begin(dst), begin(dst) + original_size, end(dst));
    dst.erase(std::unique(begin(dst), end(dst)), end(dst));
  };

  // Storage for generators and their current head.
  auto gens = std::vector<utl::generator<std::array<journey::leg, 3>>>{};
  auto heads = std::vector<std::array<journey::leg, 3>>{};
  auto const add_gen = [&](utl::generator<std::array<journey::leg, 3>> g) {
    if (g) {
      heads.emplace_back(g());
      gens.emplace_back(std::move(g));
    }
  };

  // Union of all locations on each side (static offsets + td_offsets keys).
  auto const collect_locs = [](std::vector<offset> const& static_offs,
                               td_offsets_t const& td_offs) {
    auto dst = hash_set<location_idx_t>{};
    for (auto const& o : static_offs) {
      dst.emplace(o.target());
    }
    for (auto const& [loc, _] : td_offs) {
      dst.emplace(loc);
    }
    return dst;
  };
  auto const start_locs = collect_locs(q.start_, q.td_start_);
  auto const end_locs = collect_locs(q.destination_, q.td_dest_);

  // ==============================
  // Collect route_idx_t generators
  // ------------------------------
  auto from_routes = std::vector<route_idx_t>{};
  auto to_routes = std::vector<route_idx_t>{};
  for (auto const loc : start_locs) {
    merge_sorted(from_routes, tt.location_routes_[loc]);
  }
  for (auto const loc : end_locs) {
    merge_sorted(to_routes, tt.location_routes_[loc]);
  }
  utl::sorted_diff(
      from_routes, to_routes, std::less<route_idx_t>{},
      [](auto&&, auto&&) { return false; },
      utl::overloaded{
          [](utl::op, route_idx_t) {},
          [&](route_idx_t const r, route_idx_t) {
            if (!is_allowed(q.allowed_claszes_, tt.route_clasz_[r]) ||
                (q.require_bike_transport_ && !tt.has_bike_transport(r)) ||
                (q.require_car_transport_ && !tt.has_car_transport(r))) {
              return;
            }

            auto const pseudo_fr = rt::frun{
                tt, rtt,
                rt::run{.t_ = transport{tt.route_transport_ranges_[r].from_,
                                        day_idx_t{0}},
                        .stop_range_ = {
                            0U, static_cast<stop_idx_t>(
                                    tt.route_location_seq_[r].size())}}};
            for_each_pair<Dir>(
                tt.route_location_seq_[r], start_locs, end_locs, q.prf_idx_,
                [&](stop_idx_t const start_idx, stop_idx_t const end_idx) {
                  if (sections_violate_constraints(pseudo_fr,
                                                   std::min(start_idx, end_idx),
                                                   std::max(start_idx, end_idx),
                                                   q.require_bike_transport_,
                                                   q.require_car_transport_)) {
                    return;
                  }
                  add_gen(
                      route_gen<Dir>(tt, rtt, r, start_idx, end_idx, q, time));
                });
          }});

  // =====================================
  // Collect rt_transport_idx_t generators
  // -------------------------------------
  if (rtt != nullptr) {
    auto from_rt = std::vector<rt_transport_idx_t>{};
    auto to_rt = std::vector<rt_transport_idx_t>{};
    for (auto const loc : start_locs) {
      merge_sorted(from_rt, rtt->location_rt_transports_[loc]);
    }
    for (auto const loc : end_locs) {
      merge_sorted(to_rt, rtt->location_rt_transports_[loc]);
    }
    utl::sorted_diff(
        from_rt, to_rt, std::less<rt_transport_idx_t>{},
        [](auto&&, auto&&) { return false; },
        utl::overloaded{
            [](utl::op, rt_transport_idx_t) {},
            [&](rt_transport_idx_t const x, rt_transport_idx_t) {
              if (!is_allowed(q.allowed_claszes_,
                              rtt->rt_transport_section_clasz_[x].front()) ||
                  (q.require_bike_transport_ && !rtt->has_bike_transport(x)) ||
                  (q.require_car_transport_ && !rtt->has_car_transport(x))) {
                return;
              }
              auto const rt_probe = rt::frun{
                  tt, rtt,
                  rt::run{
                      .stop_range_ =
                          {0U, static_cast<stop_idx_t>(
                                   rtt->rt_transport_location_seq_[x].size())},
                      .rt_ = x}};
              for_each_pair<Dir>(
                  rtt->rt_transport_location_seq_[x], start_locs, end_locs,
                  q.prf_idx_,
                  [&](stop_idx_t const start_idx, stop_idx_t const end_idx) {
                    if (sections_violate_constraints(
                            rt_probe, std::min(start_idx, end_idx),
                            std::max(start_idx, end_idx),
                            q.require_bike_transport_,
                            q.require_car_transport_)) {
                      return;
                    }
                    add_gen(
                        rt_gen<Dir>(tt, *rtt, x, start_idx, end_idx, q, time));
                  });
            }});
  }

  // ==========================
  // Iterate through generators
  // --------------------------
  auto const cmp = [&](std::size_t const a, std::size_t const b) {
    return kFwd ? heads[a][2].arr_time_ > heads[b][2].arr_time_
                : heads[a][0].dep_time_ < heads[b][0].dep_time_;
  };
  auto heap =
      std::priority_queue<std::size_t, std::vector<std::size_t>, decltype(cmp)>{
          cmp};
  for (auto i = std::size_t{0}; i != gens.size(); ++i) {
    heap.push(i);
  }

  while (!heap.empty()) {
    auto const idx = heap.top();
    heap.pop();
    co_yield std::move(heads[idx]);
    if (gens[idx]) {
      heads[idx] = gens[idx]();
      heap.push(idx);
    }
  }
}

template <direction Dir>
void enrich_with_slow_direct(timetable const& tt,
                             rt_timetable const* rtt,
                             query const& q,
                             interval<unixtime_t> const& time,
                             pareto_set<journey>& results) {
  constexpr auto kFwd = Dir == direction::kForward;
  if (!q.slow_direct_) {
    return;
  }
  auto direct = std::vector<journey>{};
  auto shortest_duration = duration_t::max();

  auto const time_threshold = kFwd ? time.from_ : time.to_;
  for (auto&& legs : get_direct_journeys<Dir>(tt, rtt, q, time_threshold)) {
    auto const t_check = kFwd ? legs[0].dep_time_ : legs[2].arr_time_;
    if (!time.contains(t_check)) {
      if (kFwd ? t_check >= time.to_ : t_check < time.from_) {
        break;
      }
      continue;
    }

    auto j = journey{};
    j.start_time_ = kFwd ? legs[0].dep_time_ : legs[2].arr_time_;
    j.dest_time_ = kFwd ? legs[2].arr_time_ : legs[0].dep_time_;
    for (auto& leg : legs) {
      j.legs_.push_back(std::move(leg));
    }
    j.dest_ = j.legs_.back().to_;

    if (j.travel_time() < shortest_duration) {
      shortest_duration = j.travel_time();
    }
    direct.push_back(std::move(j));
  }

  if (q.fastest_slow_direct_factor_ >= 1.0) {
    utl::erase_if(direct, [&](journey const& j) {
      return j.travel_time() >
             shortest_duration * q.fastest_slow_direct_factor_;
    });
  }

  utl::concat(results.els_, direct);
  utl::erase_duplicates(results);
}

template void enrich_with_slow_direct<direction::kForward>(
    timetable const&,
    rt_timetable const*,
    query const&,
    interval<unixtime_t> const&,
    pareto_set<journey>&);

template void enrich_with_slow_direct<direction::kBackward>(
    timetable const&,
    rt_timetable const*,
    query const&,
    interval<unixtime_t> const&,
    pareto_set<journey>&);

}  // namespace nigiri::routing
