#include "nigiri/routing/direct.h"

#include <queue>

#include "utl/concat.h"
#include "utl/erase_duplicates.h"
#include "utl/erase_if.h"
#include "utl/overloaded.h"
#include "utl/sorted_diff.h"

#include "nigiri/for_each_meta.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/special_stations.h"
#include "nigiri/td_footpath.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

std::optional<journey::leg> lookup_offset(location_idx_t const loc,
                                          unixtime_t const t,
                                          side const s,
                                          std::vector<offset> const& offsets,
                                          td_offsets_t const& td_offsets) {
  auto const is_boarding = s == side::kBoarding;
  auto const td_search_dir =
      is_boarding ? direction::kBackward : direction::kForward;

  auto const make_leg = [&](duration_t const dur,
                            transport_mode_id_t const mode_id) {
    auto const boundary = get_special_station(
        is_boarding ? special_station::kStart : special_station::kEnd);
    auto const dep = is_boarding ? t - dur : t;
    auto const arr = is_boarding ? t : t + dur;
    auto const from = is_boarding ? boundary : loc;
    auto const to = is_boarding ? loc : boundary;
    return journey::leg{direction::kForward,      from, to, dep, arr,
                        offset{loc, dur, mode_id}};
  };

  // Time-dependend offsets take precedence.
  if (auto const it = td_offsets.find(loc); it != end(td_offsets)) {
    auto const td = get_td_duration(td_search_dir, it->second, t);
    if (!td.has_value() || td->first >= footpath::kMaxDuration) {
      return std::nullopt;
    }
    return std::optional{make_leg(td->first, td->second.transport_mode_id_)};
  }

  // Search for shortest offset.
  auto best = std::optional<offset>{};
  for (auto const& o : offsets) {
    if (o.target() == loc &&
        (!best.has_value() || o.duration() < best->duration())) {
      best = o;
    }
  }

  return best.transform([&](offset const& o) {
    return make_leg(o.duration(), o.transport_mode_id_);
  });
}

std::optional<journey::leg> lookup_footpath(location_idx_t const loc,
                                            unixtime_t const t,
                                            side const s,
                                            timetable const& tt,
                                            rt_timetable const* rtt,
                                            query const& q,
                                            std::vector<offset> const& offs,
                                            location_match_mode const mode,
                                            bool const use_footpaths) {
  auto const is_boarding = s == side::kBoarding;
  auto const td_search_dir =
      is_boarding ? direction::kBackward : direction::kForward;

  auto best_dur = footpath::kMaxDuration;
  auto best_source = location_idx_t{};

  auto const has_td_arr = rtt == nullptr
                              ? nullptr
                              : (is_boarding ? &rtt->has_td_footpaths_out_
                                             : &rtt->has_td_footpaths_in_);
  auto const td_fps_arr =
      rtt == nullptr
          ? nullptr
          : (is_boarding ? &rtt->td_footpaths_out_ : &rtt->td_footpaths_in_);

  for (auto const& o : offs) {
    auto const o_duration = o.duration();
    for_each_meta(tt, mode, o.target(), [&](location_idx_t const l) {
      // Direct match - boarding/alighting at the input loc itself.
      if (l == loc && o_duration < best_dur) {
        best_dur =
            o_duration +
            (mode == location_match_mode::kExact
                 // kExact is used for transfers
                 // -> respect reflexive transfer time
                 ? adjusted_transfer_time(q.transfer_time_settings_,
                                          tt.locations_.transfer_time_[l])
                 // kIntermodal / kEquivalent: no transfer time access/egress
                 : u8_minutes{0});
        best_source = o.target();
      }

      if (!use_footpaths) {
        return;
      }

      auto eff_dur = footpath::kMaxDuration;
      if (has_td_arr != nullptr && q.prf_idx_ < has_td_arr->size() &&
          to_idx(l) < (*has_td_arr)[q.prf_idx_].size() &&
          (*has_td_arr)[q.prf_idx_][l]) {
        // td footpaths take precedence
        for_each_footpath(td_search_dir, (*td_fps_arr)[q.prf_idx_][l], t,
                          [&](footpath const fp) {
                            if (fp.target() == loc && fp.duration() < eff_dur) {
                              eff_dur = fp.duration();
                            }
                          });
      } else {
        // no td footpath -> take shortest regular footpath
        auto const& fps = is_boarding
                              ? tt.locations_.footpaths_out_[q.prf_idx_][l]
                              : tt.locations_.footpaths_in_[q.prf_idx_][l];
        for (auto const& fp : fps) {
          if (fp.target() != loc) {
            continue;
          }
          auto const adj =
              adjusted_transfer_time(q.transfer_time_settings_, fp.duration());
          if (adj < eff_dur) {
            eff_dur = adj;
          }
        }
      }

      if (eff_dur >= footpath::kMaxDuration) {
        return;
      }
      auto const total = o_duration + eff_dur;
      if (total < best_dur) {
        best_dur = total;
        best_source = o.target();
      }
    });
  }

  if (best_dur >= footpath::kMaxDuration) {
    return std::nullopt;
  }

  auto const dep = is_boarding ? t - best_dur : t;
  auto const arr = is_boarding ? t : t + best_dur;
  auto const from = is_boarding ? best_source : loc;
  auto const to = is_boarding ? loc : best_source;
  return journey::leg{direction::kForward,   from, to, dep, arr,
                      footpath{to, best_dur}};
}

namespace {

std::optional<journey::leg> lookup_access(query const& q,
                                          timetable const& tt,
                                          rt_timetable const* rtt,
                                          location_idx_t const loc,
                                          unixtime_t const t,
                                          side const s) {
  auto const is_boarding = s == side::kBoarding;
  auto const& offs = is_boarding ? q.start_ : q.destination_;
  auto const& td_offs = is_boarding ? q.td_start_ : q.td_dest_;
  auto const mode = is_boarding ? q.start_match_mode_ : q.dest_match_mode_;

  if (mode == location_match_mode::kIntermodal) {
    return lookup_offset(loc, t, s, offs, td_offs);
  }

  auto const use_footpaths = is_boarding ? q.use_start_footpaths_ : true;
  return lookup_footpath(loc, t, s, tt, rtt, q, offs, mode, use_footpaths);
}

hash_set<location_idx_t> collect_locations(timetable const& tt,
                                           rt_timetable const* rtt,
                                           query const& q,
                                           side const s) {
  auto const is_boarding = s == side::kBoarding;
  auto const& offsets = is_boarding ? q.start_ : q.destination_;
  auto const& td_offsets = is_boarding ? q.td_start_ : q.td_dest_;
  auto const mode = is_boarding ? q.start_match_mode_ : q.dest_match_mode_;
  auto const use_footpaths = is_boarding ? q.use_start_footpaths_ : true;

  auto locs = hash_set<location_idx_t>{};

  if (mode == location_match_mode::kIntermodal) {
    for (auto const& o : offsets) {
      locs.insert(o.target());
    }
  } else {
    auto const has_td_arr = rtt == nullptr
                                ? nullptr
                                : (is_boarding ? &rtt->has_td_footpaths_out_
                                               : &rtt->has_td_footpaths_in_);
    auto const td_fps_arr =
        rtt == nullptr
            ? nullptr
            : (is_boarding ? &rtt->td_footpaths_out_ : &rtt->td_footpaths_in_);
    for (auto const& o : offsets) {
      for_each_meta(tt, mode, o.target(), [&](location_idx_t const l) {
        locs.insert(l);
        if (!use_footpaths) {
          return;
        }

        auto const& fps = is_boarding
                              ? tt.locations_.footpaths_out_[q.prf_idx_][l]
                              : tt.locations_.footpaths_in_[q.prf_idx_][l];
        for (auto const& fp : fps) {
          locs.insert(fp.target());
        }

        if (has_td_arr == nullptr || q.prf_idx_ >= has_td_arr->size() ||
            to_idx(l) >= (*has_td_arr)[q.prf_idx_].size() ||
            !(*has_td_arr)[q.prf_idx_][l]) {
          return;
        }

        for (auto const& tdfp : (*td_fps_arr)[q.prf_idx_][l]) {
          locs.insert(tdfp.target_);
        }
      });
    }
  }

  for (auto const& [loc, _] : td_offsets) {
    locs.insert(loc);
  }

  return locs;
}

bool sections_violate_constraints(rt::frun const& fr,
                                  unsigned const from_section_idx,
                                  unsigned const to_section_idx,
                                  bool const require_bike,
                                  bool const require_car,
                                  bool const is_wheelchair) {
  if (!require_bike && !require_car && !is_wheelchair) {
    return false;
  }
  for (auto i = from_section_idx; i != to_section_idx; ++i) {
    auto const section_start = static_cast<stop_idx_t>(i);
    if (require_bike && !fr[section_start].bikes_allowed(event_type::kDep)) {
      return true;
    }
    if (require_car && !fr[section_start].cars_allowed(event_type::kDep)) {
      return true;
    }
    if (is_wheelchair &&
        !fr[section_start].wheelchair_accessible(event_type::kDep)) {
      return true;
    }
  }
  return false;
}

std::vector<journey::leg> assemble_legs(journey::leg const& boarding_walk,
                                        journey::leg&& transit,
                                        journey::leg const& alighting_walk) {
  auto const drop_at_boundary = [](journey::leg const& l) {
    return std::holds_alternative<offset>(l.uses_) &&
           l.dep_time_ == l.arr_time_;
  };
  auto legs = std::vector<journey::leg>{};
  legs.reserve(3);
  if (!drop_at_boundary(boarding_walk)) {
    legs.push_back(boarding_walk);
  }
  legs.push_back(std::move(transit));
  if (!drop_at_boundary(alighting_walk)) {
    legs.push_back(alighting_walk);
  }
  return legs;
}

template <direction Dir>
utl::generator<std::vector<journey::leg>> route_gen(
    timetable const& tt,
    rt_timetable const* rtt,
    route_idx_t const r,
    stop_idx_t const boarding_idx,
    stop_idx_t const alighting_idx,
    query const& q,
    unixtime_t const time) {
  constexpr auto kFwd = Dir == direction::kForward;

  auto day_int = static_cast<int>(to_idx(tt.day_idx_mam(time).first));
  auto const events =
      kFwd ? tt.event_times_at_stop(r, boarding_idx, event_type::kDep)
           : tt.event_times_at_stop(r, alighting_idx, event_type::kArr);
  auto const n_transports = tt.route_transport_ranges_[r].size();
  auto const loc_seq = tt.route_location_seq_[r];
  auto const boarding_loc = stop{loc_seq[boarding_idx]}.location_idx();
  auto const alighting_loc = stop{loc_seq[alighting_idx]}.location_idx();

  auto const day_lo =
      static_cast<int>(to_idx(tt.day_idx(tt.internal_interval_days().from_)));
  auto const day_hi =
      static_cast<int>(to_idx(tt.day_idx(tt.internal_interval_days().to_)));

  while (kFwd ? day_int <= day_hi : day_int >= day_lo) {
    auto const day = day_idx_t{static_cast<day_idx_t::value_t>(day_int)};
    auto const route_active = tt.is_route_active(r, day);
    for (auto t_offset = 0U; route_active && t_offset < n_transports;
         ++t_offset) {
      auto const idx = kFwd ? t_offset : (n_transports - 1U - t_offset);
      auto const ev = events[idx];
      auto const t = tt.route_transport_ranges_[r][idx];
      auto const ev_day_offset = ev.days();
      auto const day_off = static_cast<int>(to_idx(day)) - ev_day_offset;
      if (day_off < 0) {
        continue;
      }
      auto const start_day =
          day_idx_t{static_cast<day_idx_t::value_t>(day_off)};
      auto const& bitfields = rtt != nullptr ? rtt->bitfields_ : tt.bitfields_;
      auto const traffic_idx = rtt != nullptr ? rtt->transport_traffic_days_[t]
                                              : tt.transport_traffic_days_[t];
      if (to_idx(start_day) >= bitfields[traffic_idx].size()) {
        continue;
      }
      auto const transport_active = rtt != nullptr
                                        ? rtt->is_transport_active(t, start_day)
                                        : tt.is_transport_active(t, start_day);
      if (!transport_active) {
        continue;
      }

      auto const tr = transport{t, start_day};
      auto const boarding_time =
          tt.event_time(tr, boarding_idx, event_type::kDep);
      auto const alighting_time =
          tt.event_time(tr, alighting_idx, event_type::kArr);

      auto const boarding_walk = lookup_access(q, tt, rtt, boarding_loc,
                                               boarding_time, side::kBoarding);
      auto const alighting_walk = lookup_access(
          q, tt, rtt, alighting_loc, alighting_time, side::kAlighting);
      if (!boarding_walk.has_value() || !alighting_walk.has_value()) {
        continue;
      }

      if (kFwd ? boarding_walk->dep_time_ < time
               : alighting_walk->arr_time_ > time) {
        continue;
      }

      auto transit = journey::leg{
          direction::kForward,
          boarding_loc,
          alighting_loc,
          boarding_time,
          alighting_time,
          journey::run_enter_exit{
              rt::frun{tt, rtt,
                       rt::run{.t_ = tr,
                               .stop_range_ = {0U, static_cast<stop_idx_t>(
                                                       loc_seq.size())}}},
              boarding_idx, alighting_idx}};

      co_yield assemble_legs(*boarding_walk, std::move(transit),
                             *alighting_walk);
    }

    if constexpr (kFwd) {
      ++day_int;
    } else {
      --day_int;
    }
  }
}

template <direction Dir>
utl::generator<std::vector<journey::leg>> rt_gen(
    timetable const& tt,
    rt_timetable const& rtt,
    rt_transport_idx_t const rt_idx,
    stop_idx_t const boarding_idx,
    stop_idx_t const alighting_idx,
    query const& q,
    unixtime_t const time) {
  constexpr auto kFwd = Dir == direction::kForward;
  auto const boarding_time =
      rtt.unix_event_time(rt_idx, boarding_idx, event_type::kDep);
  auto const alighting_time =
      rtt.unix_event_time(rt_idx, alighting_idx, event_type::kArr);
  auto const loc_seq = rtt.rt_transport_location_seq_[rt_idx];
  auto const boarding_loc = stop{loc_seq[boarding_idx]}.location_idx();
  auto const alighting_loc = stop{loc_seq[alighting_idx]}.location_idx();

  auto const boarding_walk =
      lookup_access(q, tt, &rtt, boarding_loc, boarding_time, side::kBoarding);
  if (!boarding_walk.has_value()) {
    co_return;
  }

  auto const alighting_walk = lookup_access(q, tt, &rtt, alighting_loc,
                                            alighting_time, side::kAlighting);
  if (!alighting_walk.has_value()) {
    co_return;
  }

  if (kFwd ? boarding_walk->dep_time_ < time
           : alighting_walk->arr_time_ > time) {
    co_return;
  }

  auto transit = journey::leg{
      direction::kForward,
      boarding_loc,
      alighting_loc,
      boarding_time,
      alighting_time,
      journey::run_enter_exit{
          rt::frun{tt, &rtt,
                   rt::run{.stop_range_ = {0U, static_cast<stop_idx_t>(
                                                   loc_seq.size())},
                           .rt_ = rt_idx}},
          boarding_idx, alighting_idx}};

  co_yield assemble_legs(*boarding_walk, std::move(transit), *alighting_walk);
}

template <typename LocSeq, typename Fn>
void for_each_pair(LocSeq const& loc_seq,
                   hash_set<location_idx_t> const& boarding_locs,
                   hash_set<location_idx_t> const& alighting_locs,
                   profile_idx_t const prf_idx,
                   Fn&& fn) {
  auto boarding_stop_idx = std::optional<stop_idx_t>{};
  for (auto i = stop_idx_t{0U}; i != static_cast<stop_idx_t>(loc_seq.size());
       ++i) {
    auto const stp = stop{loc_seq[i]};
    auto const loc = stp.location_idx();

    if (boarding_stop_idx.has_value()) {
      if (alighting_locs.contains(loc) && stp.out_allowed(prf_idx)) {
        fn(*boarding_stop_idx, i);
        boarding_stop_idx = std::nullopt;
      }
    } else if (boarding_locs.contains(loc) && stp.in_allowed(prf_idx)) {
      boarding_stop_idx = i;
    }
  }
}

}  // namespace

template <direction Dir>
utl::generator<std::vector<journey::leg>> get_direct_journeys(
    timetable const& tt,
    rt_timetable const* rtt,
    query const& q_in,
    unixtime_t const time) {
  auto const q = q_in;
  constexpr auto kFwd = Dir == direction::kForward;
  bool const is_wheelchair = q.prf_idx_ == kWheelchairProfile;

  auto const merge_sorted = [](auto& dst, auto const& src) {
    auto const original_size = static_cast<int>(dst.size());
    dst.resize(dst.size() + src.size());
    std::copy(begin(src), end(src), begin(dst) + original_size);
    std::inplace_merge(begin(dst), begin(dst) + original_size, end(dst));
    dst.erase(std::unique(begin(dst), end(dst)), end(dst));
  };

  // Storage for generators and their current head.
  auto gens = std::vector<utl::generator<std::vector<journey::leg>>>{};
  auto heads = std::vector<std::vector<journey::leg>>{};
  auto const add_gen = [&](utl::generator<std::vector<journey::leg>>&& g) {
    if (g) {
      heads.emplace_back(g());
      gens.emplace_back(std::move(g));
    }
  };

  auto const boarding_locs = collect_locations(tt, rtt, q, side::kBoarding);
  auto const alighting_locs = collect_locations(tt, rtt, q, side::kAlighting);

  // ==============================
  // Collect route_idx_t generators
  // ------------------------------
  auto from_routes = std::vector<route_idx_t>{};
  auto to_routes = std::vector<route_idx_t>{};
  for (auto const loc : boarding_locs) {
    merge_sorted(from_routes, tt.location_routes_[loc]);
  }
  for (auto const loc : alighting_locs) {
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
                (q.require_car_transport_ && !tt.has_car_transport(r)) ||
                (is_wheelchair && !tt.has_wheelchair_transport(r))) {
              return;
            }

            auto const fr = rt::frun{
                tt, rtt,
                rt::run{.t_ = transport{tt.route_transport_ranges_[r].from_,
                                        day_idx_t{0}},
                        .stop_range_ = {
                            0U, static_cast<stop_idx_t>(
                                    tt.route_location_seq_[r].size())}}};
            for_each_pair(tt.route_location_seq_[r], boarding_locs,
                          alighting_locs, q.prf_idx_,
                          [&](stop_idx_t const a, stop_idx_t const b) {
                            if (sections_violate_constraints(
                                    fr, a, b, q.require_bike_transport_,
                                    q.require_car_transport_, is_wheelchair)) {
                              return;
                            }
                            add_gen(route_gen<Dir>(tt, rtt, r, a, b, q, time));
                          });
          }});

  // =====================================
  // Collect rt_transport_idx_t generators
  // -------------------------------------
  if (rtt != nullptr) {
    auto from_rt = std::vector<rt_transport_idx_t>{};
    auto to_rt = std::vector<rt_transport_idx_t>{};
    for (auto const loc : boarding_locs) {
      merge_sorted(from_rt, rtt->location_rt_transports_[loc]);
    }
    for (auto const loc : alighting_locs) {
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
                  (q.require_car_transport_ && !rtt->has_car_transport(x)) ||
                  (is_wheelchair && !rtt->has_wheelchair_transport(x))) {
                return;
              }

              auto const fr = rt::frun{
                  tt, rtt,
                  rt::run{
                      .stop_range_ =
                          {0U, static_cast<stop_idx_t>(
                                   rtt->rt_transport_location_seq_[x].size())},
                      .rt_ = x}};
              for_each_pair(
                  rtt->rt_transport_location_seq_[x], boarding_locs,
                  alighting_locs, q.prf_idx_,
                  [&](stop_idx_t const a, stop_idx_t const b) {
                    if (sections_violate_constraints(
                            fr, a, b, q.require_bike_transport_,
                            q.require_car_transport_, is_wheelchair)) {
                      return;
                    }
                    add_gen(rt_gen<Dir>(tt, *rtt, x, a, b, q, time));
                  });
            }});
  }

  // ==========================
  // Iterate through generators
  // --------------------------
  auto const cmp = [&](std::size_t const a, std::size_t const b) {
    return kFwd ? heads[a].back().arr_time_ > heads[b].back().arr_time_
                : heads[a].front().dep_time_ < heads[b].front().dep_time_;
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
    auto const t_check = kFwd ? legs.front().dep_time_ : legs.back().arr_time_;
    if (!time.contains(t_check)) {
      if (kFwd ? t_check >= time.to_ : t_check < time.from_) {
        break;
      }
      continue;
    }

    auto j = journey{};
    j.start_time_ = kFwd ? legs.front().dep_time_ : legs.back().arr_time_;
    j.dest_time_ = kFwd ? legs.back().arr_time_ : legs.front().dep_time_;
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

template utl::generator<std::vector<journey::leg>>
get_direct_journeys<direction::kForward>(timetable const&,
                                         rt_timetable const*,
                                         query const&,
                                         unixtime_t);

template utl::generator<std::vector<journey::leg>>
get_direct_journeys<direction::kBackward>(timetable const&,
                                          rt_timetable const*,
                                          query const&,
                                          unixtime_t);

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
