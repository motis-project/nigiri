#include "nigiri/routing/direct.h"

#include "utl/sorted_diff.h"

#include "nigiri/for_each_meta.h"
#include "nigiri/location_match_mode.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

#define trace_direct(...)

void get_direct(timetable const& tt,
                rt_timetable const* rtt,
                location_idx_t const from,
                location_idx_t const to,
                query const& q,
                interval<unixtime_t> const time,
                direction const search_dir,
                hash_set<std::pair<location_idx_t, location_idx_t>>& done,
                std::vector<journey>& direct) {
  auto const fwd = search_dir == direction::kForward;
  auto const start_ev_type = fwd ? event_type::kDep : event_type::kArr;
  auto const end_ev_type = fwd ? event_type::kArr : event_type::kDep;

  auto const get_offset = [](direction const offset_dir, location_idx_t const l,
                             std::vector<offset> const& offsets,
                             td_offsets_t const& td_offsets,
                             unixtime_t const t) -> std::optional<offset> {
    {  // Try to find normal offset.
      auto const it = utl::find_if(
          offsets, [&](offset const& o) { return o.target() == l; });
      if (it != end(offsets)) {
        return *it;
      }
    }

    {  // Try to find td-offset.
      auto const it = td_offsets.find(l);
      if (it != end(td_offsets)) {
        auto const d = get_td_duration(offset_dir, it->second, t);
        if (d.has_value()) {
          auto const& [duration, o] = *d;
          return offset{l, duration, o.transport_mode_id_};
        }
      }
    }

    return std::nullopt;
  };

  auto const add_offsets = [&](journey::leg const& l) {
    auto start_leg = std::optional<journey::leg>{};
    auto start_time = fwd ? l.dep_time_ : l.arr_time_;
    if (q.start_match_mode_ == location_match_mode::kIntermodal) {
      auto const offset_leg_start = fwd ? l.from_ : l.to_;
      auto const offset_leg_start_time = start_time;
      auto const start_offset =
          get_offset(flip(search_dir), offset_leg_start, q.start_, q.td_start_,
                     offset_leg_start_time);
      if (!start_offset.has_value()) {
        return;
      }
      start_time -= (fwd ? 1 : -1) * start_offset->duration();
      start_leg = journey::leg{flip(search_dir),
                               offset_leg_start,
                               get_special_station(special_station::kStart),
                               offset_leg_start_time,
                               start_time,
                               *start_offset};
    }

    if (!time.contains(start_time)) {
      return;
    }

    auto dest_leg = std::optional<journey::leg>{};
    auto dest_time = fwd ? l.arr_time_ : l.dep_time_;
    if (q.dest_match_mode_ == location_match_mode::kIntermodal) {
      auto const offset_leg_start = fwd ? l.to_ : l.from_;
      auto const offset_leg_start_time = dest_time;
      auto const dest_offset =
          get_offset(search_dir, offset_leg_start, q.destination_, q.td_dest_,
                     offset_leg_start_time);
      if (!dest_offset.has_value()) {
        return;
      }
      dest_time += (fwd ? 1 : -1) * dest_offset->duration();
      dest_leg = journey::leg{search_dir,
                              offset_leg_start,
                              get_special_station(special_station::kEnd),
                              offset_leg_start_time,
                              dest_time,
                              *dest_offset};
    }

    auto j = journey{.start_time_ = start_time, .dest_time_ = dest_time};
    if (start_leg.has_value()) {
      j.legs_.push_back(*start_leg);
    }
    j.legs_.push_back(l);
    if (dest_leg.has_value()) {
      j.legs_.push_back(*dest_leg);
    }
    j.dest_ = fwd ? j.legs_.back().to_ : j.legs_.back().from_;

    if (!fwd) {
      std::reverse(begin(j.legs_), end(j.legs_));
    }

    direct.push_back(std::move(j));
  };

  auto const checked = [&](journey::leg const& l) {
    auto const& ree = std::get<journey::run_enter_exit>(l.uses_);
    if (q.require_bike_transport_) {
      auto const fr = rt::frun{tt, rtt, ree.r_};
      for (auto const stop_idx : ree.stop_range_) {
        if (!fr[stop_idx].bikes_allowed()) {
          return;
        }
      }
    }

    if (q.require_car_transport_) {
      auto const fr = rt::frun{tt, rtt, ree.r_};
      for (auto const stop_idx : ree.stop_range_) {
        if (!fr[stop_idx].cars_allowed()) {
          return;
        }
      }
    }

    add_offsets(l);
  };

  auto const check_interval = utl::overloaded{
      [&](route_idx_t const r, stop_idx_t const start_stop_idx,
          stop_idx_t const end_stop_idx) {
        auto const is_transport_active = [&](transport_idx_t const t,
                                             std::size_t const day) {
          if (rtt != nullptr) {
            return rtt->bitfields_[rtt->transport_traffic_days_[t]].test(day);
          } else {
            return tt.bitfields_[tt.transport_traffic_days_[t]].test(day);
          }
        };

        auto const start_events =
            tt.event_times_at_stop(r, start_stop_idx, start_ev_type);
        auto const loc_seq = tt.route_location_seq_[r];
        auto const days = interval{
            std::chrono::time_point_cast<date::days>(time.from_),
            std::chrono::time_point_cast<date::days>(time.to_) + date::days{1}};
        auto const day_indices =
            interval{tt.day_idx(days.from_), tt.day_idx(days.to_)};

        trace_direct("    check_interval [{}, {}]: days={}", start_stop_idx,
                     end_stop_idx, days);
        for (auto const day : day_indices) {
          for (auto it = begin(start_events); it != end(start_events); ++it) {
            auto const ev = *it;
            auto const t_offset =
                static_cast<std::size_t>(&*it - start_events.data());
            auto const t = tt.route_transport_ranges_[r][t_offset];
            auto const ev_day_offset = ev.days();
            auto const start_day =
                static_cast<std::size_t>(to_idx(day) - ev_day_offset);
            if (!is_transport_active(t, start_day)) {
              trace_direct(
                  "      transport {} not inactive on {} (ev_time={})",
                  tt.transport_name(t),
                  tt.internal_interval_days().from_ + date::days{1} * start_day,
                  ev);
              continue;
            }

            trace_direct(
                "      transport {} operates on {} (ev_time={})",
                tt.transport_name(t),
                tt.internal_interval_days().from_ + date::days{1} * start_day,
                ev);
            auto const tr = transport{t, day_idx_t{start_day}};
            auto const start_time =
                tt.event_time(tr, start_stop_idx, start_ev_type);
            auto const end_time = tt.event_time(tr, end_stop_idx, end_ev_type);
            trace_direct("        time={} contains {}", time, start_time);
            checked(journey::leg{
                search_dir, stop{loc_seq[start_stop_idx]}.location_idx(),
                stop{loc_seq[end_stop_idx]}.location_idx(), start_time,
                end_time,
                journey::run_enter_exit{
                    rt::frun{
                        tt, rtt,
                        rt::run{.t_ = tr,
                                .stop_range_ = {0U, static_cast<stop_idx_t>(
                                                        loc_seq.size())}}},
                    start_stop_idx, end_stop_idx}});
          }
        }
      },
      [&](rt_transport_idx_t const rt, stop_idx_t const start,
          stop_idx_t const end) {
        auto const start_time = rtt->unix_event_time(rt, start, start_ev_type);
        auto const end_time = rtt->unix_event_time(rt, end, end_ev_type);
        auto const loc_seq = rtt->rt_transport_location_seq_[rt];
        checked(journey::leg{
            search_dir, stop{loc_seq[start]}.location_idx(),
            stop{loc_seq[end]}.location_idx(), start_time, end_time,
            journey::run_enter_exit{
                rt::frun{tt, rtt,
                         rt::run{.stop_range_ = {0U, static_cast<stop_idx_t>(
                                                         loc_seq.size())},
                                 .rt_ = rt}},
                start, end}});
      }};

  auto const for_each_from_to = [&](location_idx_t const x,
                                    location_idx_t const y, auto&& loc_seq,
                                    auto&& r) {
    auto first = std::optional<stop_idx_t>{};
    for (auto i = 0U; i != loc_seq.size(); ++i) {
      auto const stop_idx = static_cast<stop_idx_t>(
          search_dir == direction::kForward ? i : loc_seq.size() - i - 1U);
      auto const stp = stop{loc_seq[stop_idx]}.location_idx();

      if (first.has_value()) {
        if (stp == y) {
          check_interval(r, *first, stop_idx);
        }
      } else if (stp == x) {
        first = stop_idx;
      }
    }
  };

  trace_direct("direct from {} to {} in {}", location{tt, from},
               location{tt, to}, time);
  for_each_meta(
      tt, location_match_mode::kEquivalent, from, [&](location_idx_t const x) {
        for_each_meta(
            tt, location_match_mode::kEquivalent, to,
            [&](location_idx_t const y) {
              if (x == y || !done.emplace(x, y).second) {
                return;
              }

              assert(utl::is_sorted(tt.location_routes_[x], std::less<>{}) &&
                     utl::is_sorted(tt.location_routes_[y], std::less<>{}));
              utl::sorted_diff(
                  tt.location_routes_[x], tt.location_routes_[y],
                  std::less<route_idx_t>{},
                  [](auto&&, auto&&) { return false; },
                  utl::overloaded{
                      [](utl::op, route_idx_t) {},
                      [&](route_idx_t const a, route_idx_t const b) {
                        utl::verify(a == b, "{} != {}", a, b);
                        trace_direct("  found route {} visiting", a);
                        if (is_allowed(q.allowed_claszes_,
                                       tt.route_clasz_[a])) {
                          for_each_from_to(x, y, tt.route_location_seq_[a], a);
                        }
                      }});

              if (rtt == nullptr) {
                return;
              }

              assert(utl::is_sorted(rtt->location_rt_transports_[x],
                                    std::less<>{}) &&
                     utl::is_sorted(rtt->location_rt_transports_[y],
                                    std::less<>{}));
              utl::sorted_diff(
                  rtt->location_rt_transports_[x],
                  rtt->location_rt_transports_[y],
                  std::less<rt_transport_idx_t>{},
                  [](auto&&, auto&&) { return false; },
                  utl::overloaded{
                      [](utl::op, rt_transport_idx_t) {},
                      [&](rt_transport_idx_t const a,
                          rt_transport_idx_t const b) {
                        utl::verify(a == b, "{} != {}", a, b);
                        trace_direct("  found rt_transport {} visiting", a);
                        if (is_allowed(
                                q.allowed_claszes_,
                                rtt->rt_transport_section_clasz_[a].front())) {
                          for_each_from_to(
                              x, y, rtt->rt_transport_location_seq_[a], a);
                        }
                      }});
            });
      });
}

}  // namespace nigiri::routing