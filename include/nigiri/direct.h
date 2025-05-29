#pragma once

#include "utl/sorted_diff.h"

#include "nigiri/for_each_meta.h"
#include "nigiri/location_match_mode.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"

namespace nigiri {

#define trace_direct(...) fmt::println(__VA_ARGS__)

template <direction SearchDir, typename Fn>
void for_each_direct(timetable const& tt,
                     rt_timetable const* rtt,
                     location_idx_t const from,
                     location_idx_t const to,
                     routing::query const& q,
                     interval<unixtime_t> const time,
                     hash_set<std::pair<location_idx_t, location_idx_t>>& done,
                     Fn&& fn) {
  constexpr auto const kFwd = SearchDir == direction::kForward;
  auto const start_ev_type = kFwd ? event_type::kDep : event_type::kArr;
  auto const end_ev_type = kFwd ? event_type::kArr : event_type::kDep;

  auto const get_offset =
      [](direction const search_dir, location_idx_t const l,
         std::vector<routing::offset> const& offsets,
         routing::td_offsets_t const& td_offsets,
         unixtime_t const t) -> std::optional<routing::offset> {
    {  // Try to find normal offset.
      auto const it = utl::find_if(
          offsets, [&](routing::offset const& o) { return o.target() == l; });
      if (it != end(offsets)) {
        return *it;
      }
    }

    {  // Try to find td-offset.
      auto const it = td_offsets.find(l);
      if (it != end(td_offsets)) {
        auto const d = get_td_duration(search_dir, it->second, t);
        if (d.has_value()) {
          auto const& [duration, o] = *d;
          return routing::offset{l, duration, o.transport_mode_id_};
        }
      }
    }

    return std::nullopt;
  };

  auto const is_transport_active = [&](transport_idx_t const t,
                                       std::size_t const day) {
    if (rtt != nullptr) {
      return rtt->bitfields_[rtt->transport_traffic_days_[t]].test(day);
    } else {
      return tt.bitfields_[tt.transport_traffic_days_[t]].test(day);
    }
  };

  auto const add_offsets_fn = [&](routing::journey&& j) {
    auto const from_match = kFwd ? q.start_match_mode_ : q.dest_match_mode_;
    auto const to_match = kFwd ? q.dest_match_mode_ : q.start_match_mode_;
    auto const& from_offsets = kFwd ? q.start_ : q.destination_;
    auto const& to_offsets = kFwd ? q.destination_ : q.start_;
    auto const& from_td_offsets = kFwd ? q.td_start_ : q.td_dest_;
    auto const& to_td_offsets = kFwd ? q.td_dest_ : q.td_start_;

    if (from_match == routing::location_match_mode::kIntermodal) {
      auto const& l = j.legs_.front();
      auto const offset =
          get_offset(direction::kBackward, l.from_, from_offsets,
                     from_td_offsets, l.dep_time_);
      if (!offset.has_value()) {
        return;
      }
      j.start_time_ = l.dep_time_ - offset->duration();
      j.legs_.insert(
          begin(j.legs_),
          routing::journey::leg{direction::kForward,
                                get_special_station(special_station::kStart),
                                l.from_, j.start_time_, l.dep_time_, *offset});

      if (!kFwd) {
        j.dest_ = get_special_station(special_station::kStart);
      }
    }

    if (to_match == routing::location_match_mode::kIntermodal) {
      auto const& l = j.legs_.back();
      auto const offset = get_offset(direction::kForward, l.to_, to_offsets,
                                     to_td_offsets, l.arr_time_);
      if (!offset.has_value()) {
        return;
      }
      j.dest_time_ = l.arr_time_ + offset->duration();
      j.legs_.push_back(
          routing::journey::leg{direction::kForward, l.to_,
                                get_special_station(special_station::kEnd),
                                l.arr_time_, j.dest_time_, *offset});

      if (kFwd) {
        j.dest_ = get_special_station(special_station::kEnd);
      }
    }

    fn(std::move(j));
  };

  auto const checked_fn = [&](routing::journey&& j) {
    if (!q.require_bike_transport_ && !q.require_car_transport_) {
      add_offsets_fn(std::move(j));
      return;
    }

    auto const& l = j.legs_.front();
    auto const& ree = std::get<routing::journey::run_enter_exit>(l.uses_);

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

    add_offsets_fn(std::move(j));
  };

  auto const check_interval = utl::overloaded{
      [&](route_idx_t const r, stop_idx_t const start_stop_idx,
          stop_idx_t const end_stop_idx) {
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
            if (time.contains(start_time)) {
              auto const leg = routing::journey::leg{
                  SearchDir,
                  stop{loc_seq[start_stop_idx]}.location_idx(),
                  stop{loc_seq[end_stop_idx]}.location_idx(),
                  start_time,
                  end_time,
                  routing::journey::run_enter_exit{
                      rt::frun{
                          tt, rtt,
                          rt::run{.t_ = tr,
                                  .stop_range_ = {0U, static_cast<stop_idx_t>(
                                                          loc_seq.size())}}},
                      start_stop_idx, end_stop_idx}};
              trace_direct("        time={} contains {}", time, start_time);
              checked_fn(routing::journey{
                  .legs_ = {leg},
                  .start_time_ = leg.dep_time_,
                  .dest_time_ = leg.arr_time_,
                  .dest_ = leg.to_,
                  .transfers_ = 0U,
              });
            } else {
              trace_direct("        time={} doesn't contain {}", time,
                           start_time);
            }
          }
        }
      },
      [&](rt_transport_idx_t const rt, stop_idx_t const start,
          stop_idx_t const end) {
        auto const start_time = rtt->unix_event_time(rt, start, start_ev_type);
        auto const end_time = rtt->unix_event_time(rt, end, end_ev_type);
        if (time.contains(start_time)) {
          auto const loc_seq = rtt->rt_transport_location_seq_[rt];
          auto const leg = routing::journey::leg{
              SearchDir,
              stop{loc_seq[start]}.location_idx(),
              stop{loc_seq[end]}.location_idx(),
              start_time,
              end_time,
              routing::journey::run_enter_exit{
                  rt::frun{tt, rtt,
                           rt::run{.stop_range_ = {0U, static_cast<stop_idx_t>(
                                                           loc_seq.size())},
                                   .rt_ = rt}},
                  start, end}};
          trace_direct("        time={} contains rt_transport={}, ev_time={}",
                       time, rtt->transport_name(tt, rt), start_time);
          checked_fn(routing::journey{
              .legs_ = {leg},
              .start_time_ = leg.dep_time_,
              .dest_time_ = leg.arr_time_,
              .dest_ = leg.to_,
              .transfers_ = 0U,
          });
        } else {
          trace_direct(
              "      time={} doesn't contain rt_transport={}, ev_time={}", time,
              rtt->transport_name(tt, rt), start_time);
        }
      }};

  auto const for_each_from_to = [&](location_idx_t const x,
                                    location_idx_t const y, auto&& loc_seq,
                                    auto&& r) {
    auto first = std::optional<stop_idx_t>{};
    for (auto i = 0U; i != loc_seq.size(); ++i) {
      auto const stop_idx = static_cast<stop_idx_t>(
          SearchDir == direction::kForward ? i : loc_seq.size() - i - 1U);
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
  routing::for_each_meta(
      tt, routing::location_match_mode::kEquivalent, from,
      [&](location_idx_t const x) {
        routing::for_each_meta(
            tt, routing::location_match_mode::kEquivalent, to,
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
                        if (routing::is_allowed(q.allowed_claszes_,
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
                        if (routing::is_allowed(
                                q.allowed_claszes_,
                                rtt->rt_transport_section_clasz_[a].front())) {
                          for_each_from_to(
                              x, y, rtt->rt_transport_location_seq_[a], a);
                        }
                      }});
            });
      });
}

}  // namespace nigiri