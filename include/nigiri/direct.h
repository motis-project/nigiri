#pragma once

#include "utl/sorted_diff.h"

#include "nigiri/for_each_meta.h"
#include "nigiri/location_match_mode.h"
#include "nigiri/timetable.h"

#include "rt/rt_timetable.h"

namespace nigiri {

#define trace_direct(...)

template <direction SearchDir, typename Fn>
void for_each_direct(timetable const& tt,
                     rt_timetable const* rtt,
                     routing::location_match_mode const from_match_mode,
                     routing::location_match_mode const to_match_mode,
                     location_idx_t const from,
                     location_idx_t const to,
                     interval<unixtime_t> const time,
                     Fn&& fn) {
  constexpr auto const kFwd = SearchDir == direction::kForward;
  auto const start_ev_type = kFwd ? event_type::kDep : event_type::kArr;
  auto const end_ev_type = kFwd ? event_type::kArr : event_type::kDep;

  auto const is_transport_active = [&](transport_idx_t const t,
                                       std::size_t const day) {
    if (rtt != nullptr) {
      return rtt->bitfields_[rtt->transport_traffic_days_[t]].test(day);
    } else {
      return tt.bitfields_[tt.transport_traffic_days_[t]].test(day);
    }
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
                  "      transport {} not active on {} (ev_time={})",
                  tt.transport_name(t),
                  tt.internal_interval_days().from_ + date::days{1} * start_day,
                  ev);
              continue;
            }

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
                      rt::run{.t_ = tr,
                              .stop_range_ = {0U, static_cast<stop_idx_t>(
                                                      loc_seq.size())}},
                      start_stop_idx, end_stop_idx}};
              fn(routing::journey{
                  .legs_ = {leg},
                  .start_time_ = leg.dep_time_,
                  .dest_time_ = leg.arr_time_,
                  .dest_ = leg.to_,
                  .transfers_ = 0U,
              });
            } else {
              trace_direct("      time={} does't contain {}", time, start_time);
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
                  rt::run{.stop_range_ = {0U, static_cast<stop_idx_t>(
                                                  loc_seq.size())},
                          .rt_ = rt},
                  start, end}};
          fn(routing::journey{
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
  auto done = hash_set<std::pair<location_idx_t, location_idx_t>>{};
  routing::for_each_meta(
      tt, from_match_mode, from, [&](location_idx_t const x) {
        routing::for_each_meta(
            tt, to_match_mode, to, [&](location_idx_t const y) {
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
                        for_each_from_to(x, y, tt.route_location_seq_[a], a);
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
                        for_each_from_to(x, y,
                                         rtt->rt_transport_location_seq_[a], a);
                      }});
            });
      });
}

}  // namespace nigiri