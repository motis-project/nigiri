#pragma once

#include "nigiri/common/linear_lower_bound.h"
#include "nigiri/routing/raptor/debug.h"
#include "nigiri/timetable.h"

namespace nigiri::routing {

template <direction const SearchDir = direction::kForward,
          typename TrafficDaySrc,
          typename Fn>
transport get_earliest_transport(timetable const& tt,
                                 TrafficDaySrc const& traffic_day_src,
                                 [[maybe_unused]] unsigned const k,
                                 route_idx_t const r,
                                 stop_idx_t const stop_idx,
                                 day_idx_t const day_at_stop,
                                 minutes_after_midnight_t const mam_at_stop,
                                 [[maybe_unused]] location_idx_t const l,
                                 Fn&& worse_than_dest) {
  constexpr auto const kFwd = SearchDir == direction::kForward;
  constexpr auto const is_better = [](auto a, auto b) {
    return kFwd ? a < b : a > b;
  };
  constexpr auto const is_better_or_eq = [](auto a, auto b) {
    return kFwd ? a <= b : a >= b;
  };

  auto const as_int = [](auto const x) { return static_cast<int>(x.v_); };

  auto const event_times = tt.event_times_at_stop(
      r, stop_idx, kFwd ? event_type::kDep : event_type::kArr);

  auto get_begin_it = [](auto const& t) {
    if constexpr (kFwd) {
      return t.begin();
    } else {
      return t.rbegin();
    }
  };

  auto get_end_it = [](auto const& t) {
    if constexpr (kFwd) {
      return t.end();
    } else {
      return t.rend();
    }
  };

  auto const seek_first_day = [&]() {
    return linear_lb(get_begin_it(event_times), get_end_it(event_times),
                     mam_at_stop,
                     [&](delta const a, minutes_after_midnight_t const b) {
                       return is_better(a.mam(), b.count());
                     });
  };

  trace(
      "┊ │k={}    et: current_best_at_stop={}, stop_idx={}, location={}\n", k,
      tt_.to_unixtime(day_at_stop, mam_at_stop), stop_idx,
      location{tt_, stop{tt_.route_location_seq_[r][stop_idx]}.location_idx()});

  constexpr auto const kNDaysToIterate = day_idx_t::value_t{2U};
  for (auto i = day_idx_t::value_t{0U}; i != kNDaysToIterate; ++i) {
    auto const ev_time_range =
        it_range{i == 0U ? seek_first_day() : get_begin_it(event_times),
                 get_end_it(event_times)};
    if (ev_time_range.empty()) {
      continue;
    }

    auto const day = kFwd ? day_at_stop + i : day_at_stop - i;
    for (auto it = begin(ev_time_range); it != end(ev_time_range); ++it) {
      auto const t_offset = static_cast<std::size_t>(&*it - event_times.data());
      auto const ev = *it;
      auto const ev_mam = ev.mam();

      if (worse_than_dest(day, ev_mam)) {
        trace(
            "┊ │k={}      => name={}, dbg={}, day={}={}, best_mam={}, "
            "transport_mam={}, transport_time={} => TIME AT DEST {} IS "
            "BETTER!\n",
            k, tt_.transport_name(tt_.route_transport_ranges_[r][t_offset]),
            tt_.dbg(tt_.route_transport_ranges_[r][t_offset]), day,
            tt_.to_unixtime(day, 0_minutes), mam_at_stop, ev_mam,
            tt_.to_unixtime(day, duration_t{ev_mam}),
            to_unix(time_at_dest_[k]));
        return {transport_idx_t::invalid(), day_idx_t::invalid()};
      }

      auto const t = tt.route_transport_ranges_[r][t_offset];
      if (i == 0U && !is_better_or_eq(mam_at_stop.count(), ev_mam)) {
        trace(
            "┊ │k={}      => transport={}, name={}, dbg={}, day={}/{}, "
            "best_mam={}, "
            "transport_mam={}, transport_time={} => NO REACH!\n",
            k, t, tt_.transport_name(t), tt_.dbg(t), i, day, mam_at_stop,
            ev_mam, ev);
        continue;
      }

      auto const ev_day_offset = ev.days();
      auto const start_day =
          static_cast<std::size_t>(as_int(day) - ev_day_offset);
      if (!traffic_day_src
               .bitfields_[traffic_day_src.transport_traffic_days_[t]]
               .test(start_day)) {
        trace(
            "┊ │k={}      => transport={}, name={}, dbg={}, day={}/{}, "
            "ev_day_offset={}, "
            "best_mam={}, "
            "transport_mam={}, transport_time={} => NO TRAFFIC!\n",
            k, t, tt_.transport_name(t), tt_.dbg(t), i, day, ev_day_offset,
            mam_at_stop, ev_mam, ev);
        continue;
      }

      trace(
          "┊ │k={}      => ET FOUND: name={}, dbg={}, at day {} "
          "(day_offset={}) - ev_mam={}, ev_time={}, ev={}\n",
          k, tt_.transport_name(t), tt_.dbg(t), day, ev_day_offset, ev_mam, ev,
          tt_.to_unixtime(day, duration_t{ev_mam}));
      return {t, static_cast<day_idx_t>(as_int(day) - ev_day_offset)};
    }
  }
  return {};
}

}  // namespace nigiri::routing