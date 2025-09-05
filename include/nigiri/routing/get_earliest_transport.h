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
        return {transport_idx_t::invalid(), day_idx_t::invalid()};
      }

      auto const t = tt.route_transport_ranges_[r][t_offset];
      if (i == 0U && !is_better_or_eq(mam_at_stop.count(), ev_mam)) {
        continue;
      }

      auto const ev_day_offset = ev.days();
      auto const start_day =
          static_cast<std::size_t>(as_int(day) - ev_day_offset);
      if (!traffic_day_src
               .bitfields_[traffic_day_src.transport_traffic_days_[t]]
               .test(start_day)) {
        continue;
      }

      return {t, static_cast<day_idx_t>(as_int(day) - ev_day_offset)};
    }
  }
  return {};
}

}  // namespace nigiri::routing