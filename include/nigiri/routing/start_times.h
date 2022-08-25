#pragma once

#include <chrono>

#include "cista/reflection/comparable.h"
#include "cista/reflection/printable.h"

#include "utl/enumerate.h"

#include "nigiri/routing/query.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing {

struct start {
  CISTA_PRINTABLE(start, "time_at_start", "time_at_stop", "stop")
  CISTA_COMPARABLE()
  unixtime_t time_at_start_;
  unixtime_t time_at_stop_;
  location_idx_t stop_;
};

template <direction SearchDir>
void get_starts(timetable const& tt,
                interval<unixtime_t> const& start_interval,
                vector<offset> const& station_offsets,
                std::vector<start>& starts) {
  auto const add_start_times_at_stop =
      [&](route_idx_t const route_idx, size_t const stop_idx,
          location_idx_t const location_idx,
          interval<unixtime_t> const& interval_with_offset,
          duration_t const offset) {
        auto const first_day_idx =
            tt.day_idx_mam(interval_with_offset.from_).first;
        auto const last_day_idx =
            tt.day_idx_mam(interval_with_offset.to_).first;

        auto const& transport_range = tt.route_transport_ranges_[route_idx];
        for (auto transport_idx = transport_range.from_;
             transport_idx != transport_range.to_; ++transport_idx) {
          auto const& traffic_days =
              tt.bitfields_[tt.transport_traffic_days_[transport_idx]];
          auto const stop_time = tt.event_mam(
              transport_idx, stop_idx,
              (SearchDir == direction::kForward ? event_type::kDep
                                                : event_type::kArr));
          for (auto day = first_day_idx; day <= last_day_idx; ++day) {
            if (traffic_days.test(to_idx(day)) &&
                interval_with_offset.contains(tt.to_unixtime(day, stop_time))) {
              auto const ev_time = tt.to_unixtime(day, stop_time);
              starts.emplace_back(
                  start{.time_at_start_ = SearchDir == direction::kForward
                                              ? ev_time - offset
                                              : ev_time + offset,
                        .time_at_stop_ = ev_time,
                        .stop_ = location_idx});
            }
          }
        }
      };

  auto const add_start_times = [&](offset const& o) {
    // Iterate routes visiting the location.
    for (auto const& r : tt.location_routes_.at(o.location_)) {

      // Iterate the location sequence, searching the given location.
      auto const location_seq = tt.route_location_seq_.at(r);
      for (auto const [i, s] : utl::enumerate(location_seq)) {
        if (s.location_idx() != o.location_) {
          continue;
        }

        // Ignore:
        // - in-allowed=false for forward search
        // - out-allowed=false for backward search
        // - entering at last stop for forward search
        // - exiting at first stop for backward search
        if ((SearchDir == direction::kBackward &&
             (i == 0U || !s.out_allowed())) ||
            (SearchDir == direction::kForward &&
             (i == location_seq.size() - 1 || !s.in_allowed()))) {
          continue;
        }

        add_start_times_at_stop(r, i, s.location_idx(),
                                SearchDir == direction::kForward
                                    ? start_interval + o.offset_
                                    : start_interval - o.offset_,
                                o.offset_);
      }
    }
  };

  starts.clear();
  for (auto const& offset : station_offsets) {
    add_start_times(offset);

    // Add one earliest arrival query at the end of the interval. This is only
    // used to dominate journeys from the interval that are suboptimal
    // considering a journey from outside the interval (i.e. outside journey
    // departs later and arrives at the same time). These journeys outside the
    // interval will be filtered out before returning the result.
    starts.emplace_back(
        start{.time_at_start_ = SearchDir == direction::kForward
                                    ? start_interval.to_
                                    : start_interval.from_ - 1_minutes,
              .time_at_stop_ =
                  SearchDir == direction::kForward
                      ? start_interval.to_ + offset.offset_
                      : start_interval.from_ - 1_minutes - offset.offset_,
              .stop_ = offset.location_});
  }

  std::sort(begin(starts), end(starts), [](start const& a, start const& b) {
    return SearchDir == direction::kForward ? b < a : a < b;
  });
}

}  // namespace nigiri::routing
