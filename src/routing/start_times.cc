#include "nigiri/routing/start_times.h"

#include "utl/enumerate.h"
#include "utl/overloaded.h"

namespace nigiri::routing {

template <direction SearchDir>
void add_start_times_at_stop(timetable const& tt,
                             route_idx_t const route_idx,
                             size_t const stop_idx,
                             location_idx_t const location_idx,
                             interval<unixtime_t> const& interval_with_offset,
                             duration_t const offset,
                             std::vector<start>& starts) {
  auto const first_day_idx = tt.day_idx_mam(interval_with_offset.from_).first;
  auto const last_day_idx = tt.day_idx_mam(interval_with_offset.to_).first;

  auto const& transport_range = tt.route_transport_ranges_[route_idx];
  for (auto transport_idx = transport_range.from_;
       transport_idx != transport_range.to_; ++transport_idx) {
    auto const& traffic_days =
        tt.bitfields_[tt.transport_traffic_days_[transport_idx]];
    auto const stop_time =
        tt.event_mam(transport_idx, stop_idx,
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
}

template <direction SearchDir>
void add_starts_in_interval(timetable const& tt,
                            interval<unixtime_t> const& interval,
                            offset const& o,
                            std::vector<start>& starts) {
  // Iterate routes visiting the location.
  for (auto const& r : tt.location_routes_.at(o.location_)) {

    // Iterate the location sequence, searching the given location.
    auto const location_seq = tt.route_location_seq_.at(r);
    for (auto const [i, s] : utl::enumerate(location_seq)) {
      if (timetable::stop{s}.location_idx() != o.location_) {
        continue;
      }

      // Ignore:
      // - in-allowed=false for forward search
      // - out-allowed=false for backward search
      // - entering at last stop for forward search
      // - exiting at first stop for backward search
      if ((SearchDir == direction::kBackward &&
           (i == 0U || !timetable::stop{s}.out_allowed())) ||
          (SearchDir == direction::kForward &&
           (i == location_seq.size() - 1 ||
            !timetable::stop{s}.in_allowed()))) {
        continue;
      }

      add_start_times_at_stop<SearchDir>(
          tt, r, i, timetable::stop{s}.location_idx(),
          SearchDir == direction::kForward ? interval + o.offset_
                                           : interval - o.offset_,
          o.offset_, starts);
    }
  }

  // Add one earliest arrival query at the end of the interval. This is only
  // used to dominate journeys from the interval that are suboptimal
  // considering a journey from outside the interval (i.e. outside journey
  // departs later and arrives at the same time). These journeys outside the
  // interval will be filtered out before returning the result.
  starts.emplace_back(
      start{.time_at_start_ = SearchDir == direction::kForward
                                  ? interval.to_
                                  : interval.from_ - 1_minutes,
            .time_at_stop_ = SearchDir == direction::kForward
                                 ? interval.to_ + o.offset_
                                 : interval.from_ - 1_minutes - o.offset_,
            .stop_ = o.location_});
}

template <typename Fn>
void for_each_meta(timetable const& tt,
                   location_match_mode const mode,
                   location_idx_t const l,
                   Fn&& fn) {
  if (mode == location_match_mode::kExact) {
    fn(l);
  } else if (mode == location_match_mode::kOnlyChildren) {
    fn(l);
    for (auto const& eq : tt.locations_.children_.at(l)) {
      fn(eq);
    }
  } else {
    fn(l);
    for (auto const& eq : tt.locations_.equivalences_.at(l)) {
      fn(eq);
    }
  }
}

template <direction SearchDir>
void get_starts(timetable const& tt,
                variant<unixtime_t, interval<unixtime_t>> const& start_time,
                std::vector<offset> const& station_offsets,
                location_match_mode const mode,
                std::vector<start>& starts) {
  std::set<location_idx_t> seen;
  for (auto const& o : station_offsets) {
    seen.clear();
    for_each_meta(tt, mode, o.location_, [&](location_idx_t const l) {
      if (!seen.emplace(l).second) {
        return;
      }
      start_time.apply(utl::overloaded{
          [&](interval<unixtime_t> const interval) {
            add_starts_in_interval<SearchDir>(
                tt, interval, offset{l, o.offset_, o.type_}, starts);
          },
          [&](unixtime_t const t) {
            starts.emplace_back(
                start{.time_at_start_ = t,
                      .time_at_stop_ = SearchDir == direction::kForward
                                           ? t + o.offset_
                                           : t - o.offset_,
                      .stop_ = l});
          }});
    });
  }

  std::sort(begin(starts), end(starts), [](start const& a, start const& b) {
    return SearchDir == direction::kForward ? b < a : a < b;
  });
}

void collect_destinations(timetable const& tt,
                          std::vector<std::vector<offset>> const& destinations,
                          location_match_mode const match_mode,
                          std::vector<std::set<location_idx_t>>& out) {
  out.resize(std::max(out.size(), destinations.size()));
  for (auto const [i, dest] : utl::enumerate(destinations)) {
    if (match_mode == location_match_mode::kEquivalent) {
      for (auto const& d : dest) {
        for_each_meta(
            tt, match_mode, d.location_,
            [&, i = i](location_idx_t const l) { out[i].emplace(l); });
      }
    }
  }
}

template void get_starts<direction::kForward>(
    timetable const&,
    variant<unixtime_t, interval<unixtime_t>> const&,
    std::vector<offset> const&,
    location_match_mode const,
    std::vector<start>&);

template void get_starts<direction::kBackward>(
    timetable const&,
    variant<unixtime_t, interval<unixtime_t>> const&,
    std::vector<offset> const&,
    location_match_mode const,
    std::vector<start>&);

}  // namespace nigiri::routing