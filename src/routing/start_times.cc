#include "nigiri/routing/start_times.h"

#include "nigiri/routing/for_each_meta.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/special_stations.h"
#include "utl/enumerate.h"
#include "utl/get_or_create.h"
#include "utl/overloaded.h"

namespace nigiri::routing {

constexpr auto const kTracing = false;

template <typename... Args>
void trace_start(char const* fmt_str, Args... args) {
  if constexpr (kTracing) {
    fmt::print(std::cout, fmt_str, std::forward<Args&&>(args)...);
  }
}

template <typename Collection, typename Less>
std::pair<typename Collection::iterator, bool> insert_sorted(
    Collection& v, typename Collection::value_type el, Less&& less) {
  using std::begin;
  using std::end;
  auto const it =
      std::lower_bound(begin(v), end(v), el, std::forward<Less>(less));
  if (it == std::end(v) || *it != el) {
    return {v.insert(it, std::move(el)), true};
  }
  return {it, false};
}

template <typename Less>
void add_start_times_at_stop(direction const search_dir,
                             timetable const& tt,
                             rt_timetable const* rtt,
                             route_idx_t const route_idx,
                             stop_idx_t const stop_idx,
                             location_idx_t const location_idx,
                             interval<unixtime_t> const& interval_with_offset,
                             duration_t const offset,
                             std::vector<start>& starts,
                             Less&& less) {
  auto const first_day_idx = tt.day_idx_mam(interval_with_offset.from_).first;
  auto const last_day_idx = tt.day_idx_mam(interval_with_offset.to_).first;
  trace_start(
      "      add_start_times_at_stop(interval={}) - first_day_idx={}, "
      "last_day_idx={}, date_range={}\n",
      interval_with_offset, first_day_idx, last_day_idx, tt.date_range_);

  auto const& transport_range = tt.route_transport_ranges_[route_idx];
  for (auto t = transport_range.from_; t != transport_range.to_; ++t) {
    auto const& traffic_days =
        rtt == nullptr ? tt.bitfields_[tt.transport_traffic_days_[t]]
                       : rtt->bitfields_[rtt->transport_traffic_days_[t]];
    auto const stop_time =
        tt.event_mam(t, stop_idx,
                     (search_dir == direction::kForward ? event_type::kDep
                                                        : event_type::kArr));

    auto const day_offset =
        static_cast<std::uint16_t>(stop_time.count() / 1440);
    auto const stop_time_mam = duration_t{stop_time.count() % 1440};
    trace_start(
        "      interval=[{}, {}[, transport={}, name={}, stop_time={} "
        "(day_offset={}, stop_time_mam={})\n",
        interval_with_offset.from_, interval_with_offset.to_, t,
        tt.transport_name(t), stop_time, day_offset, stop_time_mam);
    for (auto day = first_day_idx; day <= last_day_idx; ++day) {
      if (traffic_days.test(to_idx(day - day_offset)) &&
          interval_with_offset.contains(tt.to_unixtime(day, stop_time_mam))) {
        auto const ev_time = tt.to_unixtime(day, stop_time_mam);
        auto const [it, inserted] = insert_sorted(
            starts,
            start{.time_at_start_ = search_dir == direction::kForward
                                        ? ev_time - offset
                                        : ev_time + offset,
                  .time_at_stop_ = ev_time,
                  .stop_ = location_idx},
            std::forward<Less>(less));
        trace_start(
            "        => ADD START: time_at_start={}, time_at_stop={}, "
            "stop={}\n",
            it->time_at_start_, it->time_at_stop_,
            location{tt, starts.back().stop_});
      } else {
        trace_start(
            "        skip: day={}, day_offset={}, date={}, active={}, "
            "in_interval={}\n",
            day, day_offset,
            tt.date_range_.from_ + to_idx(day - day_offset) * 1_days,
            traffic_days.test(to_idx(day)),
            interval_with_offset.contains(
                tt.to_unixtime(day, stop_time.as_duration())));
      }
    }
  }
}

template <typename Less>
void add_starts_in_interval(direction const search_dir,
                            timetable const& tt,
                            rt_timetable const* rtt,
                            interval<unixtime_t> const& interval,
                            location_idx_t const l,
                            duration_t const d,
                            std::vector<start>& starts,
                            bool const add_ontrip,
                            Less&& cmp) {
  trace_start(
      "    add_starts_in_interval(interval={}, stop={}, duration={}): {} "
      "routes\n",
      interval, location{tt, l},  // NOLINT(clang-analyzer-core.CallAndMessage)
      d, tt.location_routes_.at(l).size());

  // Iterate routes visiting the location.
  for (auto const& r : tt.location_routes_.at(l)) {

    // Iterate the location sequence, searching the given location.
    auto const location_seq = tt.route_location_seq_.at(r);
    trace_start("  location_seq: {}\n", r);
    for (auto const [i, s] : utl::enumerate(location_seq)) {
      auto const stp = stop{s};
      if (stp.location_idx() != l) {
        continue;
      }

      // Ignore:
      // - in-allowed=false for forward search
      // - out-allowed=false for backward search
      // - entering at last stp for forward search
      // - exiting at first stp for backward search
      if ((search_dir == direction::kBackward &&
           (i == 0U || !stp.out_allowed())) ||
          (search_dir == direction::kForward &&
           (i == location_seq.size() - 1 || !stp.in_allowed()))) {
        trace_start("    skip: i={}, out_allowed={}, in_allowed={}\n", i,
                    stp.out_allowed(), stp.in_allowed());
        continue;
      }

      trace_start("    -> no skip -> add_start_times_at_stop()\n");
      add_start_times_at_stop(
          search_dir, tt, rtt, r, static_cast<stop_idx_t>(i),
          stop{s}.location_idx(),
          search_dir == direction::kForward ? interval + d : interval - d, d,
          starts, std::forward<Less>(cmp));
    }
  }

  // Real-time starts
  if (rtt != nullptr) {
    for (auto const& rt_t : rtt->location_rt_transports_.at(l)) {
      auto const location_seq = rtt->rt_transport_location_seq_.at(rt_t);
      for (auto const [i, s] : utl::enumerate(location_seq)) {
        auto const stp = stop{s};
        if (stp.location_idx() != l) {
          continue;
        }

        if ((search_dir == direction::kBackward &&
             (i == 0U || !stp.out_allowed())) ||
            (search_dir == direction::kForward &&
             (i == location_seq.size() - 1 || !stp.in_allowed()))) {
          trace_start("    skip: i={}, out_allowed={}, in_allowed={}\n", i,
                      stp.out_allowed(), stp.in_allowed());
          continue;
        }

        auto const ev_time = rtt->unix_event_time(
            rt_t, static_cast<stop_idx_t>(i),
            (search_dir == direction::kForward ? event_type::kDep
                                               : event_type::kArr));
        auto const [it, inserted] = insert_sorted(
            starts,
            start{.time_at_start_ = search_dir == direction::kForward
                                        ? ev_time - d
                                        : ev_time + d,
                  .time_at_stop_ = ev_time,
                  .stop_ = l},
            std::forward<Less>(cmp));
        trace_start(
            "        => ADD RT START: time_at_start={}, time_at_stop={}, "
            "stop={}\n",
            it->time_at_start_, it->time_at_stop_,
            location{tt, starts.back().stop_});
      }
    }
  }

  // Add one earliest arrival query at the end of the interval. This is only
  // used to dominate journeys from the interval that are suboptimal
  // considering a journey from outside the interval (i.e. outside journey
  // departs later and arrives at the same time). These journeys outside the
  // interval will be filtered out before returning the result.
  if (add_ontrip) {
    insert_sorted(starts,
                  start{.time_at_start_ = search_dir == direction::kForward
                                              ? interval.to_
                                              : interval.from_ - 1_minutes,
                        .time_at_stop_ = search_dir == direction::kForward
                                             ? interval.to_ + d
                                             : interval.from_ - 1_minutes - d,
                        .stop_ = l},
                  cmp);
  }
}

void get_starts(direction const search_dir,
                timetable const& tt,
                rt_timetable const* rtt,
                start_time_t const& start_time,
                std::vector<offset> const& station_offsets,
                location_match_mode const mode,
                bool const use_start_footpaths,
                std::vector<start>& starts,
                bool const add_ontrip,
                profile_idx_t const prf_idx) {
  hash_map<location_idx_t, duration_t> shortest_start;

  auto const update = [&](location_idx_t const l, duration_t const d) {
    auto& val = utl::get_or_create(shortest_start, l, [d]() { return d; });
    val = std::min(val, d);
  };

  auto const fwd = search_dir == direction::kForward;
  for (auto const& o : station_offsets) {
    for_each_meta(tt, mode, o.target(), [&](location_idx_t const l) {
      update(l, o.duration());
      if (use_start_footpaths) {
        auto const footpaths = fwd ? tt.locations_.footpaths_out_[prf_idx][l]
                                   : tt.locations_.footpaths_in_[prf_idx][l];
        for (auto const& fp : footpaths) {
          update(fp.target(), o.duration() + fp.duration());
        }
      }
    });
  }

  auto const cmp = [&](start const& a, start const& b) {
    return fwd ? b < a : a < b;
  };
  for (auto const& s : shortest_start) {
    auto const l = s.first;
    auto const o = s.second;
    std::visit(utl::overloaded{
                   [&](interval<unixtime_t> const interval) {
                     add_starts_in_interval(search_dir, tt, rtt, interval, l, o,
                                            starts, add_ontrip, cmp);
                   },
                   [&](unixtime_t const t) {
                     insert_sorted(starts,
                                   start{.time_at_start_ = t,
                                         .time_at_stop_ = fwd ? t + o : t - o,
                                         .stop_ = l},
                                   cmp);
                   }},
               start_time);
  }
}

void collect_destinations(timetable const& tt,
                          std::vector<offset> const& destinations,
                          location_match_mode const match_mode,
                          std::vector<bool>& is_destination,
                          std::vector<std::uint16_t>& dist_to_dest) {
  is_destination.resize(tt.n_locations());
  utl::fill(is_destination, false);

  static constexpr auto const kIntermodalTarget =
      to_idx(get_special_station(special_station::kEnd));

  if (match_mode == location_match_mode::kIntermodal) {
    is_destination[kIntermodalTarget] = true;
    dist_to_dest.resize(tt.n_locations());
    utl::fill(dist_to_dest, std::numeric_limits<std::uint16_t>::max());
  } else {
    dist_to_dest.clear();
  }

  for (auto const& d : destinations) {
    trace_start("DEST METAS OF {}\n", location{tt, d.target_});
    for_each_meta(tt, match_mode, d.target_, [&](location_idx_t const l) {
      if (match_mode == location_match_mode::kIntermodal) {
        dist_to_dest[to_idx(l)] =
            static_cast<std::uint16_t>(d.duration_.count());
      } else {
        is_destination[to_idx(l)] = true;
      }
      trace_start("  DEST META: {}, duration={}\n", location{tt, l},
                  d.duration_);
    });
  }
}
void collect_destinations_gpu(timetable const& tt,
                          std::vector<offset> const& destinations,
                          location_match_mode const match_mode,
                          std::vector<uint8_t>& is_destination,
                          std::vector<std::uint16_t>& dist_to_dest) {
  is_destination.resize(tt.n_locations());
  utl::fill(is_destination, false);

  static constexpr auto const kIntermodalTarget =
      to_idx(get_special_station(special_station::kEnd));

  if (match_mode == location_match_mode::kIntermodal) {
    is_destination[kIntermodalTarget] = true;
    dist_to_dest.resize(tt.n_locations());
    utl::fill(dist_to_dest, std::numeric_limits<std::uint16_t>::max());
  } else {
    dist_to_dest.clear();
  }

  for (auto const& d : destinations) {
    trace_start("DEST METAS OF {}\n", location{tt, d.target_});
    for_each_meta(tt, match_mode, d.target_, [&](location_idx_t const l) {
      if (match_mode == location_match_mode::kIntermodal) {
        dist_to_dest[to_idx(l)] =
            static_cast<std::uint16_t>(d.duration_.count());
      } else {
        is_destination[to_idx(l)] = true;
      }
      trace_start("  DEST META: {}, duration={}\n", location{tt, l},
                  d.duration_);
    });
  }
}
}  // namespace nigiri::routing
