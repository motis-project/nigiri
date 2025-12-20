#include "nigiri/routing/start_times.h"

#include "utl/enumerate.h"
#include "utl/equal_ranges_linear.h"
#include "utl/get_or_create.h"
#include "utl/overloaded.h"

#include "nigiri/for_each_meta.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/special_stations.h"

namespace nigiri::routing {

constexpr auto const kTracing = false;

using location_offset_t = std::variant<duration_t, std::span<td_offset const>>;

duration_t get_duration(direction const search_dir,
                        unixtime_t const t,
                        location_offset_t const o,
                        bool const invert = true) {
  return std::visit(
      utl::overloaded{[](duration_t const x) { return x; },
                      [&](std::span<td_offset const> td) {
                        auto duration = get_td_duration(
                            invert ? flip(search_dir) : search_dir, td, t);
                        return duration.has_value() ? duration->first
                                                    : footpath::kMaxDuration;
                      }},
      o);
}

template <typename... Args>
void trace_start(char const* fmt_str, Args... args) {
  if constexpr (kTracing) {
    fmt::print(std::cout, fmt::runtime(fmt_str), std::forward<Args&&>(args)...);
  }
}

void add_start_times_at_stop(direction const search_dir,
                             timetable const& tt,
                             rt_timetable const* rtt,
                             route_idx_t const route_idx,
                             stop_idx_t const stop_idx,
                             location_idx_t const location_idx,
                             interval<unixtime_t> const& iv_at_start,
                             interval<unixtime_t> const& iv_at_stop,
                             location_offset_t const offset,
                             std::vector<start>& starts) {
  auto const is_better_or_eq = [&](auto a, auto b) {
    return search_dir == direction::kForward ? a <= b : a >= b;
  };

  auto const first_day_idx = tt.day_idx_mam(iv_at_stop.from_).first;
  auto const last_day_idx = tt.day_idx_mam(iv_at_stop.to_).first;
  trace_start(
      "      add_start_times_at_stop(interval={}) - first_day_idx={}, "
      "last_day_idx={}, date_range={}\n",
      iv_at_stop, first_day_idx, last_day_idx, tt.date_range_);

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
        iv_at_stop.from_, iv_at_stop.to_, t, tt.transport_name(t), stop_time,
        day_offset, stop_time_mam);
    for (auto day = first_day_idx; day <= last_day_idx; ++day) {
      if (traffic_days.test(to_idx(day - day_offset)) &&
          iv_at_stop.contains(tt.to_unixtime(day, stop_time_mam))) {
        auto const ev_time = tt.to_unixtime(day, stop_time_mam);
        auto const d = get_duration(search_dir, ev_time, offset);
        if (d == footpath::kMaxDuration) {
          trace_start("        {} => infeasible\n", ev_time);
          continue;
        }
        trace_start("        {} => duration={}\n", ev_time, d);
        auto const time_at_start =
            search_dir == direction::kForward ? ev_time - d : ev_time + d;
        if (!iv_at_start.contains(time_at_start)) {
          trace_start("      iv_at_start={} doesn't contain time_at_start={}\n",
                      iv_at_start, time_at_start);
          continue;
        }
        if (!starts.empty() && starts.back().time_at_start_ == time_at_start &&
            is_better_or_eq(starts.back().time_at_stop_, ev_time)) {
          trace_start("      time_at_start={} -> no improvement\n", iv_at_start,
                      time_at_start);
          continue;
        }
        auto const& s =
            starts.emplace_back(start{.time_at_start_ = time_at_start,
                                      .time_at_stop_ = ev_time,
                                      .stop_ = location_idx});
        trace_start(
            "        => ADD START: time_at_start={}, time_at_stop={}, "
            "stop={}\n",
            s.time_at_start_, s.time_at_stop_, loc{tt, starts.back().stop_});
      } else {
        trace_start(
            "        skip: day={}, day_offset={}, date={}, active={}, "
            "in_interval={}\n",
            day, day_offset,
            tt.date_range_.from_ + to_idx(day - day_offset) * 1_days,
            traffic_days.test(to_idx(day)),
            iv_at_stop.contains(tt.to_unixtime(day, stop_time.as_duration())));
      }
    }
  }
}

void add_starts_in_interval(direction const search_dir,
                            timetable const& tt,
                            rt_timetable const* rtt,
                            interval<unixtime_t> const& iv,
                            location_idx_t const l,
                            location_offset_t const location_offset,
                            duration_t const max_start_offset,
                            profile_idx_t const p,
                            std::vector<start>& starts,
                            bool const add_ontrip) {
  trace_start(
      "    add_starts_in_interval(interval={}, stop={}): {} "
      "routes\n",
      iv, loc{tt, l},  // NOLINT(clang-analyzer-core.CallAndMessage)
      tt.location_routes_.at(l).size());

  // Iterate routes visiting the location.
  for (auto const& r : tt.location_routes_.at(l)) {

    // Iterate the location sequence, searching the given location.
    auto const location_seq = tt.route_location_seq_.at(r);
    trace_start("  location_seq: route={}\n", r);
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
           (i == 0U || !stp.out_allowed(p))) ||
          (search_dir == direction::kForward &&
           (i == location_seq.size() - 1 || !stp.in_allowed(p)))) {
        trace_start("    skip: i={}, out_allowed={}, in_allowed={}\n", i,
                    stp.out_allowed(p), stp.in_allowed(p));
        continue;
      }

      trace_start("    -> no skip -> add_start_times_at_stop()\n");
      add_start_times_at_stop(
          search_dir, tt, rtt, r, static_cast<stop_idx_t>(i),
          stop{s}.location_idx(), iv,
          search_dir == direction::kForward
              ? interval{iv.from_, iv.to_ + max_start_offset}
              : interval{iv.from_ - max_start_offset, iv.to_},
          location_offset, starts);
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
             (i == 0U || !stp.out_allowed(p))) ||
            (search_dir == direction::kForward &&
             (i == location_seq.size() - 1 || !stp.in_allowed(p)))) {
          trace_start("    skip: i={}, out_allowed={}, in_allowed={}\n", i,
                      stp.out_allowed(p), stp.in_allowed(p));
          continue;
        }

        auto const ev_time = rtt->unix_event_time(
            rt_t, static_cast<stop_idx_t>(i),
            (search_dir == direction::kForward ? event_type::kDep
                                               : event_type::kArr));
        auto const d = get_duration(search_dir, ev_time, location_offset);
        auto const time_at_start =
            search_dir == direction::kForward ? ev_time - d : ev_time + d;

        if (!iv.contains(time_at_start)) {
          continue;
        }

        auto const& inserted =
            starts.emplace_back(start{.time_at_start_ = time_at_start,
                                      .time_at_stop_ = ev_time,
                                      .stop_ = l});
        trace_start(
            "        => ADD RT START: time_at_start={}, time_at_stop={}, "
            "stop={}\n",
            inserted.time_at_start_, inserted.time_at_stop_,
            loc{tt, starts.back().stop_});
      }
    }
  }

  // Add one earliest arrival query at the end of the interval. This is only
  // used to dominate journeys from the interval that are suboptimal
  // considering a journey from outside the interval (i.e. outside journey
  // departs later and arrives at the same time). These journeys outside the
  // interval will be filtered out before returning the result.
  if (add_ontrip) {
    auto const time_at_start =
        search_dir == direction::kForward ? iv.to_ : iv.from_ - 1_minutes;
    auto const d =
        get_duration(search_dir, time_at_start, location_offset, false);
    if (d != footpath::kMaxDuration) {
      starts.emplace_back(
          start{.time_at_start_ = time_at_start,
                .time_at_stop_ = search_dir == direction::kForward
                                     ? iv.to_ + d
                                     : iv.from_ - 1_minutes - d,
                .stop_ = l});
    }
  }
}

void get_starts(
    direction const search_dir,
    timetable const& tt,
    rt_timetable const* rtt,
    start_time_t const& start_time,
    std::vector<offset> const& start_offsets,
    hash_map<location_idx_t, std::vector<td_offset>> const& start_td_offsets,
    std::vector<via_stop> const& via_stops,
    duration_t const max_start_offset,
    location_match_mode const mode,
    bool const use_start_footpaths,
    std::vector<start>& starts,
    bool const add_ontrip,
    profile_idx_t const prf_idx,
    transfer_time_settings const& tts) {
  auto shortest_start = hash_map<location_idx_t, duration_t>{};
  auto const update = [&](location_idx_t const l, duration_t const offset) {
    auto const d =
        offset + (via_stops.empty() || via_stops.front().location_ != l
                      ? 0_minutes
                      : via_stops.front().stay_);
    auto& val = utl::get_or_create(shortest_start, l, [d]() { return d; });
    val = std::min(val, d);
  };

  auto const fwd = search_dir == direction::kForward;
  for (auto const& o : start_offsets) {
    for_each_meta(tt, mode, o.target(), [&](location_idx_t const l) {
      update(l, o.duration());
      if (use_start_footpaths) {
        auto const footpaths = fwd ? tt.locations_.footpaths_out_[prf_idx][l]
                                   : tt.locations_.footpaths_in_[prf_idx][l];
        for (auto const& fp : footpaths) {
          update(fp.target(),
                 o.duration() + adjusted_transfer_time(tts, fp.duration()));
        }
      }
    });
  }

  for (auto const& s : shortest_start) {
    auto const l = s.first;
    auto const o = s.second;

    std::visit(utl::overloaded{[&](interval<unixtime_t> const interval) {
                                 add_starts_in_interval(
                                     search_dir, tt, rtt, interval, l, o,
                                     max_start_offset, prf_idx, starts,
                                     add_ontrip);
                               },
                               [&](unixtime_t const t) {
                                 starts.emplace_back(
                                     start{.time_at_start_ = t,
                                           .time_at_stop_ = fwd ? t + o : t - o,
                                           .stop_ = l});
                               }},
               start_time);
  }

  for (auto const& [stop, offsets] : start_td_offsets) {
    std::visit(
        utl::overloaded{
            [&](interval<unixtime_t> const interval) {
              add_starts_in_interval(search_dir, tt, rtt, interval, stop,
                                     location_offset_t{std::span{offsets}},
                                     max_start_offset, prf_idx, starts,
                                     add_ontrip);
            },
            [&](unixtime_t const t) {
              auto const d = get_duration(search_dir, t, offsets, false);
              if (d != footpath::kMaxDuration) {
                starts.emplace_back(start{.time_at_start_ = t,
                                          .time_at_stop_ = fwd ? t + d : t - d,
                                          .stop_ = stop});
              }
            }},
        start_time);
  }
}

void collect_destinations(timetable const& tt,
                          std::vector<offset> const& dest,
                          location_match_mode const match_mode,
                          bitvec& is_dest,
                          std::vector<std::uint16_t>& dist_to_dest) {
  is_dest.resize(tt.n_locations());
  utl::fill(is_dest.blocks_, 0U);

  static constexpr auto const kIntermodalTarget =
      to_idx(get_special_station(special_station::kEnd));

  if (match_mode == location_match_mode::kIntermodal) {
    is_dest.set(kIntermodalTarget, true);
    dist_to_dest.resize(tt.n_locations());
    utl::fill(dist_to_dest, std::numeric_limits<std::uint16_t>::max());
  } else {
    dist_to_dest.clear();
  }

  for (auto const& d : dest) {
    trace_start("DEST METAS OF {}\n", loc{tt, d.target_});
    for_each_meta(tt, match_mode, d.target_, [&](location_idx_t const l) {
      if (match_mode == location_match_mode::kIntermodal) {
        dist_to_dest[to_idx(l)] =
            static_cast<std::uint16_t>(d.duration_.count());
      } else {
        is_dest.set(to_idx(l), true);
      }
      trace_start("  DEST META: {}, duration={}\n", loc{tt, l}, d.duration_);
    });
  }
}

void collect_via_destinations(timetable const& tt,
                              location_idx_t const via,
                              bitvec& is_destination) {
  is_destination.resize(tt.n_locations());
  utl::fill(is_destination.blocks_, 0U);

  trace_start("VIA METAS OF {}\n", loc{tt, via});
  for_each_meta(tt, location_match_mode::kEquivalent, via,
                [&](location_idx_t const l) {
                  is_destination.set(to_idx(l), true);
                  trace_start("  VIA META: {}\n", loc{tt, l});
                });
}

}  // namespace nigiri::routing
