#include "nigiri/routing/start_times.h"

#include "utl/enumerate.h"
#include "utl/equal_ranges_linear.h"
#include "utl/get_or_create.h"
#include "utl/helpers/algorithm.h"
#include "utl/lookup.h"
#include "utl/overloaded.h"
#include "utl/visit.h"

#include "nigiri/for_each_meta.h"
#include "nigiri/location_routes.h"
#include "nigiri/routing/search_location.h"
#include "nigiri/routing/transfers.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/special_stations.h"

namespace nigiri::routing {

constexpr auto const kTracing = false;

using location_offset_t = std::variant<duration_t, std::span<td_offset const>>;

std::optional<duration_t> get_duration(direction const search_dir,
                                       unixtime_t const t,
                                       location_offset_t const o,
                                       bool const invert = true) {
  return utl::visit(
      o, [](duration_t const x) -> std::optional<duration_t> { return x; },
      [&](std::span<td_offset const> td) {
        return get_td_duration(invert ? flip(search_dir) : search_dir, td, t)
            .transform([](auto const& x) { return x.first; });
      });
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
                             via_offset_t const v,
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
                       : rtt->traffic_days(rtt->transport_traffic_days_[t]);
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
        if (!d.has_value()) {
          trace_start("        {} => infeasible\n", ev_time);
          continue;
        }
        trace_start("        {} => duration={}\n", ev_time, *d);
        auto const time_at_start =
            search_dir == direction::kForward ? ev_time - *d : ev_time + *d;
        if (!iv_at_start.contains(time_at_start)) {
          trace_start("      iv_at_start={} doesn't contain time_at_start={}\n",
                      iv_at_start, time_at_start);
          continue;
        }
        if (!starts.empty() && starts.back().time_at_start_ == time_at_start &&
            starts.back().v_ == v &&
            is_better_or_eq(starts.back().time_at_stop_, ev_time)) {
          trace_start("      time_at_start={} -> no improvement\n", iv_at_start,
                      time_at_start);
          continue;
        }
        auto const& s =
            starts.emplace_back(start{.time_at_start_ = time_at_start,
                                      .time_at_stop_ = ev_time,
                                      .stop_ = location_idx,
                                      .v_ = v});
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
                            via_offset_t const v,
                            duration_t const max_start_offset,
                            profile_idx_t const p,
                            std::vector<start>& starts,
                            bool const add_ontrip) {
  auto const routes = static_routes(tt, rtt, l);
  trace_start(
      "    add_starts_in_interval(interval={}, stop={}): {} "
      "routes\n",
      iv, loc{tt, l},  // NOLINT(clang-analyzer-core.CallAndMessage)
      routes.size());

  for (auto const& r : routes) {

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
          location_offset, v, starts);
    }
  }

  // Real-time starts
  if (rtt != nullptr) {
    for (auto const& rt_t : rtt->location_rt_transports_.at(l)) {
      if (rtt->rt_transport_is_cancelled_[to_idx(rt_t)]) {
        continue;
      }
      auto const location_seq = rtt->rt_transport_location_seq_.at(rt_t);
      for (auto const [i, s] : utl::enumerate(location_seq)) {
        auto const stp = stop{s};
        if (search_location(*rtt, p, rt_t, static_cast<stop_idx_t>(i)) != l) {
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
        if (!d.has_value()) {
          trace_start("        {} => infeasible\n", ev_time);
          continue;
        }
        auto const time_at_start =
            search_dir == direction::kForward ? ev_time - *d : ev_time + *d;

        if (!iv.contains(time_at_start)) {
          continue;
        }

        auto const& inserted =
            starts.emplace_back(start{.time_at_start_ = time_at_start,
                                      .time_at_stop_ = ev_time,
                                      .stop_ = l,
                                      .v_ = v});
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
    if (d.has_value()) {
      starts.emplace_back(
          start{.time_at_start_ = time_at_start,
                .time_at_stop_ = search_dir == direction::kForward
                                     ? iv.to_ + *d
                                     : iv.from_ - 1_minutes - *d,
                .stop_ = l,
                .v_ = v});
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
  // A via at the start holds wherever the search counts it (see
  // collect_via_destinations).
  auto start_via = hash_set<location_idx_t>{};
  if (!via_stops.empty()) {
    for_each_meta(tt, nullptr, location_match_mode::kEquivalent,
                  via_stops.front().location_,
                  [&](location_idx_t const l) { start_via.emplace(l); });
  }

  auto const virt_rtt = rt_virts(rtt, prf_idx);
  auto shortest_start = std::array<hash_map<location_idx_t, duration_t>, 2>{};
  auto at_start = std::array<hash_map<location_idx_t, duration_t>, 2>{};
  auto const update_min = [&](via_offset_t const v, location_idx_t const l,
                              duration_t const d, bool const is_walked) {
    auto const keep = [&](hash_map<location_idx_t, duration_t>& m) {
      auto& val = utl::get_or_create(m, l, [d]() { return d; });
      val = std::min(val, d);
    };
    keep(shortest_start[v]);
    if (!is_walked) {
      keep(at_start[v]);
    }
  };

  // is_walked_from_via: the stay can be spent at the via before the walk.
  auto const update = [&](location_idx_t const l, duration_t const offset,
                          bool const is_walked,
                          bool const is_walked_from_via = false) {
    if (!is_walked_from_via &&
        !start_via.contains(rtt == nullptr ? l : rtt->static_location(l))) {
      update_min(0U, l, offset, is_walked);
      return;
    }
    update_min(1U, l, offset + via_stops.front().stay_, is_walked);
    if (via_stops.front().stay_ != 0_minutes) {
      update_min(0U, l, offset, is_walked);
    }
  };

  auto const fwd = search_dir == direction::kForward;
  for (auto const& o : start_offsets) {
    for_each_meta(tt, virt_rtt, mode, o.target(), [&](location_idx_t const l) {
      update(l, o.duration(), false);
    });
  }
  if (use_start_footpaths) {
    for (auto const& o : start_offsets) {
      for_each_meta(
          tt, virt_rtt, mode, o.target(), [&](location_idx_t const l) {
            // A walk leaves a start stop from the stop: the walks of its
            // virtual locations hold for the trips stopping there.
            if (tt.locations_.is_virt(l) &&
                utl::any_of(at_start, [&](auto const& m) {
                  return m.contains(rt::base(tt, rtt, l));
                })) {
              return;
            }
            auto const is_via = start_via.contains(
                rtt == nullptr ? l : rtt->static_location(l));
            for (auto const fp :
                 get_transfers(tt, rtt, prf_idx, search_dir, l)) {
              update(fp.target(),
                     o.duration() + adjusted_transfer_time(tts, fp.duration()),
                     true, is_via);
            }
          });
    }
  }

  for (auto v = via_offset_t{0U}; v != shortest_start.size(); ++v) {
    for (auto const& [l, o] : shortest_start[v]) {
      std::visit(utl::overloaded{[&](interval<unixtime_t> const interval) {
                                   add_starts_in_interval(
                                       search_dir, tt, rtt, interval, l, o, v,
                                       max_start_offset, prf_idx, starts,
                                       add_ontrip);
                                 },
                                 [&](unixtime_t const t) {
                                   starts.emplace_back(start{
                                       .time_at_start_ = t,
                                       .time_at_stop_ = fwd ? t + o : t - o,
                                       .stop_ = l,
                                       .v_ = v});
                                 }},
                 start_time);
    }
  }

  for (auto const& [stop, offsets] : start_td_offsets) {
    auto const v = static_cast<via_offset_t>(
        start_via.contains(rtt == nullptr ? stop : rtt->static_location(stop))
            ? 1U
            : 0U);
    auto const add_td_start = [&](location_idx_t const l) {
      std::visit(
          utl::overloaded{[&](interval<unixtime_t> const interval) {
                            add_starts_in_interval(
                                search_dir, tt, rtt, interval, l,
                                location_offset_t{std::span{offsets}}, v,
                                max_start_offset, prf_idx, starts, add_ontrip);
                          },
                          [&](unixtime_t const t) {
                            auto const d =
                                get_duration(search_dir, t, offsets, false);
                            if (d.has_value()) {
                              starts.emplace_back(
                                  start{.time_at_start_ = t,
                                        .time_at_stop_ = fwd ? t + *d : t - *d,
                                        .stop_ = l,
                                        .v_ = v});
                            }
                          }},
          start_time);
    };
    add_td_start(stop);
    if (auto const r = virt_rtt; r != nullptr) {
      r->for_each_rt_virt(stop, add_td_start);
    }
  }
}

void expand_td_offsets(timetable const& tt, query& q) {
  auto const add = [&](hash_map<location_idx_t, std::vector<td_offset>>& td) {
    auto children = std::vector<std::pair<location_idx_t, location_idx_t>>{};
    for (auto const& [l, _] : td) {
      for_each_meta(tt, nullptr, location_match_mode::kIntermodal, l,
                    [&](location_idx_t const c) {
                      if (c != l) {
                        children.emplace_back(c, l);
                      }
                    });
    }
    for (auto const& [c, l] : children) {
      auto offsets = td.at(l);
      td.emplace(c, std::move(offsets));
    }
  };
  add(q.td_start_);
  add(q.td_dest_);
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
    for_each_meta(
        tt, nullptr, match_mode, d.target_, [&](location_idx_t const l) {
          if (match_mode == location_match_mode::kIntermodal) {
            dist_to_dest[to_idx(l)] =
                std::min(dist_to_dest[to_idx(l)],
                         static_cast<std::uint16_t>(d.duration_.count()));
          } else if (!tt.locations_.is_virt(l)) {
            is_dest.set(to_idx(l), true);
          }
          trace_start("  DEST META: {}, duration={}\n", loc{tt, l},
                      d.duration_);
        });
  }
}

void collect_via_destinations(timetable const& tt,
                              location_idx_t const via,
                              bitvec& is_destination) {
  is_destination.resize(tt.n_locations());
  utl::fill(is_destination.blocks_, 0U);

  trace_start("VIA METAS OF {}\n", loc{tt, via});
  for_each_meta(tt, nullptr, location_match_mode::kEquivalent, via,
                [&](location_idx_t const l) {
                  is_destination.set(to_idx(l), true);
                  trace_start("  VIA META: {}\n", loc{tt, l});
                });
}

}  // namespace nigiri::routing
