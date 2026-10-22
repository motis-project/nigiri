#include "nigiri/loader/virt/route_stop_pairs.h"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <span>

#include "utl/get_or_create.h"

#include "nigiri/timetable.h"
#include "nigiri/transfer_rule_sides.h"

namespace nigiri::loader {

template <typename Fn>
void for_each_transfer(timetable const& tt,
                       route_idx_t const rx,
                       route_idx_t const ry,
                       std::span<delta const> arrivals,
                       std::span<delta const> departures,
                       window_t const& hull,
                       Fn&& fn) {
  auto const traffic_days =
      [&](route_idx_t const r,
          route_transport_idx_t const u) -> bitfield const& {
    return tt.bitfields_[tt.transport_traffic_days_
                             [tt.route_transport_ranges_[r].from_ + to_idx(u)]];
  };
  auto const mam_of = [](delta const d) { return i32_minutes{d.mam()}; };
  auto const arr_day = i32_minutes{date::days{arrivals.front().days()}};
  auto const dep_day = i32_minutes{date::days{departures.front().days()}};
  assert(arrivals.front().days() == arrivals.back().days());
  assert(departures.front().days() == departures.back().days());
  auto const first_dep = dep_day + mam_of(departures.front());
  auto const last_dep = dep_day + mam_of(departures.back());
  auto const lo = hull.from_;
  auto const hi = hull.to_ - duration_t{1};
  auto const n_arrivals = route_transport_idx_t{arrivals.size()};
  for (auto a = route_transport_idx_t{0U}; a != n_arrivals; ++a) {
    auto const arr = arr_day + mam_of(arrivals[to_idx(a)]);
    auto const& a_days = traffic_days(rx, a);
    for (auto offset = std::chrono::floor<date::days>(arr + lo - last_dep);
         offset <= std::chrono::floor<date::days>(arr + hi - first_dep);
         ++offset) {
      auto const arrival = arr - offset;
      auto const dep_from = arrival + lo - dep_day;
      auto const dep_to = arrival + hi - dep_day;
      auto it = std::lower_bound(
          begin(departures), end(departures), dep_from,
          [&](delta const d, i32_minutes const t) { return mam_of(d) < t; });
      if (it == end(departures) || mam_of(*it) >= dep_to) {
        continue;
      }
      auto const n = offset.count();
      auto const days = n >= 0 ? a_days << static_cast<std::size_t>(n)
                               : a_days >> static_cast<std::size_t>(-n);
      for (; it != end(departures) && mam_of(*it) < dep_to; ++it) {
        auto const b = route_transport_idx_t{it - begin(departures)};
        if (rx == ry && a == b) {
          continue;
        }
        if ((days & traffic_days(ry, b)).none()) {
          continue;
        }
        fn(a, b,
           std::chrono::duration_cast<duration_t>(dep_day + mam_of(*it) -
                                                  arrival));
      }
    }
  }
}

void add_transport_transfers(timetable const& tt,
                             location_idx_t const from,
                             location_idx_t const to,
                             transfer_durations const& pair_durations,
                             location_route_stops_t::bucket from_route_stops,
                             location_route_stops_t::bucket to_route_stops,
                             auto&& get_expected_duration,
                             route_stop_pairs& pairs) {
  for (auto& from_route_stop : from_route_stops) {
    if (!from_route_stop.has_arr_ || from_route_stop.sigs_.empty()) {
      continue;
    }
    for (auto& to_route_stop : to_route_stops) {
      if (!to_route_stop.has_dep_ || to_route_stop.sigs_.empty()) {
        continue;
      }
      auto const is_uniform =
          from_route_stop.is_uniform() && to_route_stop.is_uniform();
      auto uniform_expected_duration = duration_t{};
      auto p = route_stop_pair_idx_t::invalid();
      for_each_transfer(
          tt, from_route_stop.route_, to_route_stop.route_,
          tt.event_times_at_stop(from_route_stop.route_,
                                 from_route_stop.stop_idx_, event_type::kArr),
          tt.event_times_at_stop(to_route_stop.route_, to_route_stop.stop_idx_,
                                 event_type::kDep),
          pair_durations.hull_,
          [&](route_transport_idx_t const a, route_transport_idx_t const b,
              duration_t const gap) {
            if (p == route_stop_pair_idx_t::invalid()) {
              auto const from_route_stop_idx =
                  pairs.get_or_create_route_stop(from_route_stop, from);
              auto const to_route_stop_idx =
                  pairs.get_or_create_route_stop(to_route_stop, to);
              p = pairs.get_or_create_pair(
                  from_route_stop_idx, to_route_stop_idx, pair_durations.hull_);
              uniform_expected_duration = get_expected_duration(
                  signatures::get(from_route_stop.sigs_, a),
                  signatures::get(to_route_stop.sigs_, b));
            }
            auto const e = is_uniform
                               ? uniform_expected_duration
                               : get_expected_duration(
                                     signatures::get(from_route_stop.sigs_, a),
                                     signatures::get(to_route_stop.sigs_, b));
            pairs.add_transfer(p, transfer{a, b, gap, e});
          });
    }
  }
}

void add_base_transfers(timetable const& tt,
                        location_idx_t const from,
                        location_idx_t const to,
                        transfer_durations const& pair_durations,
                        location_route_stops_t::bucket from_route_stops,
                        location_route_stops_t::bucket to_route_stops,
                        auto&& get_expected_duration,
                        route_stop_pairs& pairs) {
  for (auto const is_arr : {true, false}) {
    auto const l = is_arr ? from : to;
    for (auto& s : is_arr ? from_route_stops : to_route_stops) {
      if (s.sigs_.empty() || !(is_arr ? s.has_arr_ : s.has_dep_)) {
        continue;
      }
      auto const from_route_stop_idx =
          is_arr ? pairs.get_or_create_route_stop(s, l)
                 : pairs.get_or_create_route_stop(to_base_stop(from));
      auto const to_route_stop_idx =
          is_arr ? pairs.get_or_create_route_stop(to_base_stop(to))
                 : pairs.get_or_create_route_stop(s, l);
      auto const p = pairs.get_or_create_pair(
          from_route_stop_idx, to_route_stop_idx, pair_durations.hull_);
      auto const n = static_cast<std::uint32_t>(
          tt.route_transport_ranges_[s.route_].size());
      for (auto t = route_transport_idx_t{0U};
           t != route_transport_idx_t{s.is_uniform() ? 1U : n}; ++t) {
        auto const e =
            is_arr
                ? get_expected_duration(signatures::get(s.sigs_, t), kEmptySig)
                : get_expected_duration(kEmptySig, signatures::get(s.sigs_, t));
        for (auto const gap : {e, e - duration_t{1}}) {
          pairs.add_transfer(p, transfer{is_arr ? t : kBaseTransport,
                                         is_arr ? kBaseTransport : t, gap, e});
        }
      }
    }
  }
}

route_stop_pairs get_route_stop_pairs(
    timetable const& tt,
    signatures const& sigs,
    hash_map<transfer_pair, transfer_durations> const& durations_of) {
  auto pairs = route_stop_pairs{};
  pairs.of_route_.resize(tt.n_routes());
  auto location_route_stops = get_location_route_stops(tt, sigs, durations_of);
  auto expected_duration_cache =
      hash_map<pair<signature_idx_t, signature_idx_t>, duration_t>{};
  for (auto const& [from_to, pair_durations] : durations_of) {
    auto const& [from, to] = from_to;
    expected_duration_cache.clear();
    auto const get_expected_duration = [&](signature_idx_t const from_sig,
                                           signature_idx_t const to_sig) {
      return utl::get_or_create(
          expected_duration_cache, pair{from_sig, to_sig}, [&]() {
            auto const rule = get_change_rule(tt, sigs.rules_[from_sig],
                                              sigs.rules_[to_sig], from, to);
            if (rule == transfer_rule_idx_t::invalid()) {
              return pair_durations.fallback_;
            }
            auto const& r = tt.transfer_rules_.rules_[rule];
            return r.is_forbidden() ? kNever : r.duration_;
          });
    };
    auto const from_route_stops = location_route_stops[from];
    auto const to_route_stops = location_route_stops[to];
    add_base_transfers(tt, from, to, pair_durations, from_route_stops,
                       to_route_stops, get_expected_duration, pairs);
    add_transport_transfers(tt, from, to, pair_durations, from_route_stops,
                            to_route_stops, get_expected_duration, pairs);
  }
  return pairs;
}

}  // namespace nigiri::loader
