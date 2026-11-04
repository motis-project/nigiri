#include "nigiri/loader/virt/route_stop_pairs.h"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <span>
#include <vector>

#include "utl/get_or_create.h"

#include "nigiri/stop.h"
#include "nigiri/timetable.h"
#include "nigiri/transfer_rule_sides.h"

namespace nigiri::loader {

struct location_route_stop {
  bool is_uniform() const { return sigs_.size() <= 1U; }

  route_idx_t route_;
  stop_idx_t stop_idx_;
  std::span<signature_idx_t const> sigs_;
  std::span<delta const> arr_;  // empty: no arrival (exit) at this stop
  std::span<delta const> dep_;  // empty: no departure (entry) at this stop
  route_stop_idx_t route_stop_{route_stop_idx_t::invalid()};
};

using location_route_stops_t = vecvec<location_idx_t, location_route_stop>;

location_route_stops_t get_location_route_stops(
    timetable const& tt,
    import_context const& ctx,
    signatures const& sigs,
    hash_map<transfer_pair, transfer_durations> const& durations_of,
    route_stop_pairs& pairs) {
  // Mark any stop that's involved in a transfer rule.
  auto is_needed = bitvec_map<location_idx_t>{};
  is_needed.resize(tt.n_locations());
  for (auto const& [stops, _] : durations_of) {
    is_needed.set(stops.from_);
    is_needed.set(stops.to_);
  }

  // For each location: collect all route stops with their signatures.
  auto route_stops = location_route_stops_t{};
  auto v = std::vector<location_route_stop>{};
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    v.clear();
    if (is_needed.test(l)) {
      for (auto const r : tt.location_routes_[l]) {
        auto const seq = tt.route_location_seq_[r];
        for (auto i = stop_idx_t{0U}; i != seq.size(); ++i) {
          auto const stp = stop{seq[i]};
          if (stp.location_idx() != l || sigs.get(r, i).empty()) {
            continue;
          }
          auto const times = [&](bool const has, event_type const ev) {
            return has ? get_stop_times(tt, ctx, r, i, ev)
                       : std::span<delta const>{};
          };
          auto s = location_route_stop{
              .route_ = r,
              .stop_idx_ = i,
              .sigs_ = sigs.get(r, i),
              .arr_ = times(i != 0U && stp.out_allowed(), event_type::kArr),
              .dep_ = times(i + 1U != seq.size() && stp.in_allowed(),
                            event_type::kDep)};
          s.route_stop_ = pairs.get_or_create_route_stop(
              s.is_uniform() ? to_signature_stop(l, s.sigs_.front())
                             : route_stop{r, i, l});
          v.push_back(s);
        }
      }
    }
    route_stops.emplace_back(v);
  }
  return route_stops;
}

template <typename Fn>
void for_each_transfer(timetable const& tt,
                       route_idx_t const rx,
                       route_idx_t const ry,
                       std::span<delta const> const arrivals,
                       std::span<delta const> const departures,
                       window_t const& hull,
                       std::vector<std::size_t>& first,
                       Fn&& fn) {
  auto const traffic_days =
      [&](route_idx_t const r,
          route_transport_idx_t const u) -> bitfield const& {
    return tt.bitfields_[tt.transport_traffic_days_
                             [tt.route_transport_ranges_[r].from_ + to_idx(u)]];
  };
  auto const mam_of = [](delta const d) { return i32_minutes{d.mam()}; };
  auto const day_of = [](i32_minutes const t) {
    return std::chrono::floor<date::days>(t);
  };
  auto const arr_day = i32_minutes{date::days{arrivals.front().days()}};
  auto const dep_day = i32_minutes{date::days{departures.front().days()}};
  assert(arrivals.front().days() == arrivals.back().days());
  assert(departures.front().days() == departures.back().days());
  auto const first_dep = dep_day + mam_of(departures.front());
  auto const last_dep = dep_day + mam_of(departures.back());
  auto const lo = hull.from_;
  auto const hi = hull.to_ - duration_t{1};

  auto const min_offset =
      day_of(arr_day + mam_of(arrivals.front()) + lo - last_dep);
  auto const max_offset =
      day_of(arr_day + mam_of(arrivals.back()) + hi - first_dep);
  first.assign(static_cast<std::size_t>((max_offset - min_offset).count() + 1),
               0U);

  auto const n_arrivals = route_transport_idx_t{arrivals.size()};
  for (auto a = route_transport_idx_t{0U}; a != n_arrivals; ++a) {
    auto const arr = arr_day + mam_of(arrivals[to_idx(a)]);
    auto const& a_days = traffic_days(rx, a);
    for (auto offset = day_of(arr + lo - last_dep);
         offset <= day_of(arr + hi - first_dep); ++offset) {
      auto const arrival = arr - offset;
      auto const dep_from = arrival + lo - dep_day;
      auto const dep_to = arrival + hi - dep_day;
      auto& i = first[static_cast<std::size_t>((offset - min_offset).count())];
      while (i != departures.size() && mam_of(departures[i]) < dep_from) {
        ++i;
      }
      if (i == departures.size() || mam_of(departures[i]) >= dep_to) {
        continue;
      }
      auto const n = offset.count();
      auto const days = n >= 0 ? a_days << static_cast<std::size_t>(n)
                               : a_days >> static_cast<std::size_t>(-n);
      for (auto j = i; j != departures.size() && mam_of(departures[j]) < dep_to;
           ++j) {
        auto const b = route_transport_idx_t{j};
        if (rx == ry && a == b && n == 0) {
          continue;
        }
        if ((days & traffic_days(ry, b)).none()) {
          continue;
        }
        fn(a, b,
           std::chrono::duration_cast<duration_t>(
               dep_day + mam_of(departures[j]) - arrival));
      }
    }
  }
}

void add_transport_transfers(
    timetable const& tt,
    transfer_durations const& pair_durations,
    location_route_stops_t::const_bucket from_route_stops,
    location_route_stops_t::const_bucket to_route_stops,
    auto&& get_expected_duration,
    route_stop_pairs& pairs) {
  auto first = std::vector<std::size_t>{};
  for (auto const& from_route_stop : from_route_stops) {
    if (from_route_stop.arr_.empty()) {
      continue;
    }
    for (auto const& to_route_stop : to_route_stops) {
      if (to_route_stop.dep_.empty()) {
        continue;
      }
      auto p = route_stop_pair_idx_t::invalid();
      for_each_transfer(
          tt, from_route_stop.route_, to_route_stop.route_,
          from_route_stop.arr_, to_route_stop.dep_, pair_durations.hull_, first,
          [&](route_transport_idx_t const a, route_transport_idx_t const b,
              duration_t const gap) {
            if (p == route_stop_pair_idx_t::invalid()) {
              p = pairs.get_or_create_pair(from_route_stop.route_stop_,
                                           to_route_stop.route_stop_,
                                           pair_durations);
            }
            pairs.add_transfer(
                p, transfer{a, b, gap,
                            get_expected_duration(
                                signatures::get(from_route_stop.sigs_, a),
                                signatures::get(to_route_stop.sigs_, b))});
          });
    }
  }
}

void add_base_transfers(timetable const& tt,
                        location_idx_t const from,
                        location_idx_t const to,
                        transfer_durations const& pair_durations,
                        location_route_stops_t::const_bucket from_route_stops,
                        location_route_stops_t::const_bucket to_route_stops,
                        auto&& get_expected_duration,
                        route_stop_pairs& pairs) {
  for (auto const is_arr : {true, false}) {
    for (auto const& s : is_arr ? from_route_stops : to_route_stops) {
      if ((is_arr ? s.arr_ : s.dep_).empty()) {
        continue;
      }
      auto const from_route_stop_idx =
          is_arr ? s.route_stop_
                 : pairs.get_or_create_route_stop(to_base_stop(from));
      auto const to_route_stop_idx =
          is_arr ? pairs.get_or_create_route_stop(to_base_stop(to))
                 : s.route_stop_;
      auto const p = pairs.get_or_create_pair(
          from_route_stop_idx, to_route_stop_idx, pair_durations);
      auto const n = static_cast<std::uint32_t>(
          tt.route_transport_ranges_[s.route_].size());
      for (auto t = route_transport_idx_t{0U};
           t != route_transport_idx_t{s.is_uniform() ? 1U : n}; ++t) {
        auto const e =
            is_arr
                ? get_expected_duration(signatures::get(s.sigs_, t), kEmptySig)
                : get_expected_duration(kEmptySig, signatures::get(s.sigs_, t));

        // Enforce expected duration for transfer from/to base:
        // Window [e,e] for base transfers (rules qualified on maximum one side)
        // -> exact duration is enforced with gap=e + gap=e-1 transfers:
        //    - gap=e, expected=e => works => transfer duration *at most* e
        //    - gap=e-1, expected=e => fails => transfer duration *at least* e
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
    import_context const& ctx,
    signatures const& sigs,
    hash_map<transfer_pair, transfer_durations> const& durations_of) {
  auto pairs = route_stop_pairs{};
  pairs.of_route_.resize(tt.n_routes());
  auto const location_route_stops =
      get_location_route_stops(tt, ctx, sigs, durations_of, pairs);
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
    add_transport_transfers(tt, pair_durations, from_route_stops,
                            to_route_stops, get_expected_duration, pairs);
  }
  return pairs;
}

}  // namespace nigiri::loader
