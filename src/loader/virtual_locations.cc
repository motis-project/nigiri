#include "nigiri/loader/virtual_locations.h"

#include <cstdint>
#include <algorithm>
#include <deque>
#include <iterator>
#include <limits>
#include <optional>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "utl/equal_ranges_linear.h"
#include "utl/erase_duplicates.h"
#include "utl/get_or_create.h"
#include "utl/helpers/algorithm.h"
#include "utl/lookup.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "nigiri/loader/register.h"
#include "nigiri/loader/transfer_rules.h"
#include "nigiri/footpath.h"
#include "nigiri/logging.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/transfers.h"
#include "nigiri/shapes_storage.h"
#include "nigiri/stop.h"
#include "nigiri/timetable.h"
#include "nigiri/transfer_rule_sides.h"

namespace nigiri::loader {

using route_transport_idx_t =
    cista::strong<std::uint32_t, struct route_transport_>;
using part_idx_t = cista::strong<std::uint32_t, struct part_>;
using unit_idx_t = cista::strong<std::uint32_t, struct unit_>;
using place_idx_t = cista::strong<std::uint32_t, struct place_>;
using signature_id_t = cista::strong<std::uint32_t, struct signature_id_>;
using sides_t = std::vector<transfer_rule_side_idx>;
using stop_seq_t = std::vector<stop::value_type>;

constexpr auto const kNever = i32_minutes{routing::kMaxTravelTime}.count();
constexpr auto const kMinBound = std::numeric_limits<std::int32_t>::min();
constexpr auto const kMaxBound = std::numeric_limits<std::int32_t>::max();
constexpr auto const kMaxFootpath =
    static_cast<std::int32_t>(footpath::kMaxDuration.count()) - 1;
constexpr auto const kMaxTransferTime =
    static_cast<std::int32_t>(kNoTransferAllowed.count()) - 1;

struct route_stop {
  CISTA_COMPARABLE()
  route_idx_t route_;
  stop_idx_t idx_;
};

struct minutes_range {
  void extend(std::int32_t const m) {
    lo_ = std::min(lo_, m);
    hi_ = std::max(hi_, m);
  }
  std::int32_t lo_, hi_;
};

struct duration_bounds {
  bool contains(std::int32_t const d) const { return lo_ < d && d <= hi_; }
  void add(std::int32_t const gap, bool const must_work) {
    if (must_work) {
      hi_ = std::min(hi_, gap);
    } else {
      lo_ = std::max(lo_, gap);
    }
  }
  void intersect(duration_bounds const& o) {
    lo_ = std::max(lo_, o.lo_);
    hi_ = std::min(hi_, o.hi_);
  }
  std::int32_t lo_{kMinBound};
  std::int32_t hi_{kMaxBound};
};

struct virt_transfer {
  route_transport_idx_t from_, to_;
  std::int16_t gap_;
  std::int16_t expected_;
};

struct virt_node {
  CISTA_COMPARABLE()
  route_stop rs_;
  location_idx_t pinned_;
};

using node_pair = pair<virt_node, virt_node>;
using place_ref = pair<location_idx_t, place_idx_t>;

struct virt_unit {
  virt_node node_;
  part_idx_t part_;
  location_idx_t stop_;
  place_idx_t place_;
  bool is_movable_;
};

// A global rule applies at a station only if both its sides occur there.
hash_map<location_idx_t, std::vector<transfer_rule_idx_t>> get_global_rules(
    timetable const& tt) {
  auto const& tr = tt.transfer_rules_;
  auto global_rules =
      hash_map<location_idx_t, std::vector<transfer_rule_idx_t>>{};
  if (utl::none_of(tr.rules_,
                   [](transfer_rule const& r) { return r.is_global(); })) {
    return global_rules;
  }

  // the route ids (attribute groups) arriving / departing at each station
  using event_t =
      std::tuple<location_idx_t, route_id_idx_t, source_idx_t, event_type>;
  auto events = hash_set<event_t>{};
  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    auto const seq = tt.route_location_seq_[r];
    for (auto const t : tt.route_transport_ranges_[r]) {
      if (!has_transfer_rules(tt, t)) {
        continue;
      }
      auto const sections = tt.transport_to_trip_section_[t];
      for (auto i = stop_idx_t{0U}; i != seq.size(); ++i) {
        auto const stp = stop{seq[i]};
        auto const station = tt.locations_.get_root_idx(stp.location_idx());
        if (tr.is_global_excluded(station)) {
          continue;
        }
        auto const add = [&](std::size_t const section, event_type const ev) {
          for (auto const trip :
               tt.merged_trips_[sections[sections.size() == 1U ? 0U
                                                               : section]]) {
            if (auto const route = tt.trip_route_id_[trip];
                route != route_id_idx_t::invalid()) {
              events.emplace(station, route,
                             tt.trip_id_src_[tt.trip_ids_[trip].front()], ev);
            }
          }
        };
        if (i != 0U && stp.out_allowed()) {
          add(i - 1U, event_type::kArr);
        }
        if (i + 1U != seq.size() && stp.in_allowed()) {
          add(i, event_type::kDep);
        }
      }
    }
  }

  auto sorted = utl::to_vec(events);
  events = {};
  utl::sort(sorted);
  auto arr = std::vector<transfer_rule_idx_t>{};
  auto dep = std::vector<transfer_rule_idx_t>{};
  utl::equal_ranges_linear(
      sorted,
      [](event_t const& a, event_t const& b) {
        return std::get<0>(a) == std::get<0>(b);
      },
      [&](auto const from, auto const to) {
        arr.clear();
        dep.clear();
        for (auto it = from; it != to; ++it) {
          auto const [_, route, src, ev] = *it;
          for (auto const s : values_of(tr.route_rules_, route)) {
            auto const& rule = tr.rules_[s.rule()];
            if (rule.is_global() && rule.src_ == src && s.ev_type() == ev) {
              (ev == event_type::kArr ? arr : dep).push_back(s.rule());
            }
          }
        }
        utl::erase_duplicates(arr);
        utl::erase_duplicates(dep);
        auto both = std::vector<transfer_rule_idx_t>{};
        std::ranges::set_intersection(arr, dep, std::back_inserter(both));
        if (!both.empty()) {
          global_rules.emplace(std::get<0>(*from), std::move(both));
        }
      });
  return global_rules;
}

std::int32_t get_minutes(transfer_rule const& r) {
  return r.duration_ == footpath::kMaxDuration
             ? kNever
             : static_cast<std::int32_t>(r.duration_.count());
}

hash_map<transfer_pair, minutes_range> get_rule_times(timetable const& tt) {
  auto rule_times = hash_map<transfer_pair, minutes_range>{};
  for (auto const& rule : tt.transfer_rules_.rules_) {
    auto const m = get_minutes(rule);
    rule_times
        .emplace(transfer_pair{rule.from_stop_, rule.to_stop_},
                 minutes_range{m, m})
        .first->second.extend(m);
  }
  return rule_times;
}

bool has_event(timetable const& tt, route_stop const rs, event_type const ev) {
  auto const seq = tt.route_location_seq_[rs.route_];
  auto const stp = stop{seq[rs.idx_]};
  return ev == event_type::kArr
             ? rs.idx_ != 0U && stp.out_allowed()
             : rs.idx_ + 1U != seq.size() && stp.in_allowed();
}

std::optional<std::int32_t> pick_duration(
    duration_bounds const& b,
    std::int32_t const fallback,
    std::int32_t const max,
    hash_map<std::int32_t, std::uint32_t> const& expected = {}) {
  if (b.contains(fallback)) {
    return fallback;
  }
  if (b.hi_ >= kNever) {
    return kNever;
  }
  auto const hi = std::min(b.hi_, max);
  if (b.lo_ + 1 > hi) {
    return std::nullopt;
  }
  auto best = hi;
  auto best_n = std::uint32_t{0U};
  for (auto const& [e, n] : expected) {
    if (b.lo_ < e && e <= hi && (n > best_n || (n == best_n && e < best))) {
      best = e;
      best_n = n;
    }
  }
  return best;
}

struct virt_state {
  std::vector<sides_t> signatures_;
  hash_map<route_stop, vector_map<route_transport_idx_t, signature_id_t>>
      signature_of_;
  hash_map<pair<location_idx_t, location_idx_t>, std::int32_t> fallbacks_;
  hash_map<node_pair, std::vector<virt_transfer>> transfers_;
  hash_map<node_pair, duration_bounds> windows_;
  hash_map<route_idx_t, std::vector<node_pair>> route_keys_;
  hash_map<route_idx_t, vector_map<route_transport_idx_t, part_idx_t>> parts_;
  hash_map<route_idx_t, part_idx_t> n_parts_;
  vector_map<unit_idx_t, virt_unit> units_;
  hash_map<pair<virt_node, part_idx_t>, unit_idx_t> unit_of_;
  vector_map<unit_idx_t, std::vector<unit_idx_t>> out_, in_;
  hash_map<pair<unit_idx_t, unit_idx_t>, duration_bounds> bounds_;
  hash_map<pair<unit_idx_t, unit_idx_t>, hash_map<std::int32_t, std::uint32_t>>
      expected_;
  hash_map<node_pair, std::vector<pair<unit_idx_t, unit_idx_t>>> key_units_;
  hash_map<location_idx_t, vector_map<place_idx_t, std::vector<unit_idx_t>>>
      places_;
  hash_map<pair<place_ref, place_ref>, std::int32_t> durations_;
};

bool is_pinned(virt_node const& n) {
  return n.rs_.route_ == route_idx_t::invalid();
}

std::int32_t get_fallback(timetable const& tt,
                          virt_state& vs,
                          location_idx_t const from,
                          location_idx_t const to) {
  return utl::get_or_create(vs.fallbacks_, pair{from, to}, [&]() {
    auto const walk =
        routing::shortest_transfer(tt, nullptr, kDefaultProfile, from, to);
    return walk.has_value() ? static_cast<std::int32_t>(walk->count()) : kNever;
  });
}

void add_signatures(timetable const& tt, virt_state& vs) {
  auto const& tr = tt.transfer_rules_;
  auto const global_rules = get_global_rules(tt);
  auto ids = hash_map<sides_t, signature_id_t>{};
  vs.signatures_.emplace_back();
  ids.emplace(sides_t{}, signature_id_t{0U});
  auto sig = sides_t{};
  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    auto const transports = tt.route_transport_ranges_[r];
    auto const seq = tt.route_location_seq_[r];
    for (auto const t : transports) {
      if (!has_transfer_rules(tt, t)) {
        continue;
      }
      for (auto i = stop_idx_t{0U}; i != seq.size(); ++i) {
        auto const base = stop{seq[i]}.location_idx();
        get_stop_signature(tt, t, i, base, sig);
        auto const globals =
            global_rules.find(tt.locations_.get_root_idx(base));
        std::erase_if(sig, [&](transfer_rule_side_idx const s) {
          return tr.rules_[s.rule()].is_global() &&
                 (globals == end(global_rules) ||
                  !std::ranges::binary_search(globals->second, s.rule()));
        });
        if (sig.empty()) {
          continue;
        }
        auto& ids_at =
            utl::get_or_create(vs.signature_of_, route_stop{r, i}, [&]() {
              return vector_map<route_transport_idx_t, signature_id_t>(
                  transports.size(), signature_id_t{0U});
            });
        ids_at[route_transport_idx_t{to_idx(t - transports.from_)}] =
            utl::get_or_create(ids, sig, [&]() {
              vs.signatures_.push_back(sig);
              return signature_id_t{vs.signatures_.size() - 1U};
            });
      }
    }
  }
}

hash_set<pair<location_idx_t, location_idx_t>> get_stop_pairs(
    timetable const& tt, virt_state const& vs) {
  auto const& locs = tt.locations_;
  auto rule_stops = hash_set<location_idx_t>{};
  for (auto const& [rs, _] : vs.signature_of_) {
    rule_stops.insert(
        stop{tt.route_location_seq_[rs.route_][rs.idx_]}.location_idx());
  }
  auto const covered = [&](location_idx_t const s) {
    auto v = std::vector<location_idx_t>{s};
    for (auto const c : locs.children_[s]) {
      v.push_back(c);
    }
    return v;
  };
  auto pairs = hash_set<pair<location_idx_t, location_idx_t>>{};
  auto has_global = false;
  for (auto const& r : tt.transfer_rules_.rules_) {
    if (!r.is_qualified(event_type::kArr) &&
        !r.is_qualified(event_type::kDep)) {
      continue;
    }
    if (r.is_global()) {
      has_global = true;
      continue;
    }
    for (auto const f : covered(r.from_stop_)) {
      for (auto const t : covered(r.to_stop_)) {
        if (rule_stops.contains(f) || rule_stops.contains(t)) {
          pairs.insert(pair{f, t});
        }
      }
    }
  }
  if (has_global) {
    for (auto const l : rule_stops) {
      auto const root = locs.get_root_idx(l);
      if (!tt.transfer_rules_.is_global_excluded(root)) {
        for (auto const f : covered(root)) {
          for (auto const t : covered(root)) {
            pairs.insert(pair{f, t});
          }
        }
      }
    }
  }
  return pairs;
}

void add_transfers(timetable const& tt, virt_state& vs) {
  auto const& rules = tt.transfer_rules_.rules_;
  auto const& locs = tt.locations_;
  auto const rule_times = get_rule_times(tt);
  auto const stop_pairs = get_stop_pairs(tt, vs);
  auto const location_of = [&](route_stop const rs) {
    return stop{tt.route_location_seq_[rs.route_][rs.idx_]}.location_idx();
  };
  auto const pinned_at = [](location_idx_t const l) {
    return virt_node{route_stop{route_idx_t::invalid(), stop_idx_t{0U}}, l};
  };
  auto const node_of = [&](route_stop const rs) {
    return vs.signature_of_.contains(rs)
               ? virt_node{rs, location_idx_t::invalid()}
               : pinned_at(location_of(rs));
  };
  auto const signature = [&](route_stop const rs,
                             route_transport_idx_t const u) {
    auto const it = vs.signature_of_.find(rs);
    return it == end(vs.signature_of_) ? signature_id_t{0U} : it->second[u];
  };

  auto route_stops = hash_map<location_idx_t, std::vector<route_stop>>{};
  auto const get_route_stops =
      [&](location_idx_t const l) -> std::vector<route_stop> const& {
    return utl::get_or_create(route_stops, l, [&]() {
      auto v = std::vector<route_stop>{};
      for (auto const r : tt.location_routes_[l]) {
        auto const seq = tt.route_location_seq_[r];
        for (auto i = stop_idx_t{0U}; i != seq.size(); ++i) {
          if (stop{seq[i]}.location_idx() == l) {
            v.push_back(route_stop{r, i});
          }
        }
      }
      return v;
    });
  };
  using events_t = std::vector<std::pair<i32_minutes, route_transport_idx_t>>;
  auto event_store = std::deque<events_t>{};
  auto events = hash_map<pair<route_stop, event_type>, events_t*>{};
  auto const get_events = [&](route_stop const rs,
                              event_type const ev) -> events_t const& {
    return *utl::get_or_create(events, pair{rs, ev}, [&]() {
      auto const transports = tt.route_transport_ranges_[rs.route_];
      auto& v = event_store.emplace_back();
      for (auto u = route_transport_idx_t{0U}; u != transports.size(); ++u) {
        v.emplace_back(
            i32_minutes{tt.event_mam(rs.route_, transports.from_ + to_idx(u),
                                     rs.idx_, ev)
                            .as_duration()},
            u);
      }
      utl::sort(v);
      return &v;
    });
  };
  auto const traffic_days =
      [&](route_idx_t const r,
          route_transport_idx_t const u) -> bitfield const& {
    return tt.bitfields_[tt.transport_traffic_days_
                             [tt.route_transport_ranges_[r].from_ + to_idx(u)]];
  };

  auto expected =
      hash_map<pair<signature_id_t, signature_id_t>, std::int32_t>{};
  auto transfers = std::vector<virt_transfer>{};
  for (auto const& [lx, ly] : stop_pairs) {
    auto const fallback = get_fallback(tt, vs, lx, ly);
    expected.clear();
    auto const get_expected = [&](signature_id_t const sa,
                                  signature_id_t const sb) {
      return utl::get_or_create(expected, pair{sa, sb}, [&]() {
        auto const rule = get_change_rule(tt, vs.signatures_[to_idx(sa)],
                                          vs.signatures_[to_idx(sb)], lx, ly);
        return rule == transfer_rule_idx_t::invalid()
                   ? fallback
                   : get_minutes(rules[rule]);
      });
    };
    auto window = minutes_range{fallback, fallback};
    auto const extend = [&](location_idx_t const from,
                            location_idx_t const to) {
      if (auto const r = utl::lookup(rule_times, transfer_pair{from, to})) {
        window.extend(r->lo_);
        window.extend(r->hi_);
      }
    };
    for (auto const from : {lx, locs.parents_[lx]}) {
      for (auto const to : {ly, locs.parents_[ly]}) {
        extend(from, to);
      }
    }
    if (auto const station = locs.get_root_idx(lx);
        station == locs.get_root_idx(ly) &&
        !tt.transfer_rules_.is_global_excluded(station)) {
      extend(location_idx_t::invalid(), location_idx_t::invalid());
    }
    auto const note = [&](node_pair const& key) {
      auto& w = vs.windows_[key];
      w.add(window.lo_ - 1, false);
      if (window.hi_ < kNever) {
        w.add(window.hi_, true);
      }
      for (auto const n : {key.first, key.second}) {
        if (!is_pinned(n)) {
          vs.route_keys_[n.rs_.route_].push_back(key);
        }
      }
    };
    for (auto const is_arr : {true, false}) {
      for (auto const rs : get_route_stops(is_arr ? lx : ly)) {
        if (!vs.signature_of_.contains(rs) ||
            !has_event(tt, rs, is_arr ? event_type::kArr : event_type::kDep)) {
          continue;
        }
        auto const key = is_arr ? node_pair{node_of(rs), pinned_at(ly)}
                                : node_pair{pinned_at(lx), node_of(rs)};
        auto& all = vs.transfers_[key];
        for (auto t = route_transport_idx_t{0U};
             t != tt.route_transport_ranges_[rs.route_].size(); ++t) {
          auto const none = route_transport_idx_t{0U};
          auto const e = is_arr ? get_expected(signature(rs, t), {})
                                : get_expected({}, signature(rs, t));
          for (auto const gap : {e, e - 1}) {
            all.push_back(virt_transfer{is_arr ? t : none, is_arr ? none : t,
                                        static_cast<std::int16_t>(gap),
                                        static_cast<std::int16_t>(e)});
          }
        }
        note(key);
      }
    }
    auto const lo = i32_minutes{window.lo_};
    auto const hi = i32_minutes{window.hi_};
    for (auto const x : get_route_stops(lx)) {
      if (!has_event(tt, x, event_type::kArr)) {
        continue;
      }
      for (auto const y : get_route_stops(ly)) {
        auto const nx = node_of(x);
        auto const ny = node_of(y);
        if (!has_event(tt, y, event_type::kDep) ||
            (is_pinned(nx) && is_pinned(ny))) {
          continue;
        }
        auto const& departures = get_events(y, event_type::kDep);
        transfers.clear();
        for (auto const& [arr, a] : get_events(x, event_type::kArr)) {
          auto const& a_days = traffic_days(x.route_, a);
          for (auto offset = std::chrono::floor<date::days>(
                   arr + lo - departures.back().first);
               offset <= std::chrono::floor<date::days>(
                             arr + hi - departures.front().first);
               ++offset) {
            auto const arrival = arr - offset;
            auto const n = offset.count();
            for (auto it = std::lower_bound(
                     begin(departures), end(departures),
                     std::pair{arrival + lo, route_transport_idx_t{0U}});
                 it != end(departures) && it->first < arrival + hi; ++it) {
              auto const b = it->second;
              if (x.route_ == y.route_ && a == b) {
                continue;
              }
              auto const days = n >= 0 ? a_days << static_cast<std::size_t>(n)
                                       : a_days >> static_cast<std::size_t>(-n);
              if ((days & traffic_days(y.route_, b)).none()) {
                continue;
              }
              auto const e = get_expected(signature(x, a), signature(y, b));
              transfers.push_back(virt_transfer{
                  a, b,
                  static_cast<std::int16_t>((it->first - arrival).count()),
                  static_cast<std::int16_t>(e)});
            }
          }
        }
        if (transfers.empty()) {
          continue;
        }
        auto const key = node_pair{nx, ny};
        auto& all = vs.transfers_[key];
        all.insert(end(all), begin(transfers), end(transfers));
        note(key);
      }
    }
  }
  for (auto& [_, keys] : vs.route_keys_) {
    utl::erase_duplicates(keys);
  }
}

part_idx_t get_part(virt_state const& vs,
                    virt_node const& n,
                    route_transport_idx_t const t) {
  return is_pinned(n) ? part_idx_t{0U} : vs.parts_.at(n.rs_.route_)[t];
}

place_ref get_place(virt_state const& vs, unit_idx_t const u) {
  return place_ref{vs.units_[u].stop_, vs.units_[u].place_};
}

unit_idx_t get_unit(timetable const& tt,
                    virt_state& vs,
                    virt_node const& n,
                    part_idx_t const p,
                    place_idx_t const place = place_idx_t{0U}) {
  return utl::get_or_create(vs.unit_of_, pair{n, p}, [&]() {
    auto const s = is_pinned(n)
                       ? n.pinned_
                       : stop{tt.route_location_seq_[n.rs_.route_][n.rs_.idx_]}
                             .location_idx();
    auto const u = unit_idx_t{vs.units_.size()};
    vs.units_.push_back(virt_unit{.node_ = n,
                                  .part_ = p,
                                  .stop_ = s,
                                  .place_ = place,
                                  .is_movable_ = !is_pinned(n)});
    vs.out_.emplace_back();
    vs.in_.emplace_back();
    auto& places = vs.places_[s];
    if (places.size() <= to_idx(place)) {
      places.resize(to_idx(place) + 1U);
    }
    places[place].push_back(u);
    return u;
  });
}

void aggregate(timetable const& tt, virt_state& vs, node_pair const& key) {
  auto& kp = vs.key_units_[key];
  for (auto const& p : kp) {
    vs.bounds_[p] = vs.windows_.at(key);
    vs.expected_[p].clear();
  }
  auto const& [x, y] = key;
  auto agg = hash_map<pair<part_idx_t, part_idx_t>, duration_bounds>{};
  auto expected = hash_map<pair<part_idx_t, part_idx_t>,
                           hash_map<std::int32_t, std::uint32_t>>{};
  for (auto const& t : vs.transfers_.at(key)) {
    auto const parts = pair{get_part(vs, x, t.from_), get_part(vs, y, t.to_)};
    agg.emplace(parts, vs.windows_.at(key))
        .first->second.add(t.gap_, t.gap_ >= t.expected_);
    ++expected[parts][t.expected_];
  }
  for (auto const& [parts, b] : agg) {
    auto const ux = get_unit(tt, vs, x, parts.first);
    auto const uy = get_unit(tt, vs, y, parts.second);
    auto const p = pair{ux, uy};
    vs.expected_[p] = std::move(expected[parts]);
    if (auto const [it, is_new] = vs.bounds_.emplace(p, b); is_new) {
      vs.out_[ux].push_back(uy);
      vs.in_[uy].push_back(ux);
      kp.push_back(p);
    } else {
      it->second = b;
    }
  }
}

std::int32_t get_duration(timetable const& tt,
                          virt_state& vs,
                          place_ref const p,
                          place_ref const q) {
  auto const it = vs.durations_.find(pair{p, q});
  return it == end(vs.durations_) ? get_fallback(tt, vs, p.first, q.first)
                                  : it->second;
}

void set_duration(timetable const& tt,
                  virt_state& vs,
                  place_ref const p,
                  place_ref const q,
                  std::int32_t const d) {
  if (d == get_fallback(tt, vs, p.first, q.first)) {
    vs.durations_.erase(pair{p, q});
  } else {
    vs.durations_[pair{p, q}] = d;
  }
}

bool is_satisfied(timetable const& tt,
                  virt_state& vs,
                  unit_idx_t const u,
                  unit_idx_t const v) {
  return vs.bounds_.at(pair{u, v})
      .contains(get_duration(tt, vs, get_place(vs, u), get_place(vs, v)));
}

bool is_any(unit_idx_t, unit_idx_t) { return true; }

template <typename Fn>
std::pair<duration_bounds, hash_map<std::int32_t, std::uint32_t>> get_bounds(
    virt_state const& vs, place_ref const p, place_ref const q, Fn&& is_kept) {
  auto b = duration_bounds{};
  auto expected = hash_map<std::int32_t, std::uint32_t>{};
  for (auto const u : vs.places_.at(p.first)[p.second]) {
    for (auto const v : vs.out_[u]) {
      if (get_place(vs, v) == q && is_kept(u, v)) {
        b.intersect(vs.bounds_.at(pair{u, v}));
        for (auto const& [e, n] : vs.expected_.at(pair{u, v})) {
          expected[e] += n;
        }
      }
    }
  }
  return {b, std::move(expected)};
}

std::optional<std::int32_t> pick(
    timetable const& tt,
    virt_state& vs,
    place_ref const p,
    place_ref const q,
    std::pair<duration_bounds, hash_map<std::int32_t, std::uint32_t>> const&
        b) {
  if (p.second == place_idx_t{0U} && q.second == place_idx_t{0U}) {
    return std::nullopt;
  }
  return pick_duration(b.first, get_fallback(tt, vs, p.first, q.first),
                       p == q ? kMaxTransferTime : kMaxFootpath, b.second);
}

struct unit_move {
  unit_idx_t u_;
  place_idx_t to_;  // invalid: a new place
};

bool try_moves(timetable const& tt,
               virt_state& vs,
               std::vector<unit_move> const& moves,
               pair<unit_idx_t, unit_idx_t> const target) {
  auto const is_moved = [&](unit_idx_t const u) {
    return utl::any_of(moves, [&](unit_move const& m) { return m.u_ == u; });
  };
  auto held = hash_set<pair<unit_idx_t, unit_idx_t>>{};
  for (auto const& m : moves) {
    for (auto const v : vs.out_[m.u_]) {
      if (is_satisfied(tt, vs, m.u_, v)) {
        held.emplace(m.u_, v);
      }
    }
    for (auto const v : vs.in_[m.u_]) {
      if (is_satisfied(tt, vs, v, m.u_)) {
        held.emplace(v, m.u_);
      }
    }
  }

  auto const relocate = [&](unit_idx_t const u, place_idx_t const to) {
    auto& places = vs.places_.at(vs.units_[u].stop_);
    std::erase(places[vs.units_[u].place_], u);
    places[to].push_back(u);
    vs.units_[u].place_ = to;
  };
  auto const old = utl::to_vec(
      moves, [&](unit_move const& m) { return vs.units_[m.u_].place_; });
  auto created = std::vector<location_idx_t>{};
  for (auto const& m : moves) {
    auto& places = vs.places_.at(vs.units_[m.u_].stop_);
    if (m.to_ == place_idx_t::invalid()) {
      places.emplace_back();
      created.push_back(vs.units_[m.u_].stop_);
    }
    relocate(m.u_, m.to_ == place_idx_t::invalid()
                       ? place_idx_t{places.size() - 1U}
                       : m.to_);
  }

  auto touched = std::vector<pair<place_ref, place_ref>>{};
  for (auto const& m : moves) {
    for (auto const v : vs.out_[m.u_]) {
      touched.push_back(pair{get_place(vs, m.u_), get_place(vs, v)});
    }
    for (auto const v : vs.in_[m.u_]) {
      touched.push_back(pair{get_place(vs, v), get_place(vs, m.u_)});
    }
  }
  utl::erase_duplicates(touched);
  auto plan =
      std::vector<std::pair<pair<place_ref, place_ref>, std::int32_t>>{};
  auto is_possible = true;
  for (auto const& [p, q] : touched) {
    auto const d = get_duration(tt, vs, p, q);
    auto const b =
        get_bounds(vs, p, q, [&](unit_idx_t const a, unit_idx_t const c) {
          return pair{a, c} == target ||
                 (is_moved(a) || is_moved(c)
                      ? held.contains(pair{a, c})
                      : vs.bounds_.at(pair{a, c}).contains(d));
        });
    if (b.first.contains(d)) {
      continue;
    }
    auto const x = pick(tt, vs, p, q, b);
    if (!x.has_value()) {
      is_possible = false;
      break;
    }
    plan.emplace_back(pair{p, q}, *x);
  }

  if (!is_possible) {
    for (auto i = moves.size(); i-- != 0U;) {
      relocate(moves[i].u_, old[i]);
    }
    for (auto const s : created) {
      vs.places_.at(s).resize(vs.places_.at(s).size() - 1U);
    }
    return false;
  }
  for (auto const& [pq, d] : plan) {
    set_duration(tt, vs, pq.first, pq.second, d);
  }
  return true;
}

std::vector<std::vector<route_transport_idx_t>> get_split(timetable const& tt,
                                                          virt_state& vs,
                                                          unit_idx_t const u,
                                                          unit_idx_t const v,
                                                          bool const is_from) {
  auto const& x = vs.units_[u];
  auto const& y = vs.units_[v];
  auto const& z = is_from ? x : y;
  if (!z.is_movable_) {
    return {};
  }
  auto const key = node_pair{x.node_, y.node_};
  auto b = hash_map<route_transport_idx_t, duration_bounds>{};
  for (auto const& t : vs.transfers_.at(key)) {
    if (get_part(vs, x.node_, t.from_) == x.part_ &&
        get_part(vs, y.node_, t.to_) == y.part_) {
      b.emplace(is_from ? t.from_ : t.to_, vs.windows_.at(key))
          .first->second.add(t.gap_, t.gap_ >= t.expected_);
    }
  }
  auto const fallback = get_fallback(tt, vs, x.stop_, y.stop_);
  auto const max = u == v ? kMaxTransferTime : kMaxFootpath;
  auto const single = [&](route_transport_idx_t const t) {
    return std::ranges::count(vs.parts_.at(z.node_.rs_.route_), z.part_) > 1
               ? std::vector<std::vector<route_transport_idx_t>>{{t}}
               : std::vector<std::vector<route_transport_idx_t>>{};
  };
  auto items = utl::to_vec(b);
  utl::sort(items, [](auto const& a, auto const& c) {
    return std::pair{a.second.hi_, a.first} < std::pair{c.second.hi_, c.first};
  });
  auto groups = std::vector<
      std::pair<duration_bounds, std::vector<route_transport_idx_t>>>{};
  for (auto const& [t, tb] : items) {
    if (!pick_duration(tb, fallback, max).has_value()) {
      return single(t);
    }
    auto i = groups.empty() ? duration_bounds{} : groups.back().first;
    i.intersect(tb);
    if (!groups.empty() && pick_duration(i, fallback, max).has_value()) {
      groups.back().first = i;
      groups.back().second.push_back(t);
    } else {
      groups.push_back({tb, {t}});
    }
  }
  if (groups.size() < 2U) {
    return single(items.front().first);
  }
  auto moved = utl::to_vec(groups, [](auto& g) { return std::move(g.second); });
  moved.erase(std::ranges::max_element(moved, {},
                                       [](auto const& g) { return g.size(); }));
  return moved;
}

void split(timetable const& tt,
           virt_state& vs,
           unit_idx_t const u,
           std::vector<std::vector<route_transport_idx_t>> const& moved,
           std::vector<pair<unit_idx_t, unit_idx_t>>& work) {
  auto const r = vs.units_[u].node_.rs_.route_;
  auto const p = vs.units_[u].part_;
  auto& np = vs.n_parts_.at(r);
  auto const first = np;
  for (auto const& g : moved) {
    for (auto const t : g) {
      vs.parts_.at(r)[t] = np;
    }
    ++np;
  }
  for (auto const& key : vs.route_keys_.at(r)) {
    for (auto const& m : {key.first, key.second}) {
      if (auto const it = vs.unit_of_.find(pair{m, p});
          !is_pinned(m) && m.rs_.route_ == r && it != end(vs.unit_of_)) {
        auto const place = vs.units_[it->second].place_;
        for (auto q = first; q != np; ++q) {
          get_unit(tt, vs, m, q, place);
        }
      }
    }
  }
  for (auto const& key : vs.route_keys_.at(r)) {
    aggregate(tt, vs, key);
    for (auto const& [a, b] : vs.key_units_.at(key)) {
      if (!is_satisfied(tt, vs, a, b)) {
        work.push_back(pair{a, b});
      }
    }
  }
}

void resolve(timetable const& tt, virt_state& vs) {
  for (auto const& [r, _] : vs.route_keys_) {
    vs.parts_.emplace(
        r, vector_map<route_transport_idx_t, part_idx_t>(
               tt.route_transport_ranges_[r].size(), part_idx_t{0U}));
    vs.n_parts_.emplace(r, part_idx_t{1U});
  }
  for (auto const& [key, _] : vs.transfers_) {
    aggregate(tt, vs, key);
  }
  auto work = std::vector<pair<unit_idx_t, unit_idx_t>>{};
  for (auto u = unit_idx_t{0U}; u != vs.units_.size(); ++u) {
    for (auto const v : vs.out_[u]) {
      if (!is_satisfied(tt, vs, u, v)) {
        work.push_back(pair{u, v});
      }
    }
  }
  auto const n_initial = work.size();
  auto n_adjusted = 0U;
  auto n_moved = 0U;
  auto n_split = 0U;
  auto n_unresolved = 0U;
  auto const fresh = place_idx_t::invalid();
  while (!work.empty()) {
    auto const [u, v] = work.back();
    work.pop_back();
    if (is_satisfied(tt, vs, u, v)) {
      continue;
    }
    auto const p = get_place(vs, u);
    auto const q = get_place(vs, v);
    if (auto const d = pick(tt, vs, p, q, get_bounds(vs, p, q, is_any))) {
      set_duration(tt, vs, p, q, *d);
      ++n_adjusted;
      continue;
    }
    auto const target = pair{u, v};
    auto const get_candidates = [&](unit_idx_t const m) {
      auto const at = get_place(vs, m);
      auto const other = get_place(vs, m == u ? v : u);
      auto const& places = vs.places_.at(at.first);
      auto c = std::vector<place_idx_t>{};
      for (auto x = place_idx_t{0U}; x != places.size(); ++x) {
        if (x != at.second && (x == place_idx_t{0U} || !places[x].empty())) {
          c.push_back(x);
        }
      }
      std::ranges::stable_sort(c, {}, [&](place_idx_t const x) {
        auto const xp = place_ref{at.first, x};
        return !(vs.durations_.contains(pair{xp, other}) ||
                 vs.durations_.contains(pair{other, xp}));
      });
      c.push_back(fresh);
      return c;
    };
    auto tries = std::vector<std::vector<unit_move>>{};
    for (auto const m : {u, v}) {
      if (vs.units_[m].is_movable_ && (m == u || u != v)) {
        for (auto const c : get_candidates(m)) {
          tries.push_back({{m, c}});
        }
      }
    }
    if (u != v && vs.units_[u].is_movable_ && vs.units_[v].is_movable_) {
      for (auto const c : get_candidates(u)) {
        tries.push_back({{u, c}, {v, fresh}});
      }
      for (auto const c : get_candidates(v)) {
        if (c != fresh) {
          tries.push_back({{u, fresh}, {v, c}});
        }
      }
    }
    if (utl::any_of(tries, [&](std::vector<unit_move> const& moves) {
          return try_moves(tt, vs, moves, target);
        })) {
      ++n_moved;
      continue;
    }
    auto const from = get_split(tt, vs, u, v, true);
    auto const to = get_split(tt, vs, u, v, false);
    auto const use_from =
        !from.empty() && (to.empty() || from.size() <= to.size());
    if (from.empty() && to.empty()) {
      ++n_unresolved;
      continue;
    }
    split(tt, vs, use_from ? u : v, use_from ? from : to, work);
    work.push_back(target);
    ++n_split;
  }

  auto n_merged = 0U;
  auto const no_target = pair{unit_idx_t::invalid(), unit_idx_t::invalid()};
  for (auto& [s, places] : vs.places_) {
    for (auto p = place_idx_t{places.size()}; p-- > place_idx_t{1U};) {
      for (auto q = place_idx_t{0U}; !places[p].empty() && q != places.size();
           ++q) {
        if (q != p && (q == place_idx_t{0U} || !places[q].empty()) &&
            try_moves(tt, vs,
                      utl::to_vec(
                          places[p],
                          [&](unit_idx_t const m) { return unit_move{m, q}; }),
                      no_target)) {
          ++n_merged;
        }
      }
    }
  }
  log(log_lvl::info, "loader.virtual_locations",
      "{} conflicts: {} adjusted, {} moved, {} splits, {} unresolved, {} "
      "places merged",
      n_initial, n_adjusted, n_moved, n_split, n_unresolved, n_merged);
}

void split_routes(timetable& tt,
                  shapes_storage* shapes,
                  vector_map<route_idx_t, route_idx_t> const& origin,
                  vector_map<route_idx_t, std::vector<transport_idx_t>> const&
                      route_transports) {
  auto const n_routes = tt.route_transport_ranges_.size();
  auto const old_ranges = tt.route_transport_ranges_;
  auto const old_time_ranges = tt.route_stop_time_ranges_;
  auto const old_times = std::move(tt.route_stop_times_);

  auto order = vector_map<transport_idx_t, transport_idx_t>{};
  tt.route_transport_ranges_.clear();
  tt.route_stop_time_ranges_.clear();
  tt.route_stop_times_ = {};
  for (auto r = route_idx_t{0U}; r != origin.size(); ++r) {
    auto const o = origin[r];
    auto const& transports = route_transports[r];
    tt.route_transport_ranges_.emplace_back(
        transport_idx_t{order.size()},
        transport_idx_t{order.size() + transports.size()});
    order.insert(end(order), begin(transports), end(transports));

    auto const get_time = [&](transport_idx_t const t, stop_idx_t const i,
                              event_type const ev) {
      return old_times[old_time_ranges[o].from_ +
                       old_ranges[o].size() *
                           (i * 2U - (ev == event_type::kArr ? 1U : 0U)) +
                       to_idx(t - old_ranges[o].from_)];
    };
    auto const begin_times = tt.route_stop_times_.size();
    auto const n_stops =
        static_cast<stop_idx_t>(tt.route_location_seq_[r].size());
    for (auto i = stop_idx_t{1U}; i != n_stops; ++i) {
      for (auto const t : transports) {
        tt.route_stop_times_.push_back(
            get_time(t, static_cast<stop_idx_t>(i - 1U), event_type::kDep));
      }
      for (auto const t : transports) {
        tt.route_stop_times_.push_back(get_time(t, i, event_type::kArr));
      }
    }
    tt.route_stop_time_ranges_.emplace_back(
        static_cast<std::uint32_t>(begin_times),
        static_cast<std::uint32_t>(tt.route_stop_times_.size()));

    if (r < n_routes) {
      continue;
    }
    auto const clasz = tt.route_clasz_[o];
    tt.route_clasz_.push_back(clasz);
    auto const section_clasz = utl::to_vec(tt.route_section_clasz_[o]);
    tt.route_section_clasz_.emplace_back(section_clasz);
    for (auto f = 0U; f != kNumRouteFlags; ++f) {
      auto& flags = tt.route_flags_[f];
      flags.resize(flags.size() + 2U);
      flags.set(to_idx(r) * 2U, flags.test(to_idx(o) * 2U));
      flags.set(to_idx(r) * 2U + 1U, flags.test(to_idx(o) * 2U + 1U));
      auto const per_section = utl::to_vec(tt.route_flags_per_section_[f][o]);
      tt.route_flags_per_section_[f].emplace_back(per_section);
    }
    if (shapes != nullptr && shapes->route_bboxes_.size() == to_idx(r)) {
      auto const bbox = shapes->route_bboxes_[o];
      auto const segments = utl::to_vec(shapes->route_segment_bboxes_[o]);
      shapes->route_bboxes_.push_back(bbox);
      shapes->route_segment_bboxes_.emplace_back(segments);
    }
  }

  auto inverse = vector_map<transport_idx_t, transport_idx_t>{};
  inverse.resize(order.size());
  for (auto t = transport_idx_t{0U}; t != order.size(); ++t) {
    inverse[order[t]] = t;
  }
  auto const permute = [&](auto& v) {
    auto copy = v;
    for (auto t = transport_idx_t{0U}; t != order.size(); ++t) {
      copy[t] = v[order[t]];
    }
    v = std::move(copy);
  };
  auto const permute_buckets = [&](auto& v) {
    auto copy = std::remove_reference_t<decltype(v)>{};
    for (auto const t : order) {
      copy.emplace_back(v[t]);
    }
    v = std::move(copy);
  };
  permute(tt.transport_first_dep_offset_);
  permute(tt.transport_traffic_days_);
  permute_buckets(tt.transport_to_trip_section_);
  permute_buckets(tt.transport_section_attributes_);
  permute_buckets(tt.transport_section_providers_);
  permute_buckets(tt.transport_section_directions_);
  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    for (auto const t : tt.route_transport_ranges_[r]) {
      tt.transport_route_[t] = r;
    }
  }
  for (auto trip = trip_idx_t{0U}; trip != tt.trip_transport_ranges_.size();
       ++trip) {
    for (auto& range : tt.trip_transport_ranges_[trip]) {
      range.first = inverse[range.first];
    }
  }

  auto location_routes = mutable_fws_multimap<location_idx_t, route_idx_t>{};
  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    for (auto const s : tt.route_location_seq_[r]) {
      auto routes = location_routes[stop{s}.location_idx()];
      if (routes.empty() || routes.back() != r) {
        routes.push_back(r);
      }
    }
  }
  tt.location_routes_.clear();
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    tt.location_routes_.emplace_back(location_routes[l]);
  }
}

bool write(timetable& tt,
           virt_state& vs,
           shapes_storage* shapes,
           mutable_fws_multimap<location_idx_t, footpath>& durations) {
  auto virts = hash_map<place_ref, location_idx_t>{};
  for (auto const& [s, places] : vs.places_) {
    for (auto p = place_idx_t{1U}; p < places.size(); ++p) {
      if (places[p].empty()) {
        continue;
      }
      auto const self = place_ref{s, p};
      auto const d = get_duration(tt, vs, self, self);
      auto l = location{};
      l.src_ = tt.locations_.src_[s];
      l.pos_ = tt.locations_.coordinates_[s];
      l.type_ = location_type::kVirt;
      l.parent_ = s;
      l.transfer_time_ = d >= kNever ? footpath::kMaxDuration : duration_t{d};
      auto const v = register_location(tt, l);
      tt.locations_.children_[s].push_back(v);
      virts.emplace(self, v);
    }
  }
  if (virts.empty()) {
    return false;
  }
  auto const location_of =
      [&](place_ref const p) -> std::optional<location_idx_t> {
    if (p.second == place_idx_t{0U}) {
      return p.first;
    }
    auto const it = virts.find(p);
    return it == end(virts) ? std::nullopt : std::optional{it->second};
  };

  durations[location_idx_t{tt.n_locations() - 1U}];
  auto n_footpaths = std::size_t{0U};
  for (auto const& [pq, d] : vs.durations_) {
    auto const from = location_of(pq.first);
    auto const to = location_of(pq.second);
    if (pq.first == pq.second || !from.has_value() || !to.has_value()) {
      continue;
    }
    durations[*from].push_back(
        footpath{*to, d >= kNever ? footpath::kMaxDuration : duration_t{d}});
    ++n_footpaths;
  }

  auto origin = vector_map<route_idx_t, route_idx_t>{};
  auto route_transports =
      vector_map<route_idx_t, std::vector<transport_idx_t>>{};
  auto const n_routes = tt.n_routes();
  for (auto r = route_idx_t{0U}; r != n_routes; ++r) {
    auto const transports = tt.route_transport_ranges_[r];
    origin.push_back(r);
    route_transports.emplace_back(begin(transports), end(transports));
  }
  auto seqs = vector_map<part_idx_t, stop_seq_t>{};
  for (auto r = route_idx_t{0U}; r != n_routes; ++r) {
    auto const it = vs.parts_.find(r);
    if (it == end(vs.parts_)) {
      continue;
    }
    auto const& parts = it->second;
    auto const transports = tt.route_transport_ranges_[r];
    auto const seq = tt.route_location_seq_[r];
    seqs.clear();
    seqs.resize(to_idx(vs.n_parts_.at(r)), stop_seq_t{seq.begin(), seq.end()});
    for (auto p = part_idx_t{0U}; p != seqs.size(); ++p) {
      for (auto i = stop_idx_t{0U}; i != seq.size(); ++i) {
        auto const u = vs.unit_of_.find(
            pair{virt_node{route_stop{r, i}, location_idx_t::invalid()}, p});
        if (u != end(vs.unit_of_)) {
          seqs[p][i] =
              stop{seq[i]}
                  .with_location(*location_of(get_place(vs, u->second)))
                  .value();
        }
      }
    }
    auto groups =
        std::vector<std::pair<stop_seq_t, std::vector<transport_idx_t>>>{};
    for (auto t = route_transport_idx_t{0U}; t != transports.size(); ++t) {
      auto const& s = seqs[parts[t]];
      auto g = utl::find_if(
          groups, [&](auto const& group) { return group.first == s; });
      if (g == end(groups)) {
        g = groups.insert(end(groups), {s, {}});
      }
      g->second.push_back(transports.from_ + to_idx(t));
    }
    route_transports[r] = std::move(groups.front().second);
    std::ranges::copy(groups.front().first, tt.route_location_seq_[r].begin());
    for (auto& [group_seq, group_transports] : std::span{groups}.subspan(1U)) {
      origin.push_back(r);
      route_transports.push_back(std::move(group_transports));
      tt.route_location_seq_.emplace_back(group_seq);
    }
  }
  split_routes(tt, shapes, origin, route_transports);

  log(log_lvl::info, "loader.virtual_locations",
      "{} virtual locations, {} routes -> {}, {} footpaths", virts.size(),
      n_routes, tt.n_routes(), n_footpaths);
  return true;
}

bool add_virtual_locations(
    timetable& tt,
    shapes_storage* shapes,
    mutable_fws_multimap<location_idx_t, footpath>& durations) {
  if (tt.transfer_rules_.rules_.empty()) {
    return false;
  }
  auto const timer = scoped_timer{"loader.virtual_locations"};
  auto vs = virt_state{};
  add_signatures(tt, vs);
  add_transfers(tt, vs);
  resolve(tt, vs);
  return write(tt, vs, shapes, durations);
}

}  // namespace nigiri::loader
