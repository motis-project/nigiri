#include "nigiri/loader/virtual_locations.h"

#include <cstdint>
#include <algorithm>
#include <chrono>
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
#include "nigiri/common/interval.h"
#include "nigiri/footpath.h"
#include "nigiri/for_each_meta.h"
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
using key_idx_t = cista::strong<std::uint32_t, struct key_>;
using signature_id_t = cista::strong<std::uint32_t, struct signature_id_>;
using sides_t = std::vector<transfer_rule_side_idx>;
using stop_seq_t = std::vector<stop::value_type>;
using unit_pair = pair<unit_idx_t, unit_idx_t>;
using signatures_t = vector_map<route_transport_idx_t, signature_id_t>;
// How often each expected time occurs among the transfers of a pair.
using expected_t = std::vector<pair<i32_minutes, std::uint32_t>>;
// The durations a footpath (or transfer time) may have: [from, to).
using bounds_t = interval<i32_minutes>;

constexpr auto const kNever = i32_minutes{routing::kMaxTravelTime};
constexpr auto const kAnyDuration =
    bounds_t{i32_minutes{std::numeric_limits<std::int32_t>::min()},
             i32_minutes{std::numeric_limits<std::int32_t>::max()}};
constexpr auto const kMaxFootpath =
    i32_minutes{footpath::kMaxDuration} - i32_minutes{1};
constexpr auto const kMaxTransferTime =
    i32_minutes{kNoTransferAllowed} - i32_minutes{1};

struct route_stop {
  CISTA_COMPARABLE()
  route_idx_t route_;
  stop_idx_t idx_;
};

struct virt_transfer {
  route_transport_idx_t from_, to_;
  duration_t gap_;
  duration_t expected_;
};

// A route stop, or the stop itself (base: everything without a signature).
struct virt_node {
  CISTA_COMPARABLE()
  route_stop rs_;
  location_idx_t base_;
};

using node_pair = pair<virt_node, virt_node>;

// A route part at a route stop, or the stop itself.
struct virt_unit {
  virt_node node_;
  part_idx_t part_;
  location_idx_t stop_;
};

void count(expected_t& h, i32_minutes const e, std::uint32_t const n = 1U) {
  auto const it = utl::find_if(h, [&](auto const& x) { return x.first == e; });
  if (it == end(h)) {
    h.push_back(pair<i32_minutes, std::uint32_t>{e, n});
  } else {
    it->second += n;
  }
}

// A meeting that must work allows at most its gap, one that must not work
// needs more than its gap.
void narrow(bounds_t& b, i32_minutes const gap, bool const must_work) {
  if (must_work) {
    b.to_ = std::min(b.to_, gap + i32_minutes{1});
  } else {
    b.from_ = std::max(b.from_, gap + i32_minutes{1});
  }
}

// The smallest range covering both.
void extend(bounds_t& a, bounds_t const& b) {
  a.from_ = std::min(a.from_, b.from_);
  a.to_ = std::max(a.to_, b.to_);
}

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

i32_minutes get_minutes(transfer_rule const& r) {
  return r.duration_ == footpath::kMaxDuration ? kNever
                                               : i32_minutes{r.duration_};
}

// The range of the rule times per stop pair.
hash_map<transfer_pair, bounds_t> get_rule_times(timetable const& tt) {
  auto rule_times = hash_map<transfer_pair, bounds_t>{};
  for (auto const& rule : tt.transfer_rules_.rules_) {
    auto const m = get_minutes(rule);
    auto const r = bounds_t{m, m + i32_minutes{1}};
    extend(rule_times.emplace(transfer_pair{rule.from_stop_, rule.to_stop_}, r)
               .first->second,
           r);
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

// The duration for the bounds: the fallback if it fits, a ban if nothing has
// to work, else the most frequent expected time within them, else the largest
// storable one. None if no storable duration fits.
std::optional<i32_minutes> pick_duration(bounds_t const& b,
                                         i32_minutes const fallback,
                                         i32_minutes const max,
                                         expected_t const& expected = {}) {
  if (b.contains(fallback)) {
    return fallback;
  }
  if (b.contains(kNever)) {
    return kNever;
  }
  auto const to = std::min(b.to_, max + i32_minutes{1});
  if (b.from_ >= to) {
    return std::nullopt;
  }
  auto best = to - i32_minutes{1};
  auto best_n = std::uint32_t{0U};
  for (auto const& [e, n] : expected) {
    if (b.from_ <= e && e < to && (n > best_n || (n == best_n && e < best))) {
      best = e;
      best_n = n;
    }
  }
  return best;
}

bool is_admissible(bounds_t const& b, bool const is_self) {
  return pick_duration(b, kNever, is_self ? kMaxTransferTime : kMaxFootpath)
      .has_value();
}

struct virt_state {
  std::vector<sides_t> signatures_;
  hash_map<route_stop, signatures_t> signature_of_;
  hash_map<pair<location_idx_t, location_idx_t>, i32_minutes> fallbacks_;
  // Keys: the node pairs with transfers, their transfers and the durations
  // their stop pair allows at all.
  hash_map<node_pair, key_idx_t> key_of_;
  vector_map<key_idx_t, node_pair> keys_;
  vector_map<key_idx_t, std::vector<virt_transfer>> transfers_;
  vector_map<key_idx_t, bounds_t> windows_;
  vector_map<key_idx_t, std::vector<unit_pair>> key_units_;
  vector_map<route_idx_t, std::vector<key_idx_t>> route_keys_;
  vector_map<route_idx_t, vector_map<route_transport_idx_t, part_idx_t>> parts_;
  vector_map<route_idx_t, part_idx_t> n_parts_;
  vector_map<unit_idx_t, virt_unit> units_;
  hash_map<pair<virt_node, part_idx_t>, unit_idx_t> unit_of_;
  vector_map<unit_idx_t, std::vector<unit_idx_t>> out_, in_;
  // The bounds between two units, after merging between two places (keyed by
  // the units standing for the places).
  hash_map<unit_pair, bounds_t> bounds_;
  hash_map<unit_pair, expected_t> expected_;
  // The unit standing for the place of the unit: itself for a place.
  vector_map<unit_idx_t, unit_idx_t> place_;
};

bool is_base(virt_node const& n) {
  return n.rs_.route_ == route_idx_t::invalid();
}

bool is_base(virt_state const& vs, unit_idx_t const u) {
  return is_base(vs.units_[u].node_);
}

virt_node base_node(location_idx_t const l) {
  return virt_node{route_stop{route_idx_t::invalid(), stop_idx_t{0U}}, l};
}

i32_minutes get_fallback(timetable const& tt,
                         virt_state& vs,
                         location_idx_t const from,
                         location_idx_t const to) {
  return utl::get_or_create(vs.fallbacks_, pair{from, to}, [&]() {
    auto const walk =
        routing::shortest_transfer(tt, nullptr, kDefaultProfile, from, to);
    return walk.has_value() ? i32_minutes{*walk} : kNever;
  });
}

// The key of the node pair, registered with its routes on first use.
key_idx_t get_key(virt_state& vs,
                  node_pair const& key,
                  bounds_t const& window) {
  return utl::get_or_create(vs.key_of_, key, [&]() {
    auto const k = key_idx_t{vs.keys_.size()};
    vs.keys_.push_back(key);
    vs.transfers_.emplace_back();
    vs.windows_.push_back(window);
    vs.key_units_.emplace_back();
    auto const& [x, y] = key;
    if (!is_base(x)) {
      vs.route_keys_[x.rs_.route_].push_back(k);
    }
    if (!is_base(y) && (is_base(x) || x.rs_.route_ != y.rs_.route_)) {
      vs.route_keys_[y.rs_.route_].push_back(k);
    }
    return k;
  });
}

void add_signatures(timetable const& tt, virt_state& vs) {
  auto const& tr = tt.transfer_rules_;
  auto const global_rules = get_global_rules(tt);
  auto ids = hash_map<sides_t, signature_id_t>{};
  vs.signatures_.emplace_back();
  ids.emplace(sides_t{}, signature_id_t{0U});
  auto sig = sides_t{};
  auto section_sides = std::vector<sides_t>{};
  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    auto const transports = tt.route_transport_ranges_[r];
    auto const seq = tt.route_location_seq_[r];
    for (auto const t : transports) {
      // The sides of the transport's sections once, filtered per stop.
      auto const sections = tt.transport_to_trip_section_[t];
      section_sides.resize(sections.size());
      auto has_sides = false;
      for (auto k = 0U; k != sections.size(); ++k) {
        get_section_sides(tt, sections[k], section_sides[k]);
        has_sides |= !section_sides[k].empty();
      }
      if (!has_sides) {
        continue;
      }
      auto const section = [&](stop_idx_t const k) -> sides_t const& {
        return section_sides[sections.size() == 1U ? 0U : k];
      };
      for (auto i = stop_idx_t{0U}; i != seq.size(); ++i) {
        auto const stp = stop{seq[i]};
        auto const base = stp.location_idx();
        sig.clear();
        if (i != 0U && stp.out_allowed()) {
          add_stop_sides(tt, section(static_cast<stop_idx_t>(i - 1U)),
                         event_type::kArr, base, sig);
        }
        if (i + 1U != seq.size() && stp.in_allowed()) {
          add_stop_sides(tt, section(i), event_type::kDep, base, sig);
        }
        utl::erase_duplicates(sig);
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
              return signatures_t(transports.size(), signature_id_t{0U});
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
  auto pairs = hash_set<pair<location_idx_t, location_idx_t>>{};
  // The pairs of the stops' locations (a station: itself and its children)
  // with a rule stop on one side; all of them inside a station of global rules.
  auto const add = [&](location_idx_t const from, location_idx_t const to,
                       bool const is_global) {
    routing::for_each_meta(
        tt, routing::location_match_mode::kOnlyChildren, from,
        [&](location_idx_t const f) {
          routing::for_each_meta(
              tt, routing::location_match_mode::kOnlyChildren, to,
              [&](location_idx_t const t) {
                if (is_global || rule_stops.contains(f) ||
                    rule_stops.contains(t)) {
                  pairs.insert(pair{f, t});
                }
              });
        });
  };
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
    add(r.from_stop_, r.to_stop_, false);
  }
  if (has_global) {
    for (auto const l : rule_stops) {
      auto const root = locs.get_root_idx(l);
      if (!tt.transfer_rules_.is_global_excluded(root)) {
        add(root, root, true);
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
  vs.route_keys_.resize(tt.n_routes());
  auto const location_of = [&](route_stop const rs) {
    return stop{tt.route_location_seq_[rs.route_][rs.idx_]}.location_idx();
  };
  auto const sigs_of = [&](route_stop const rs) -> signatures_t const* {
    auto const it = vs.signature_of_.find(rs);
    return it == end(vs.signature_of_) ? nullptr : &it->second;
  };
  auto const node_of = [&](route_stop const rs, signatures_t const* sigs) {
    return sigs != nullptr ? virt_node{rs, location_idx_t::invalid()}
                           : base_node(location_of(rs));
  };
  auto const sig = [](signatures_t const* sigs,
                      route_transport_idx_t const u) {
    return sigs == nullptr ? signature_id_t{0U} : (*sigs)[u];
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

  auto expected = hash_map<pair<signature_id_t, signature_id_t>, i32_minutes>{};
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

    // The durations possible at the stop pair: gaps outside get the same
    // answer for all of them.
    auto window = bounds_t{fallback, fallback + i32_minutes{1}};
    auto const extend_by = [&](location_idx_t const from,
                               location_idx_t const to) {
      if (auto const r = utl::lookup(rule_times, transfer_pair{from, to})) {
        extend(window, *r);
      }
    };
    for (auto const from : {lx, locs.parents_[lx]}) {
      for (auto const to : {ly, locs.parents_[ly]}) {
        extend_by(from, to);
      }
    }
    if (auto const station = locs.get_root_idx(lx);
        station == locs.get_root_idx(ly) &&
        !tt.transfer_rules_.is_global_excluded(station)) {
      extend_by(location_idx_t::invalid(), location_idx_t::invalid());
    }
    for (auto const is_arr : {true, false}) {
      for (auto const rs : get_route_stops(is_arr ? lx : ly)) {
        auto const* sigs = sigs_of(rs);
        if (sigs == nullptr ||
            !has_event(tt, rs, is_arr ? event_type::kArr : event_type::kDep)) {
          continue;
        }
        auto const node = virt_node{rs, location_idx_t::invalid()};
        auto const k = get_key(vs,
                               is_arr ? node_pair{node, base_node(ly)}
                                      : node_pair{base_node(lx), node},
                               window);
        auto& all = vs.transfers_[k];
        for (auto t = route_transport_idx_t{0U};
             t != tt.route_transport_ranges_[rs.route_].size(); ++t) {
          auto const none = route_transport_idx_t{0U};
          auto const e = is_arr ? get_expected(sig(sigs, t), {})
                                : get_expected({}, sig(sigs, t));
          for (auto const gap : {e, e - i32_minutes{1}}) {
            all.push_back(
                virt_transfer{is_arr ? t : none, is_arr ? none : t,
                              std::chrono::duration_cast<duration_t>(gap),
                              std::chrono::duration_cast<duration_t>(e)});
          }
        }
      }
    }
    auto const lo = window.from_;
    auto const hi = window.to_ - i32_minutes{1};
    for (auto const x : get_route_stops(lx)) {
      if (!has_event(tt, x, event_type::kArr)) {
        continue;
      }
      auto const* xs = sigs_of(x);
      auto const nx = node_of(x, xs);
      for (auto const y : get_route_stops(ly)) {
        auto const* ys = sigs_of(y);
        auto const ny = node_of(y, ys);
        if (!has_event(tt, y, event_type::kDep) ||
            (is_base(nx) && is_base(ny))) {
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
              auto const e = get_expected(sig(xs, a), sig(ys, b));
              transfers.push_back(virt_transfer{
                  a, b,
                  std::chrono::duration_cast<duration_t>(it->first - arrival),
                  std::chrono::duration_cast<duration_t>(e)});
            }
          }
        }
        if (transfers.empty()) {
          continue;
        }
        auto const k = get_key(vs, node_pair{nx, ny}, window);
        auto& all = vs.transfers_[k];
        all.insert(end(all), begin(transfers), end(transfers));
      }
    }
  }
}

part_idx_t get_part(virt_state const& vs,
                    virt_node const& n,
                    route_transport_idx_t const t) {
  return is_base(n) ? part_idx_t{0U} : vs.parts_[n.rs_.route_][t];
}

unit_idx_t get_unit(timetable const& tt,
                    virt_state& vs,
                    virt_node const& n,
                    part_idx_t const p) {
  return utl::get_or_create(vs.unit_of_, pair{n, p}, [&]() {
    auto const s = is_base(n)
                       ? n.base_
                       : stop{tt.route_location_seq_[n.rs_.route_][n.rs_.idx_]}
                             .location_idx();
    auto const u = unit_idx_t{vs.units_.size()};
    vs.units_.push_back(virt_unit{.node_ = n, .part_ = p, .stop_ = s});
    vs.out_.emplace_back();
    vs.in_.emplace_back();
    vs.place_.push_back(u);
    return u;
  });
}

// The bounds of the unit pairs of a key for the current parts.
void aggregate(timetable const& tt, virt_state& vs, key_idx_t const k) {
  constexpr auto const kNoCell = std::numeric_limits<std::uint32_t>::max();
  auto const window = vs.windows_[k];
  auto& kp = vs.key_units_[k];
  for (auto const& p : kp) {
    vs.bounds_[p] = window;
    vs.expected_[p].clear();
  }
  auto const& [x, y] = vs.keys_[k];
  auto const n_parts = [&](virt_node const& n) {
    return is_base(n) ? 1U : to_idx(vs.n_parts_[n.rs_.route_]);
  };
  auto const ny = n_parts(y);
  auto cell = std::vector<std::uint32_t>(n_parts(x) * ny, kNoCell);
  auto cells = std::vector<pair<part_idx_t, part_idx_t>>{};
  auto agg = std::vector<bounds_t>{};
  auto expected = std::vector<expected_t>{};
  for (auto const& t : vs.transfers_[k]) {
    auto const px = get_part(vs, x, t.from_);
    auto const py = get_part(vs, y, t.to_);
    auto& c = cell[to_idx(px) * ny + to_idx(py)];
    if (c == kNoCell) {
      c = static_cast<std::uint32_t>(cells.size());
      cells.push_back(pair<part_idx_t, part_idx_t>{px, py});
      agg.push_back(window);
      expected.emplace_back();
    }
    narrow(agg[c], t.gap_, t.gap_ >= t.expected_);
    count(expected[c], t.expected_);
  }
  for (auto c = 0U; c != cells.size(); ++c) {
    auto const ux = get_unit(tt, vs, x, cells[c].first);
    auto const uy = get_unit(tt, vs, y, cells[c].second);
    auto const p = pair{ux, uy};
    vs.expected_[p] = std::move(expected[c]);
    if (auto const [it, is_new] = vs.bounds_.emplace(p, agg[c]); is_new) {
      vs.out_[ux].push_back(uy);
      vs.in_[uy].push_back(ux);
      kp.push_back(p);
    } else {
      it->second = agg[c];
    }
  }
}

// The transports of a unit that must leave its part so that the pair can get
// a duration: groups by their own bounds (1-D stabbing; a transport whose own
// bounds are empty forms a group alone), the biggest stays. Empty if this side
// cannot split (a single transport): the other side can then.
std::vector<std::vector<route_transport_idx_t>> get_split(timetable const& tt,
                                                          virt_state& vs,
                                                          unit_idx_t const u,
                                                          unit_idx_t const v,
                                                          bool const is_from) {
  auto const& x = vs.units_[u];
  auto const& y = vs.units_[v];
  auto const& z = is_from ? x : y;
  if (is_base(z.node_)) {
    return {};
  }
  auto const k = vs.key_of_.at(node_pair{x.node_, y.node_});
  auto const n = tt.route_transport_ranges_[z.node_.rs_.route_].size();
  auto b = std::vector<bounds_t>(n, vs.windows_[k]);
  auto seen = std::vector<bool>(n, false);
  for (auto const& t : vs.transfers_[k]) {
    if (get_part(vs, x.node_, t.from_) == x.part_ &&
        get_part(vs, y.node_, t.to_) == y.part_) {
      auto const i = to_idx(is_from ? t.from_ : t.to_);
      seen[i] = true;
      narrow(b[i], t.gap_, t.gap_ >= t.expected_);
    }
  }
  auto const fallback = get_fallback(tt, vs, x.stop_, y.stop_);
  auto const max = u == v ? kMaxTransferTime : kMaxFootpath;
  auto items = std::vector<std::pair<route_transport_idx_t, bounds_t>>{};
  for (auto i = 0U; i != n; ++i) {
    if (seen[i]) {
      items.emplace_back(route_transport_idx_t{i}, b[i]);
    }
  }
  utl::sort(items, [](auto const& a, auto const& c) {
    return std::pair{a.second.to_, a.first} < std::pair{c.second.to_, c.first};
  });
  auto groups =
      std::vector<std::pair<bounds_t, std::vector<route_transport_idx_t>>>{};
  for (auto const& [t, tb] : items) {
    auto const both = groups.empty() ? tb : groups.back().first.intersect(tb);
    if (!groups.empty() && pick_duration(both, fallback, max).has_value()) {
      groups.back().first = both;
      groups.back().second.push_back(t);
    } else {
      groups.push_back({tb, {t}});
    }
  }
  if (groups.size() < 2U) {
    return {};
  }
  auto moved = utl::to_vec(groups, [](auto& g) { return std::move(g.second); });
  moved.erase(std::ranges::max_element(moved, {},
                                       [](auto const& g) { return g.size(); }));
  return moved;
}

// Route parts: a pair of units whose transports cannot agree on one duration
// splits a route, whatever the locations are. New parts for the moved
// transports: the route's pairs are bounded again, those without a duration
// go back to work.
void split_parts(timetable const& tt,
                 virt_state& vs,
                 std::uint32_t& n_split,
                 std::uint32_t& n_unresolved) {
  vs.parts_.resize(tt.n_routes());
  vs.n_parts_.resize(tt.n_routes(), part_idx_t{1U});
  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    if (!vs.route_keys_[r].empty()) {
      vs.parts_[r] = vector_map<route_transport_idx_t, part_idx_t>(
          tt.route_transport_ranges_[r].size(), part_idx_t{0U});
    }
  }
  for (auto k = key_idx_t{0U}; k != vs.keys_.size(); ++k) {
    aggregate(tt, vs, k);
  }
  auto work = std::vector<unit_pair>{};
  for (auto const& [p, b] : vs.bounds_) {
    if (!is_admissible(b, p.first == p.second)) {
      work.push_back(p);
    }
  }
  while (!work.empty()) {
    auto const [u, v] = work.back();
    work.pop_back();
    if (is_admissible(vs.bounds_.at(pair{u, v}), u == v)) {
      continue;
    }
    auto const from = get_split(tt, vs, u, v, true);
    auto const to = get_split(tt, vs, u, v, false);
    if (from.empty() && to.empty()) {
      ++n_unresolved;
      continue;
    }
    auto const use_from =
        !from.empty() && (to.empty() || from.size() <= to.size());
    auto const r = vs.units_[use_from ? u : v].node_.rs_.route_;
    auto& np = vs.n_parts_[r];
    for (auto const& g : use_from ? from : to) {
      for (auto const t : g) {
        vs.parts_[r][t] = np;
      }
      ++np;
    }
    for (auto const k : vs.route_keys_[r]) {
      aggregate(tt, vs, k);
      for (auto const& [a, b] : vs.key_units_[k]) {
        if (!is_admissible(vs.bounds_.at(pair{a, b}), a == b)) {
          work.push_back(pair{a, b});
        }
      }
    }
    work.push_back(pair{u, v});
    ++n_split;
  }
}

unit_idx_t get_place(virt_state const& vs, unit_idx_t u) {
  while (vs.place_[u] != u) {
    u = vs.place_[u];
  }
  return u;
}

// Whether the unit (a place of its own) can join the place p: every place
// pair it touches keeps a duration. Between two stops themselves the walk
// stays. If so, the unit joins: the bounds of the place pairs are narrowed.
// Pairs are keyed by places: a unit's pairs were folded into its place when
// it joined, so a partner's pairs always exist under the partner's place.
bool join(timetable const& tt,
          virt_state& vs,
          unit_idx_t const u,
          unit_idx_t const p) {
  auto merged = hash_map<unit_pair, pair<bounds_t, expected_t>>{};
  auto const fold = [&](unit_pair const key, unit_pair const old) {
    auto& m = utl::get_or_create(merged, key, [&]() {
      auto const b = vs.bounds_.find(key);
      auto const e = vs.expected_.find(key);
      return pair{b == end(vs.bounds_) ? kAnyDuration : b->second,
                  e == end(vs.expected_) ? expected_t{} : e->second};
    });
    m.first = m.first.intersect(vs.bounds_.at(old));
    for (auto const& [e, n] : vs.expected_[old]) {
      count(m.second, e, n);
    }
  };
  auto out = utl::to_vec(vs.out_[u],
                         [&](unit_idx_t const v) { return get_place(vs, v); });
  auto in = utl::to_vec(vs.in_[u],
                        [&](unit_idx_t const v) { return get_place(vs, v); });
  utl::erase_duplicates(out);
  utl::erase_duplicates(in);
  for (auto const q : out) {
    fold(unit_pair{p, q == u ? p : q}, unit_pair{u, q});
  }
  for (auto const q : in) {
    if (q != u) {
      fold(unit_pair{q, p}, unit_pair{q, u});
    }
  }
  for (auto const& [key, m] : merged) {
    auto const [a, b] = key;
    if (is_base(vs, a) && is_base(vs, b)
            ? !m.first.contains(get_fallback(tt, vs, vs.units_[a].stop_,
                                             vs.units_[b].stop_))
            : !is_admissible(m.first, a == b)) {
      return false;
    }
  }
  for (auto& [key, m] : merged) {
    vs.bounds_[key] = m.first;
    vs.expected_[key] = std::move(m.second);
  }
  vs.place_[u] = p;
  return true;
}

// Places: a unit stays at its stop if its stop can take it, else it joins the
// first place at its stop that can, else it is a place of its own.
void merge_places(timetable const& tt,
                  virt_state& vs,
                  std::uint32_t& n_at_stop,
                  std::uint32_t& n_joined,
                  std::uint32_t& n_places) {
  auto places = hash_map<location_idx_t, std::vector<unit_idx_t>>{};
  for (auto u = unit_idx_t{0U}; u != vs.units_.size(); ++u) {
    if (is_base(vs.units_[u].node_)) {
      continue;
    }
    auto const s = vs.units_[u].stop_;
    if (join(tt, vs, u, get_unit(tt, vs, base_node(s), part_idx_t{0U}))) {
      ++n_at_stop;
      continue;
    }
    auto& at_stop = places[s];
    if (utl::any_of(at_stop,
                    [&](unit_idx_t const p) { return join(tt, vs, u, p); })) {
      ++n_joined;
    } else {
      at_stop.push_back(u);
      ++n_places;
    }
  }
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
  auto const is_place = [&](unit_idx_t const u) { return vs.place_[u] == u; };
  auto const to_duration = [](i32_minutes const d) {
    return d >= kNever ? footpath::kMaxDuration
                       : std::chrono::duration_cast<duration_t>(d);
  };

  auto virts = vector_map<unit_idx_t, location_idx_t>(
      vs.units_.size(), location_idx_t::invalid());
  auto n_virts = std::size_t{0U};
  for (auto u = unit_idx_t{0U}; u != vs.units_.size(); ++u) {
    if (!is_place(u) || is_base(vs, u)) {
      continue;
    }
    auto const s = vs.units_[u].stop_;
    auto const fallback = get_fallback(tt, vs, s, s);
    auto const self = vs.bounds_.find(pair{u, u});
    auto const d = self == end(vs.bounds_)
                       ? std::optional{fallback}
                       : pick_duration(self->second, fallback, kMaxTransferTime,
                                       vs.expected_.at(pair{u, u}));
    utl::verify(d.has_value(), "virtual location without transfer time");
    auto l = location{};
    l.src_ = tt.locations_.src_[s];
    l.pos_ = tt.locations_.coordinates_[s];
    l.type_ = location_type::kVirt;
    l.parent_ = s;
    l.transfer_time_ = to_duration(*d);
    virts[u] = register_location(tt, l);
    tt.locations_.children_[s].push_back(virts[u]);
    ++n_virts;
  }
  if (n_virts == 0U) {
    return false;
  }
  auto const location_of = [&](unit_idx_t const u) {
    return is_base(vs, u) ? vs.units_[u].stop_ : virts[u];
  };

  durations[location_idx_t{tt.n_locations() - 1U}];
  auto n_footpaths = std::size_t{0U};
  for (auto const& [p, b] : vs.bounds_) {
    auto const [a, c] = p;
    if (a == c || !is_place(a) || !is_place(c) ||
        (is_base(vs, a) && is_base(vs, c))) {
      continue;
    }
    auto const fallback =
        get_fallback(tt, vs, vs.units_[a].stop_, vs.units_[c].stop_);
    auto const d = pick_duration(b, fallback, kMaxFootpath, vs.expected_.at(p));
    utl::verify(d.has_value(), "place pair without duration");
    if (*d == fallback) {
      continue;
    }
    durations[location_of(a)].push_back(
        footpath{location_of(c), to_duration(*d)});
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
  auto n_regrouped = std::size_t{0U};
  for (auto r = route_idx_t{0U}; r != n_routes; ++r) {
    auto const& parts = vs.parts_[r];
    if (parts.empty()) {
      continue;
    }
    auto const transports = tt.route_transport_ranges_[r];
    auto const seq = tt.route_location_seq_[r];
    seqs.clear();
    seqs.resize(to_idx(vs.n_parts_[r]), stop_seq_t{seq.begin(), seq.end()});
    for (auto p = part_idx_t{0U}; p != seqs.size(); ++p) {
      for (auto i = stop_idx_t{0U}; i != seq.size(); ++i) {
        auto const u = vs.unit_of_.find(
            pair{virt_node{route_stop{r, i}, location_idx_t::invalid()}, p});
        if (u != end(vs.unit_of_)) {
          seqs[p][i] = stop{seq[i]}
                           .with_location(location_of(get_place(vs, u->second)))
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
    n_regrouped += seqs.size() - groups.size();
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
      "{} virtual locations, {} routes -> {}, {} footpaths, {} parts regrouped",
      n_virts, n_routes, tt.n_routes(), n_footpaths, n_regrouped);
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
  auto n_split = 0U;
  auto n_unresolved = 0U;
  split_parts(tt, vs, n_split, n_unresolved);
  auto const n_units = vs.units_.size();
  auto const n_pairs = vs.bounds_.size();
  auto n_at_stop = 0U;
  auto n_joined = 0U;
  auto n_places = 0U;
  merge_places(tt, vs, n_at_stop, n_joined, n_places);
  log(log_lvl::info, "loader.virtual_locations",
      "{} units, {} pairs, {} splits, {} unresolved; {} at their stop, {} "
      "joined, {} places",
      n_units, n_pairs, n_split, n_unresolved, n_at_stop, n_joined, n_places);
  return write(tt, vs, shapes, durations);
}

}  // namespace nigiri::loader
