#include "nigiri/loader/virtual_locations.h"

#include <chrono>
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
using route_stop_idx_t = cista::strong<std::uint32_t, struct route_stop_>;
using route_stop_pair_idx_t =
    cista::strong<std::uint32_t, struct route_stop_pair_>;
using signature_id_t = cista::strong<std::uint32_t, struct signature_id_>;
using sides_t = std::vector<transfer_rule_side_idx>;
using stop_seq_t = std::vector<stop::value_type>;
using route_stop_pair = pair<route_stop_idx_t, route_stop_idx_t>;
using signatures_t = vector_map<route_transport_idx_t, signature_id_t>;
// Durations and how often each one occurs.
using histogram_t = std::vector<pair<duration_t, std::uint32_t>>;
// The durations a footpath (or transfer time) may have: [from, to).
using window_t = interval<duration_t>;

constexpr auto const kNever = duration_t{routing::kMaxTravelTime};
constexpr auto const kAnyDuration =
    window_t{duration_t::min(), duration_t::max()};
constexpr auto const kMaxFootpath = footpath::kMaxDuration - duration_t{1};
constexpr auto const kMaxTransferTime =
    duration_t{kNoTransferAllowed} - duration_t{1};

// A route of the timetable (base route) at one of its stops.
struct base_route_stop {
  CISTA_COMPARABLE()
  route_idx_t route_;
  stop_idx_t idx_;
};

struct transfer {
  route_transport_idx_t from_, to_;
  duration_t gap_;
  duration_t expected_;
};

// A route at one of its stops: a base route until it splits, then the routes
// the split makes (indices above the timetable's). Or the stop itself (no
// route, the base): everything without a signature there.
struct route_stop {
  CISTA_COMPARABLE()
  route_idx_t route_;
  stop_idx_t idx_;
  location_idx_t stop_;
};

// A stop pair: the walk before any virtual location (never if none) and the
// durations the rules allow there (set when its transfers are added).
struct stop_pair {
  duration_t fallback_;
  window_t window_;
};

void count(histogram_t& h, duration_t const e, std::uint32_t const n = 1U) {
  auto const it = utl::find_if(h, [&](auto const& x) { return x.first == e; });
  if (it == end(h)) {
    h.push_back(pair<duration_t, std::uint32_t>{e, n});
  } else {
    it->second += n;
  }
}

// A meeting that must work allows at most its gap, one that must not work
// needs more than its gap.
void narrow(window_t& w, duration_t const gap, bool const must_work) {
  if (must_work) {
    w.to_ = std::min(w.to_, gap + duration_t{1});
  } else {
    w.from_ = std::max(w.from_, gap + duration_t{1});
  }
}

// The smallest window covering both.
void extend(window_t& a, window_t const& b) {
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

duration_t get_minutes(transfer_rule const& r) {
  return r.duration_ == footpath::kMaxDuration ? kNever
                                               : duration_t{r.duration_};
}

// The window of the rule times per stop pair.
hash_map<transfer_pair, window_t> get_rule_times(timetable const& tt) {
  auto rule_times = hash_map<transfer_pair, window_t>{};
  for (auto const& rule : tt.transfer_rules_.rules_) {
    auto const m = get_minutes(rule);
    auto const r = window_t{m, m + duration_t{1}};
    extend(rule_times.emplace(transfer_pair{rule.from_stop_, rule.to_stop_}, r)
               .first->second,
           r);
  }
  return rule_times;
}

bool has_event(timetable const& tt,
               base_route_stop const rs,
               event_type const ev) {
  auto const seq = tt.route_location_seq_[rs.route_];
  auto const stp = stop{seq[rs.idx_]};
  return ev == event_type::kArr
             ? rs.idx_ != 0U && stp.out_allowed()
             : rs.idx_ + 1U != seq.size() && stp.in_allowed();
}

// The duration for the window: the fallback if it fits, a ban if nothing has
// to work, else the most frequent expected time within it, else the largest
// storable one. None if no storable duration fits.
std::optional<duration_t> pick_duration(window_t const& w,
                                        duration_t const fallback,
                                        duration_t const max,
                                        histogram_t const& expected = {}) {
  if (w.contains(fallback)) {
    return fallback;
  }
  if (w.contains(kNever)) {
    return kNever;
  }
  auto const to = std::min(w.to_, max + duration_t{1});
  if (w.from_ >= to) {
    return std::nullopt;
  }
  auto best = to - duration_t{1};
  auto best_n = std::uint32_t{0U};
  for (auto const& [e, n] : expected) {
    if (w.from_ <= e && e < to && (n > best_n || (n == best_n && e < best))) {
      best = e;
      best_n = n;
    }
  }
  return best;
}

bool is_admissible(window_t const& w, bool const is_self) {
  return pick_duration(w, kNever, is_self ? kMaxTransferTime : kMaxFootpath)
      .has_value();
}

struct virt_state {
  std::vector<sides_t> signatures_;
  hash_map<base_route_stop, signatures_t> signature_of_;
  hash_map<pair<location_idx_t, location_idx_t>, stop_pair> stop_pairs_;
  // Routes: the base route of each, itself for the timetable's; per base
  // route the route of each transport (allocated on its first split).
  vector_map<route_idx_t, route_idx_t> route_base_route_;
  vector_map<route_idx_t, vector_map<route_transport_idx_t, route_idx_t>>
      base_route_transport_route_;
  vector_map<route_stop_idx_t, route_stop> route_stops_;
  hash_map<route_stop, route_stop_idx_t> route_stop_of_;
  vector_map<route_stop_idx_t, std::vector<route_stop_idx_t>> route_stop_out_,
      route_stop_in_;
  // The route stop standing for the location of the route stop: itself for a
  // location.
  vector_map<route_stop_idx_t, route_stop_idx_t> route_stop_location_;
  // Pairs with transfers: their transfers, the window the transfers leave
  // (after merging between two locations, keyed by the route stops standing
  // for them) and how often each expected time occurs.
  vector_map<route_stop_pair_idx_t, route_stop_pair> route_stop_pairs_;
  hash_map<route_stop_pair, route_stop_pair_idx_t> route_stop_pair_of_;
  vector_map<route_stop_pair_idx_t, std::vector<transfer>>
      route_stop_pair_transfers_;
  vector_map<route_stop_pair_idx_t, window_t> route_stop_pair_windows_;
  vector_map<route_stop_pair_idx_t, histogram_t> route_stop_pair_expected_;
  vector_map<route_idx_t, std::vector<route_stop_pair_idx_t>>
      route_route_stop_pairs_;
};

bool is_base(route_stop const& s) { return s.route_ == route_idx_t::invalid(); }

bool is_base(virt_state const& vs, route_stop_idx_t const s) {
  return is_base(vs.route_stops_[s]);
}

route_stop base_stop(location_idx_t const l) {
  return {route_idx_t::invalid(), stop_idx_t{0U}, l};
}

stop_pair& get_stop_pair(timetable const& tt,
                         virt_state& vs,
                         location_idx_t const from,
                         location_idx_t const to) {
  return utl::get_or_create(vs.stop_pairs_, pair{from, to}, [&]() {
    auto const walk =
        routing::shortest_transfer(tt, nullptr, kDefaultProfile, from, to);
    return stop_pair{walk.value_or(kNever), kAnyDuration};
  });
}

duration_t get_fallback(timetable const& tt,
                        virt_state& vs,
                        location_idx_t const from,
                        location_idx_t const to) {
  return get_stop_pair(tt, vs, from, to).fallback_;
}

// The route of a transport of the base route.
route_idx_t get_route(virt_state const& vs,
                      route_idx_t const base_route,
                      route_transport_idx_t const t) {
  auto const& routes = vs.base_route_transport_route_[base_route];
  return routes.empty() ? base_route : routes[t];
}

route_stop_idx_t get_route_stop(virt_state& vs, route_stop const& s) {
  return utl::get_or_create(vs.route_stop_of_, s, [&]() {
    auto const idx = route_stop_idx_t{vs.route_stops_.size()};
    vs.route_stops_.push_back(s);
    vs.route_stop_out_.emplace_back();
    vs.route_stop_in_.emplace_back();
    vs.route_stop_location_.push_back(idx);
    return idx;
  });
}

// The pair, registered with its routes on first use.
route_stop_pair_idx_t get_pair(virt_state& vs,
                               route_stop_idx_t const a,
                               route_stop_idx_t const b,
                               window_t const& window) {
  return utl::get_or_create(
      vs.route_stop_pair_of_, route_stop_pair{a, b}, [&]() {
        auto const p = route_stop_pair_idx_t{vs.route_stop_pairs_.size()};
        vs.route_stop_pairs_.push_back(route_stop_pair{a, b});
        vs.route_stop_pair_transfers_.emplace_back();
        vs.route_stop_pair_windows_.push_back(window);
        vs.route_stop_pair_expected_.emplace_back();
        vs.route_stop_out_[a].push_back(b);
        vs.route_stop_in_[b].push_back(a);
        auto const& x = vs.route_stops_[a];
        auto const& y = vs.route_stops_[b];
        if (!is_base(x)) {
          vs.route_route_stop_pairs_[x.route_].push_back(p);
        }
        if (!is_base(y) && (is_base(x) || x.route_ != y.route_)) {
          vs.route_route_stop_pairs_[y.route_].push_back(p);
        }
        return p;
      });
}

void add_transfer(virt_state& vs,
                  route_stop_pair_idx_t const p,
                  transfer const& t) {
  vs.route_stop_pair_transfers_[p].push_back(t);
  narrow(vs.route_stop_pair_windows_[p], t.gap_, t.gap_ >= t.expected_);
  count(vs.route_stop_pair_expected_[p], t.expected_);
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
            utl::get_or_create(vs.signature_of_, base_route_stop{r, i}, [&]() {
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
    routing::for_each_meta(tt, routing::location_match_mode::kOnlyChildren,
                           from, [&](location_idx_t const f) {
                             routing::for_each_meta(
                                 tt,
                                 routing::location_match_mode::kOnlyChildren,
                                 to, [&](location_idx_t const t) {
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
  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    vs.route_base_route_.push_back(r);
  }
  vs.base_route_transport_route_.resize(tt.n_routes());
  vs.route_route_stop_pairs_.resize(tt.n_routes());
  auto const location_of = [&](base_route_stop const rs) {
    return stop{tt.route_location_seq_[rs.route_][rs.idx_]}.location_idx();
  };
  auto const sigs_of = [&](base_route_stop const rs) -> signatures_t const* {
    auto const it = vs.signature_of_.find(rs);
    return it == end(vs.signature_of_) ? nullptr : &it->second;
  };
  auto const stop_of = [&](base_route_stop const rs, signatures_t const* sigs) {
    return get_route_stop(
        vs, sigs != nullptr ? route_stop{rs.route_, rs.idx_, location_of(rs)}
                            : base_stop(location_of(rs)));
  };
  auto const sig = [](signatures_t const* sigs, route_transport_idx_t const u) {
    return sigs == nullptr ? signature_id_t{0U} : (*sigs)[u];
  };

  auto route_stops = hash_map<location_idx_t, std::vector<base_route_stop>>{};
  auto const get_route_stops =
      [&](location_idx_t const l) -> std::vector<base_route_stop> const& {
    return utl::get_or_create(route_stops, l, [&]() {
      auto v = std::vector<base_route_stop>{};
      for (auto const r : tt.location_routes_[l]) {
        auto const seq = tt.route_location_seq_[r];
        for (auto i = stop_idx_t{0U}; i != seq.size(); ++i) {
          if (stop{seq[i]}.location_idx() == l) {
            v.push_back(base_route_stop{r, i});
          }
        }
      }
      return v;
    });
  };
  using events_t = std::vector<std::pair<i32_minutes, route_transport_idx_t>>;
  auto event_store = std::deque<events_t>{};
  auto events = hash_map<pair<base_route_stop, event_type>, events_t*>{};
  auto const get_events = [&](base_route_stop const rs,
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

  auto expected = hash_map<pair<signature_id_t, signature_id_t>, duration_t>{};
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
    auto window = window_t{fallback, fallback + duration_t{1}};
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
    vs.stop_pairs_.at(pair{lx, ly}).window_ = window;
    for (auto const is_arr : {true, false}) {
      for (auto const rs : get_route_stops(is_arr ? lx : ly)) {
        auto const* sigs = sigs_of(rs);
        if (sigs == nullptr ||
            !has_event(tt, rs, is_arr ? event_type::kArr : event_type::kDep)) {
          continue;
        }
        auto const x =
            is_arr ? stop_of(rs, sigs) : get_route_stop(vs, base_stop(lx));
        auto const y =
            is_arr ? get_route_stop(vs, base_stop(ly)) : stop_of(rs, sigs);
        auto const p = get_pair(vs, x, y, window);
        for (auto t = route_transport_idx_t{0U};
             t != tt.route_transport_ranges_[rs.route_].size(); ++t) {
          auto const none = route_transport_idx_t{0U};
          auto const e = is_arr ? get_expected(sig(sigs, t), {})
                                : get_expected({}, sig(sigs, t));
          for (auto const gap : {e, e - duration_t{1}}) {
            add_transfer(
                vs, p, transfer{is_arr ? t : none, is_arr ? none : t, gap, e});
          }
        }
      }
    }
    auto const lo = window.from_;
    auto const hi = window.to_ - duration_t{1};
    for (auto const x : get_route_stops(lx)) {
      if (!has_event(tt, x, event_type::kArr)) {
        continue;
      }
      auto const* xs = sigs_of(x);
      for (auto const y : get_route_stops(ly)) {
        auto const* ys = sigs_of(y);
        if (!has_event(tt, y, event_type::kDep) ||
            (xs == nullptr && ys == nullptr)) {
          continue;
        }
        auto const& departures = get_events(y, event_type::kDep);
        auto p = std::optional<route_stop_pair_idx_t>{};
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
              if (!p.has_value()) {
                auto const sx = stop_of(x, xs);
                auto const sy = stop_of(y, ys);
                p = get_pair(vs, sx, sy, window);
              }
              add_transfer(vs, *p,
                           transfer{a, b,
                                    std::chrono::duration_cast<duration_t>(
                                        it->first - arrival),
                                    get_expected(sig(xs, a), sig(ys, b))});
            }
          }
        }
      }
    }
  }
}

// After a split: the transfers whose transports left the routes of the pair
// move to the pairs of their new routes; the windows start over.
void repartition(virt_state& vs, route_stop_pair_idx_t const p) {
  auto const [a, b] = vs.route_stop_pairs_[p];
  auto const x = vs.route_stops_[a];
  auto const y = vs.route_stops_[b];
  auto const window = vs.stop_pairs_.at(pair{x.stop_, y.stop_}).window_;
  auto const transfers = std::move(vs.route_stop_pair_transfers_[p]);
  vs.route_stop_pair_transfers_[p] = {};
  vs.route_stop_pair_windows_[p] = window;
  vs.route_stop_pair_expected_[p].clear();
  auto const route_of = [&](route_stop const& s,
                            route_transport_idx_t const t) {
    return is_base(s) ? s.route_
                      : get_route(vs, vs.route_base_route_[s.route_], t);
  };
  for (auto const& t : transfers) {
    auto sx = x;
    auto sy = y;
    sx.route_ = route_of(x, t.from_);
    sy.route_ = route_of(y, t.to_);
    if (sx.route_ == x.route_ && sy.route_ == y.route_) {
      add_transfer(vs, p, t);
    } else {
      auto const a2 = get_route_stop(vs, sx);
      auto const b2 = get_route_stop(vs, sy);
      add_transfer(vs, get_pair(vs, a2, b2, window), t);
    }
  }
}

// The transports of a route stop that must leave its route so that the pair
// can get a duration: groups by their own windows (1-D stabbing; a transport
// whose own window is empty forms a group alone), the biggest stays. Empty if
// this side cannot split (a single transport): the other side can then.
std::vector<std::vector<route_transport_idx_t>> get_split(
    timetable const& tt,
    virt_state& vs,
    route_stop_pair_idx_t const p,
    bool const is_from) {
  auto const [a, b] = vs.route_stop_pairs_[p];
  auto const& x = vs.route_stops_[a];
  auto const& y = vs.route_stops_[b];
  auto const& z = is_from ? x : y;
  if (is_base(z)) {
    return {};
  }
  auto const& sp = vs.stop_pairs_.at(pair{x.stop_, y.stop_});
  auto const n =
      tt.route_transport_ranges_[vs.route_base_route_[z.route_]].size();
  auto w = std::vector<window_t>(n, sp.window_);
  auto seen = std::vector<bool>(n, false);
  for (auto const& t : vs.route_stop_pair_transfers_[p]) {
    auto const i = to_idx(is_from ? t.from_ : t.to_);
    seen[i] = true;
    narrow(w[i], t.gap_, t.gap_ >= t.expected_);
  }
  auto const max = a == b ? kMaxTransferTime : kMaxFootpath;
  auto items = std::vector<std::pair<route_transport_idx_t, window_t>>{};
  for (auto i = 0U; i != n; ++i) {
    if (seen[i]) {
      items.emplace_back(route_transport_idx_t{i}, w[i]);
    }
  }
  utl::sort(items, [](auto const& l, auto const& r) {
    return std::pair{l.second.to_, l.first} < std::pair{r.second.to_, r.first};
  });
  auto groups =
      std::vector<std::pair<window_t, std::vector<route_transport_idx_t>>>{};
  for (auto const& [t, tw] : items) {
    auto const both = groups.empty() ? tw : groups.back().first.intersect(tw);
    if (!groups.empty() && pick_duration(both, sp.fallback_, max).has_value()) {
      groups.back().first = both;
      groups.back().second.push_back(t);
    } else {
      groups.push_back({tw, {t}});
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

// Routes: a pair of route stops whose transports cannot agree on one duration
// splits a route, whatever the locations are. New routes for the moved
// transports, with the next indices: the route's pairs are partitioned again,
// those without a duration go back to work.
void split_routes(timetable const& tt,
                  virt_state& vs,
                  std::uint32_t& n_split,
                  std::uint32_t& n_unresolved) {
  auto const is_pair_admissible = [&](route_stop_pair_idx_t const p) {
    auto const [a, b] = vs.route_stop_pairs_[p];
    return is_admissible(vs.route_stop_pair_windows_[p], a == b);
  };
  auto work = std::vector<route_stop_pair_idx_t>{};
  for (auto p = route_stop_pair_idx_t{0U}; p != vs.route_stop_pairs_.size();
       ++p) {
    if (!is_pair_admissible(p)) {
      work.push_back(p);
    }
  }
  while (!work.empty()) {
    auto const p = work.back();
    work.pop_back();
    if (is_pair_admissible(p)) {
      continue;
    }
    auto const from = get_split(tt, vs, p, true);
    auto const to = get_split(tt, vs, p, false);
    if (from.empty() && to.empty()) {
      ++n_unresolved;
      continue;
    }
    auto const use_from =
        !from.empty() && (to.empty() || from.size() <= to.size());
    auto const [a, b] = vs.route_stop_pairs_[p];
    auto const r = vs.route_stops_[use_from ? a : b].route_;
    auto const base_route = vs.route_base_route_[r];
    auto& routes = vs.base_route_transport_route_[base_route];
    if (routes.empty()) {
      routes = vector_map<route_transport_idx_t, route_idx_t>(
          tt.route_transport_ranges_[base_route].size(), base_route);
    }
    auto const first_new_route = route_idx_t{vs.route_base_route_.size()};
    for (auto const& g : use_from ? from : to) {
      auto const new_route = route_idx_t{vs.route_base_route_.size()};
      vs.route_base_route_.push_back(base_route);
      vs.route_route_stop_pairs_.emplace_back();
      for (auto const t : g) {
        routes[t] = new_route;
      }
    }
    auto const pairs = vs.route_route_stop_pairs_[r];
    for (auto const q : pairs) {
      repartition(vs, q);
    }
    auto const recheck = [&](route_idx_t const q) {
      for (auto const pq : vs.route_route_stop_pairs_[q]) {
        if (!is_pair_admissible(pq)) {
          work.push_back(pq);
        }
      }
    };
    recheck(r);
    for (auto q = first_new_route;
         q != route_idx_t{vs.route_base_route_.size()}; ++q) {
      recheck(q);
    }
    work.push_back(p);
    ++n_split;
  }
}

// Whether the route stop (a location of its own) can join the location l:
// every location pair it touches keeps a duration. Between two bases the
// walk stays. If so, the route stop joins: the windows of the location pairs
// are narrowed. Pairs are keyed by locations: a route stop's pairs were folded
// into its location when it joined, so a partner's pairs always exist under
// the partner's location.
bool join(timetable const& tt,
          virt_state& vs,
          route_stop_idx_t const s,
          route_stop_idx_t const l) {
  auto merged = hash_map<route_stop_pair, pair<window_t, histogram_t>>{};
  auto const fold = [&](route_stop_pair const key, route_stop_pair const old) {
    auto& m = utl::get_or_create(merged, key, [&]() {
      auto const it = vs.route_stop_pair_of_.find(key);
      return it == end(vs.route_stop_pair_of_)
                 ? pair{kAnyDuration, histogram_t{}}
                 : pair{vs.route_stop_pair_windows_[it->second],
                        vs.route_stop_pair_expected_[it->second]};
    });
    auto const o = vs.route_stop_pair_of_.at(old);
    m.first = m.first.intersect(vs.route_stop_pair_windows_[o]);
    for (auto const& [e, n] : vs.route_stop_pair_expected_[o]) {
      count(m.second, e, n);
    }
  };
  auto const location = [&](route_stop_idx_t const v) {
    return vs.route_stop_location_[v];
  };
  auto out = utl::to_vec(vs.route_stop_out_[s], location);
  auto in = utl::to_vec(vs.route_stop_in_[s], location);
  utl::erase_duplicates(out);
  utl::erase_duplicates(in);
  for (auto const q : out) {
    fold(route_stop_pair{l, q == s ? l : q}, route_stop_pair{s, q});
  }
  for (auto const q : in) {
    if (q != s) {
      fold(route_stop_pair{q, l}, route_stop_pair{q, s});
    }
  }
  for (auto const& [key, m] : merged) {
    auto const [a, b] = key;
    if (is_base(vs, a) && is_base(vs, b)
            ? !m.first.contains(get_fallback(tt, vs, vs.route_stops_[a].stop_,
                                             vs.route_stops_[b].stop_))
            : !is_admissible(m.first, a == b)) {
      return false;
    }
  }
  for (auto& [key, m] : merged) {
    auto const p = get_pair(vs, key.first, key.second, kAnyDuration);
    vs.route_stop_pair_windows_[p] = m.first;
    vs.route_stop_pair_expected_[p] = std::move(m.second);
  }
  vs.route_stop_location_[s] = l;
  return true;
}

// Locations: a route stop joins the first location at its stop that can take
// it, the base first, else it is a location of its own.
void merge_locations(timetable const& tt,
                     virt_state& vs,
                     std::uint32_t& n_joined,
                     std::uint32_t& n_locations) {
  auto locations = hash_map<location_idx_t, std::vector<route_stop_idx_t>>{};
  for (auto s = route_stop_idx_t{0U}; s != vs.route_stops_.size(); ++s) {
    if (is_base(vs, s)) {
      continue;
    }
    auto const st = vs.route_stops_[s].stop_;
    auto& at_stop = locations[st];
    if (at_stop.empty()) {
      at_stop.push_back(get_route_stop(vs, base_stop(st)));
    }
    if (utl::any_of(at_stop, [&](route_stop_idx_t const l) {
          return join(tt, vs, s, l);
        })) {
      ++n_joined;
    } else {
      at_stop.push_back(s);
      ++n_locations;
    }
  }
}

void rewrite_routes(timetable& tt,
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
  auto const is_location = [&](route_stop_idx_t const s) {
    return vs.route_stop_location_[s] == s;
  };
  auto const to_duration = [](duration_t const d) {
    return d >= kNever ? footpath::kMaxDuration : d;
  };

  auto virts = vector_map<route_stop_idx_t, location_idx_t>(
      vs.route_stops_.size(), location_idx_t::invalid());
  auto n_virts = std::size_t{0U};
  for (auto s = route_stop_idx_t{0U}; s != vs.route_stops_.size(); ++s) {
    if (!is_location(s) || is_base(vs, s)) {
      continue;
    }
    auto const st = vs.route_stops_[s].stop_;
    auto const fallback = get_fallback(tt, vs, st, st);
    auto const self = vs.route_stop_pair_of_.find(pair{s, s});
    auto const d =
        self == end(vs.route_stop_pair_of_)
            ? std::optional{fallback}
            : pick_duration(vs.route_stop_pair_windows_[self->second], fallback,
                            kMaxTransferTime,
                            vs.route_stop_pair_expected_[self->second]);
    utl::verify(d.has_value(), "virtual location without transfer time");
    auto l = location{};
    l.src_ = tt.locations_.src_[st];
    l.pos_ = tt.locations_.coordinates_[st];
    l.type_ = location_type::kVirt;
    l.parent_ = st;
    l.transfer_time_ = to_duration(*d);
    virts[s] = register_location(tt, l);
    tt.locations_.children_[st].push_back(virts[s]);
    ++n_virts;
  }
  if (n_virts == 0U) {
    return false;
  }
  auto const location_of = [&](route_stop_idx_t const s) {
    return is_base(vs, s) ? vs.route_stops_[s].stop_ : virts[s];
  };

  durations[location_idx_t{tt.n_locations() - 1U}];
  auto n_footpaths = std::size_t{0U};
  for (auto p = route_stop_pair_idx_t{0U}; p != vs.route_stop_pairs_.size();
       ++p) {
    auto const [a, c] = vs.route_stop_pairs_[p];
    if (a == c || !is_location(a) || !is_location(c) ||
        (is_base(vs, a) && is_base(vs, c))) {
      continue;
    }
    auto const fallback = get_fallback(tt, vs, vs.route_stops_[a].stop_,
                                       vs.route_stops_[c].stop_);
    auto const d = pick_duration(vs.route_stop_pair_windows_[p], fallback,
                                 kMaxFootpath, vs.route_stop_pair_expected_[p]);
    utl::verify(d.has_value(), "location pair without duration");
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
  // The routes of each base route: itself and the ones its splits made.
  auto base_route_routes = hash_map<route_idx_t, std::vector<route_idx_t>>{};
  for (auto r = route_idx_t{n_routes};
       r != route_idx_t{vs.route_base_route_.size()}; ++r) {
    base_route_routes[vs.route_base_route_[r]].push_back(r);
  }
  auto seqs = hash_map<route_idx_t, stop_seq_t>{};
  auto n_regrouped = std::size_t{0U};
  for (auto r = route_idx_t{0U}; r != n_routes; ++r) {
    if (vs.route_route_stop_pairs_[r].empty()) {
      continue;
    }
    auto const transports = tt.route_transport_ranges_[r];
    auto const seq = tt.route_location_seq_[r];
    auto routes = std::vector{r};
    if (auto const it = base_route_routes.find(r);
        it != end(base_route_routes)) {
      routes.insert(end(routes), begin(it->second), end(it->second));
    }
    seqs.clear();
    for (auto const route : routes) {
      auto& route_seq = seqs[route];
      route_seq.assign(seq.begin(), seq.end());
      for (auto i = stop_idx_t{0U}; i != seq.size(); ++i) {
        auto const s = vs.route_stop_of_.find(
            route_stop{route, i, stop{seq[i]}.location_idx()});
        if (s != end(vs.route_stop_of_)) {
          route_seq[i] = stop{seq[i]}
                             .with_location(location_of(
                                 vs.route_stop_location_[s->second]))
                             .value();
        }
      }
    }
    auto groups =
        std::vector<std::pair<stop_seq_t, std::vector<transport_idx_t>>>{};
    for (auto t = route_transport_idx_t{0U}; t != transports.size(); ++t) {
      auto const& s = seqs.at(get_route(vs, r, t));
      auto g = utl::find_if(
          groups, [&](auto const& group) { return group.first == s; });
      if (g == end(groups)) {
        g = groups.insert(end(groups), {s, {}});
      }
      g->second.push_back(transports.from_ + to_idx(t));
    }
    n_regrouped += routes.size() - groups.size();
    route_transports[r] = std::move(groups.front().second);
    std::ranges::copy(groups.front().first, tt.route_location_seq_[r].begin());
    for (auto& [group_seq, group_transports] : std::span{groups}.subspan(1U)) {
      origin.push_back(r);
      route_transports.push_back(std::move(group_transports));
      tt.route_location_seq_.emplace_back(group_seq);
    }
  }
  rewrite_routes(tt, shapes, origin, route_transports);

  log(log_lvl::info, "loader.virtual_locations",
      "{} virtual locations, {} routes -> {}, {} footpaths, {} routes "
      "regrouped",
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
  split_routes(tt, vs, n_split, n_unresolved);
  auto const n_route_stops = vs.route_stops_.size();
  auto const n_pairs = vs.route_stop_pairs_.size();
  auto n_joined = 0U;
  auto n_locations = 0U;
  merge_locations(tt, vs, n_joined, n_locations);
  log(log_lvl::info, "loader.virtual_locations",
      "{} route stops, {} pairs, {} splits, {} unresolved; {} joined, {} "
      "locations",
      n_route_stops, n_pairs, n_split, n_unresolved, n_joined, n_locations);
  return write(tt, vs, shapes, durations);
}

}  // namespace nigiri::loader
