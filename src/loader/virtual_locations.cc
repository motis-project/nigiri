#include "nigiri/loader/virtual_locations.h"

#include <chrono>
#include <cstdint>
#include <algorithm>
#include <deque>
#include <optional>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

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
using split_route_idx_t = cista::strong<std::uint32_t, struct split_route_>;
using signature_idx_t = cista::strong<std::uint32_t, struct signature_idx_>;
using stop_seq_t = std::vector<stop::value_type>;
using route_stop_pair_t = pair<route_stop_idx_t, route_stop_idx_t>;
using histogram_t = std::vector<pair<duration_t, std::uint32_t>>;
using window_t = interval<duration_t>;

constexpr auto const kBaseTransport = route_transport_idx_t{0U};
constexpr auto const kEmptySig = signature_idx_t{0U};
constexpr auto const kNever = duration_t{routing::kMaxTravelTime};
constexpr auto const kAnyDuration =
    window_t{duration_t::min(), duration_t::max()};
constexpr auto const kMaxFootpath = footpath::kMaxDuration - duration_t{1};
constexpr auto const kMaxTransferTime =
    duration_t{kNoTransferAllowed} - duration_t{1};

struct transfer {
  route_transport_idx_t from_, to_;
  duration_t gap_;
  duration_t expected_;
};

struct route_stop {
  CISTA_COMPARABLE()
  split_route_idx_t route_;
  stop_idx_t idx_;
  location_idx_t stop_;
};

struct stop_pair {
  duration_t fallback_;
  window_t window_;
};

struct location_pair {
  route_stop_idx_t to_;
  window_t window_;
  histogram_t expected_;
};

using events_t = std::vector<std::pair<i32_minutes, route_transport_idx_t>>;

void count(histogram_t& h, duration_t const e, std::uint32_t const n = 1U) {
  auto const it = utl::find_if(h, [&](auto const& x) { return x.first == e; });
  if (it == end(h)) {
    h.push_back(pair<duration_t, std::uint32_t>{e, n});
  } else {
    it->second += n;
  }
}

void narrow(window_t& w, duration_t const gap, bool const must_work) {
  if (must_work) {
    w.to_ = std::min(w.to_, gap + duration_t{1});
  } else {
    w.from_ = std::max(w.from_, gap + duration_t{1});
  }
}

void extend(window_t& a, window_t const& b) {
  a.from_ = std::min(a.from_, b.from_);
  a.to_ = std::max(a.to_, b.to_);
}

duration_t get_minutes(transfer_rule const& r) {
  return r.duration_ == footpath::kMaxDuration ? kNever
                                               : duration_t{r.duration_};
}

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
               route_idx_t const r,
               stop_idx_t const i,
               event_type const ev) {
  auto const seq = tt.route_location_seq_[r];
  auto const stp = stop{seq[i]};
  return ev == event_type::kArr ? i != 0U && stp.out_allowed()
                                : i + 1U != seq.size() && stp.in_allowed();
}

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
  auto in_window = expected | std::views::filter([&](auto const& x) {
                     return w.from_ <= x.first && x.first < to;
                   });
  auto const best = std::ranges::max_element(in_window, {}, [](auto const& x) {
    return std::pair{x.second, -x.first.count()};
  });
  return best == in_window.end() ? to - duration_t{1} : best->first;
}

bool is_admissible(window_t const& w, bool const is_self) {
  return pick_duration(w, kNever, is_self ? kMaxTransferTime : kMaxFootpath)
      .has_value();
}

bool is_base(route_stop const& s) {
  return s.route_ == split_route_idx_t::invalid();
}

route_stop get_base_stop(location_idx_t const l) {
  return {split_route_idx_t::invalid(), stop_idx_t{0U}, l};
}

split_route_idx_t to_split_route(route_idx_t const r) {
  return split_route_idx_t{to_idx(r)};
}

struct transport_routes {
  split_route_idx_t get_route(route_idx_t const base_route,
                              route_transport_idx_t const t) const {
    auto const& route = transport_route_[base_route];
    return route.empty() ? to_split_route(base_route) : route[t];
  }

  vector_map<split_route_idx_t, route_idx_t> base_route_;
  vector_map<route_idx_t, vector_map<route_transport_idx_t, split_route_idx_t>>
      transport_route_;
};

struct route_stop_pairs {
  bool is_base(route_stop_idx_t const s) const {
    return loader::is_base(route_stops_[s]);
  }

  route_stop_idx_t get_or_create_route_stop(route_stop const& s) {
    if (auto const it = route_stop_of_.find(s); it != end(route_stop_of_)) {
      return it->second;
    }
    if (!loader::is_base(s)) {
      get_or_create_route_stop(get_base_stop(s.stop_));
    }
    auto const idx = route_stop_idx_t{route_stops_.size()};
    route_stops_.push_back(s);
    route_stop_of_.emplace(s, idx);
    out_.emplace_back();
    in_.emplace_back();
    return idx;
  }

  route_stop_pair_idx_t get_or_create_pair(route_stop_idx_t const a,
                                           route_stop_idx_t const b,
                                           window_t const& window) {
    return utl::get_or_create(pair_of_, route_stop_pair_t{a, b}, [&]() {
      auto const p = route_stop_pair_idx_t{pairs_.size()};
      pairs_.push_back(route_stop_pair_t{a, b});
      transfers_.emplace_back();
      windows_.push_back(window);
      expected_.emplace_back();
      out_[a].push_back(b);
      in_[b].push_back(a);
      auto const& x = route_stops_[a];
      auto const& y = route_stops_[b];
      if (!loader::is_base(x)) {
        of_route_[x.route_].push_back(p);
      }
      if (!loader::is_base(y) && (loader::is_base(x) || x.route_ != y.route_)) {
        of_route_[y.route_].push_back(p);
      }
      return p;
    });
  }

  void add_transfer(route_stop_pair_idx_t const p, transfer const& t) {
    transfers_[p].push_back(t);
    narrow(windows_[p], t.gap_, t.gap_ >= t.expected_);
    count(expected_[p], t.expected_);
  }

  hash_map<pair<location_idx_t, location_idx_t>, stop_pair> stop_pairs_;
  vector_map<route_stop_idx_t, route_stop> route_stops_;
  hash_map<route_stop, route_stop_idx_t> route_stop_of_;
  vector_map<route_stop_idx_t, std::vector<route_stop_idx_t>> out_, in_;
  vector_map<route_stop_pair_idx_t, route_stop_pair_t> pairs_;
  hash_map<route_stop_pair_t, route_stop_pair_idx_t> pair_of_;
  vector_map<route_stop_pair_idx_t, std::vector<transfer>> transfers_;
  vector_map<route_stop_pair_idx_t, window_t> windows_;
  vector_map<route_stop_pair_idx_t, histogram_t> expected_;
  vector_map<split_route_idx_t, std::vector<route_stop_pair_idx_t>> of_route_;
};

struct route_stop_locations {
  location_pair const* find(route_stop_idx_t const from,
                            route_stop_idx_t const to) const {
    auto const& v = pairs_[from];
    auto const it =
        std::lower_bound(begin(v), end(v), to,
                         [](location_pair const& x, route_stop_idx_t const t) {
                           return x.to_ < t;
                         });
    return it != end(v) && it->to_ == to ? &*it : nullptr;
  }

  vector_map<route_stop_idx_t, route_stop_idx_t> location_;
  vector_map<route_stop_idx_t, std::vector<location_pair>> pairs_;
};

duration_t get_walk(timetable const& tt,
                    location_idx_t const from,
                    location_idx_t const to) {
  return routing::shortest_transfer(tt, nullptr, kDefaultProfile, from, to)
      .value_or(kNever);
}

// ==========
// Signatures
// ----------

using sig_t = std::vector<transfer_rule_side_idx>;
using transport_sigs_t = vector_map<route_transport_idx_t, signature_idx_t>;

struct signatures {
  vector_map<signature_idx_t, sig_t> sides_;
  hash_map<route_stop, transport_sigs_t> transport_signatures_;
};

signatures get_signatures(timetable const& tt) {
  auto signature_of = hash_map<sig_t, signature_idx_t>{};

  auto sigs = signatures{};
  sigs.sides_.emplace_back();  // kEmptySig

  auto sig = sig_t{};
  auto rules = std::vector<section_rules>{};

  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    auto const transports = tt.route_transport_ranges_[r];
    auto const seq = tt.route_location_seq_[r];
    for (auto const t : transports) {
      auto const t_in_r_idx =
          route_transport_idx_t{to_idx(t - transports.from_)};

      auto const sections = tt.transport_to_trip_section_[t];
      rules.resize(sections.size());
      for (auto k = 0U; k != sections.size(); ++k) {
        get_section_rules(tt, sections[k], rules[k]);
      }
      if (utl::all_of(rules, [](auto&& x) { return x.empty(); })) {
        continue;
      }

      for (auto i = stop_idx_t{0U}; i != seq.size(); ++i) {
        auto const stp = stop{seq[i]};
        auto const base = stp.location_idx();

        sig.clear();
        if (i != 0U && stp.out_allowed()) {
          auto const arr_rules = rules[sections.size() == 1U ? 0U : i - 1U];
          add_stop_sides(tt, arr_rules, event_type::kArr, base, sig);
        }
        if (i + 1U != seq.size() && stp.in_allowed()) {
          auto const dep_rules = rules[sections.size() == 1U ? 0U : i];
          add_stop_sides(tt, dep_rules, event_type::kDep, base, sig);
        }

        if (sig.empty()) {
          continue;
        }

        utl::erase_duplicates(sig);

        auto& transport_signatures = utl::get_or_create(
            sigs.transport_signatures_, route_stop{to_split_route(r), i, base},
            [&]() { return transport_sigs_t(transports.size(), kEmptySig); });
        transport_signatures[t_in_r_idx] =
            utl::get_or_create(signature_of, sig, [&]() {
              sigs.sides_.push_back(sig);
              return signature_idx_t{sigs.sides_.size() - 1U};
            });
      }
    }
  }
  return sigs;
}

hash_map<pair<location_idx_t, location_idx_t>, stop_pair> get_stop_pairs(
    timetable const& tt, signatures const& sigs) {
  auto const& locs = tt.locations_;
  auto const rule_times = get_rule_times(tt);
  auto rule_stops = hash_set<location_idx_t>{};
  for (auto const& [rs, _] : sigs.transport_signatures_) {
    rule_stops.insert(rs.stop_);
  }
  auto const get_stop_pair = [&](location_idx_t const from,
                                 location_idx_t const to) {
    auto const fallback = get_walk(tt, from, to);
    auto window = window_t{fallback, fallback + duration_t{1}};
    auto const extend_by = [&](location_idx_t const a, location_idx_t const b) {
      if (auto const r = utl::lookup(rule_times, transfer_pair{a, b})) {
        extend(window, *r);
      }
    };
    for (auto const a : {from, locs.parents_[from]}) {
      for (auto const b : {to, locs.parents_[to]}) {
        extend_by(a, b);
      }
    }
    return stop_pair{fallback, window};
  };
  auto pairs = hash_map<pair<location_idx_t, location_idx_t>, stop_pair>{};
  for (auto const& r : tt.transfer_rules_.rules_) {
    if (!r.is_qualified(event_type::kArr) &&
        !r.is_qualified(event_type::kDep)) {
      continue;
    }
    routing::for_each_meta(
        tt, routing::location_match_mode::kOnlyChildren, r.from_stop_,
        [&](location_idx_t const f) {
          routing::for_each_meta(
              tt, routing::location_match_mode::kOnlyChildren, r.to_stop_,
              [&](location_idx_t const t) {
                if (rule_stops.contains(f) || rule_stops.contains(t)) {
                  utl::get_or_create(pairs, pair{f, t},
                                     [&]() { return get_stop_pair(f, t); });
                }
              });
        });
  }
  return pairs;
}

template <typename Fn>
void for_each_meeting(timetable const& tt,
                      route_idx_t const rx,
                      route_idx_t const ry,
                      events_t const& arrivals,
                      events_t const& departures,
                      window_t const& w,
                      Fn&& fn) {
  auto const traffic_days =
      [&](route_idx_t const r,
          route_transport_idx_t const u) -> bitfield const& {
    return tt.bitfields_[tt.transport_traffic_days_
                             [tt.route_transport_ranges_[r].from_ + to_idx(u)]];
  };
  auto const lo = w.from_;
  auto const hi = w.to_ - duration_t{1};
  for (auto const& [arr, a] : arrivals) {
    auto const& a_days = traffic_days(rx, a);
    for (auto offset =
             std::chrono::floor<date::days>(arr + lo - departures.back().first);
         offset <=
         std::chrono::floor<date::days>(arr + hi - departures.front().first);
         ++offset) {
      auto const arrival = arr - offset;
      auto const n = offset.count();
      for (auto it = std::lower_bound(
               begin(departures), end(departures),
               std::pair{arrival + lo, route_transport_idx_t{0U}});
           it != end(departures) && it->first < arrival + hi; ++it) {
        auto const b = it->second;
        if (rx == ry && a == b) {
          continue;
        }
        auto const days = n >= 0 ? a_days << static_cast<std::size_t>(n)
                                 : a_days >> static_cast<std::size_t>(-n);
        if ((days & traffic_days(ry, b)).none()) {
          continue;
        }
        fn(a, b, std::chrono::duration_cast<duration_t>(it->first - arrival));
      }
    }
  }
}

route_stop_pairs get_route_stop_pairs(timetable const& tt,
                                      signatures const& sigs) {
  auto const& rules = tt.transfer_rules_.rules_;
  auto pairs = route_stop_pairs{};
  pairs.stop_pairs_ = get_stop_pairs(tt, sigs);
  pairs.of_route_.resize(tt.n_routes());
  auto const sigs_of = [&](route_stop const rs) -> transport_sigs_t const* {
    auto const it = sigs.transport_signatures_.find(rs);
    return it == end(sigs.transport_signatures_) ? nullptr : &it->second;
  };
  auto const stop_of = [&](route_stop const rs,
                           transport_sigs_t const* sigs_at) {
    return pairs.get_or_create_route_stop(
        sigs_at != nullptr ? rs : get_base_stop(rs.stop_));
  };
  auto const sig = [](transport_sigs_t const* sigs_at,
                      route_transport_idx_t const u) {
    return sigs_at == nullptr ? kEmptySig : (*sigs_at)[u];
  };

  auto route_stops =
      hash_map<location_idx_t, std::vector<pair<route_idx_t, stop_idx_t>>>{};
  auto const get_route_stops = [&](location_idx_t const l)
      -> std::vector<pair<route_idx_t, stop_idx_t>> const& {
    return utl::get_or_create(route_stops, l, [&]() {
      auto v = std::vector<pair<route_idx_t, stop_idx_t>>{};
      for (auto const r : tt.location_routes_[l]) {
        auto const seq = tt.route_location_seq_[r];
        for (auto i = stop_idx_t{0U}; i != seq.size(); ++i) {
          if (stop{seq[i]}.location_idx() == l) {
            v.push_back(pair{r, i});
          }
        }
      }
      return v;
    });
  };
  auto event_store = std::deque<events_t>{};
  auto events =
      hash_map<pair<pair<route_idx_t, stop_idx_t>, event_type>, events_t*>{};
  auto const get_events = [&](route_idx_t const r, stop_idx_t const i,
                              event_type const ev) -> events_t const& {
    return *utl::get_or_create(events, pair{pair{r, i}, ev}, [&]() {
      auto const transports = tt.route_transport_ranges_[r];
      auto& v = event_store.emplace_back();
      for (auto u = route_transport_idx_t{0U}; u != transports.size(); ++u) {
        v.emplace_back(
            i32_minutes{tt.event_mam(r, transports.from_ + to_idx(u), i, ev)
                            .as_duration()},
            u);
      }
      utl::sort(v);
      return &v;
    });
  };

  auto expected =
      hash_map<pair<signature_idx_t, signature_idx_t>, duration_t>{};
  for (auto const& [stops, sp] : pairs.stop_pairs_) {
    auto const [lx, ly] = stops;
    expected.clear();
    auto const get_expected = [&](signature_idx_t const sa,
                                  signature_idx_t const sb) {
      return utl::get_or_create(expected, pair{sa, sb}, [&]() {
        auto const rule =
            get_change_rule(tt, sigs.sides_[sa], sigs.sides_[sb], lx, ly);
        return rule == transfer_rule_idx_t::invalid()
                   ? sp.fallback_
                   : get_minutes(rules[rule]);
      });
    };
    for (auto const is_arr : {true, false}) {
      auto const l = is_arr ? lx : ly;
      for (auto const [r, i] : get_route_stops(l)) {
        auto const rs = route_stop{to_split_route(r), i, l};
        auto const* sigs_at = sigs_of(rs);
        if (sigs_at == nullptr ||
            !has_event(tt, r, i,
                       is_arr ? event_type::kArr : event_type::kDep)) {
          continue;
        }
        auto const x = is_arr
                           ? stop_of(rs, sigs_at)
                           : pairs.get_or_create_route_stop(get_base_stop(lx));
        auto const y = is_arr
                           ? pairs.get_or_create_route_stop(get_base_stop(ly))
                           : stop_of(rs, sigs_at);
        auto const p = pairs.get_or_create_pair(x, y, sp.window_);
        for (auto t = route_transport_idx_t{0U};
             t != tt.route_transport_ranges_[r].size(); ++t) {
          auto const e = is_arr ? get_expected(sig(sigs_at, t), kEmptySig)
                                : get_expected(kEmptySig, sig(sigs_at, t));
          for (auto const gap : {e, e - duration_t{1}}) {
            pairs.add_transfer(p,
                               transfer{is_arr ? t : kBaseTransport,
                                        is_arr ? kBaseTransport : t, gap, e});
          }
        }
      }
    }
    for (auto const [rx, ix] : get_route_stops(lx)) {
      if (!has_event(tt, rx, ix, event_type::kArr)) {
        continue;
      }
      auto const x = route_stop{to_split_route(rx), ix, lx};
      auto const* xs = sigs_of(x);
      for (auto const [ry, iy] : get_route_stops(ly)) {
        auto const y = route_stop{to_split_route(ry), iy, ly};
        auto const* ys = sigs_of(y);
        if (!has_event(tt, ry, iy, event_type::kDep) ||
            (xs == nullptr && ys == nullptr)) {
          continue;
        }
        auto p = std::optional<route_stop_pair_idx_t>{};
        for_each_meeting(
            tt, rx, ry, get_events(rx, ix, event_type::kArr),
            get_events(ry, iy, event_type::kDep), sp.window_,
            [&](route_transport_idx_t const a, route_transport_idx_t const b,
                duration_t const gap) {
              if (!p.has_value()) {
                p = pairs.get_or_create_pair(stop_of(x, xs), stop_of(y, ys),
                                             sp.window_);
              }
              pairs.add_transfer(
                  *p,
                  transfer{a, b, gap, get_expected(sig(xs, a), sig(ys, b))});
            });
      }
    }
  }
  return pairs;
}

transport_routes split_routes(timetable const& tt, route_stop_pairs& pairs) {
  auto routes = transport_routes{};
  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    routes.base_route_.push_back(r);
  }
  routes.transport_route_.resize(tt.n_routes());
  auto const is_pair_admissible = [&](route_stop_pair_idx_t const p) {
    auto const [a, b] = pairs.pairs_[p];
    return is_admissible(pairs.windows_[p], a == b);
  };
  auto const repartition = [&](route_stop_pair_idx_t const p) {
    auto const [a, b] = pairs.pairs_[p];
    auto const x = pairs.route_stops_[a];
    auto const y = pairs.route_stops_[b];
    auto const window = pairs.stop_pairs_.at(pair{x.stop_, y.stop_}).window_;
    auto const transfers = std::move(pairs.transfers_[p]);
    pairs.transfers_[p] = {};
    pairs.windows_[p] = window;
    pairs.expected_[p].clear();
    auto const route_of = [&](route_stop const& s,
                              route_transport_idx_t const t) {
      return is_base(s) ? s.route_
                        : routes.get_route(routes.base_route_[s.route_], t);
    };
    for (auto const& t : transfers) {
      auto sx = x;
      auto sy = y;
      sx.route_ = route_of(x, t.from_);
      sy.route_ = route_of(y, t.to_);
      if (sx.route_ == x.route_ && sy.route_ == y.route_) {
        pairs.add_transfer(p, t);
      } else {
        auto const a2 = pairs.get_or_create_route_stop(sx);
        auto const b2 = pairs.get_or_create_route_stop(sy);
        pairs.add_transfer(pairs.get_or_create_pair(a2, b2, window), t);
      }
    }
  };
  auto const get_split = [&](route_stop_pair_idx_t const p, bool const is_from)
      -> std::vector<std::vector<route_transport_idx_t>> {
    auto const [a, b] = pairs.pairs_[p];
    auto const& x = pairs.route_stops_[a];
    auto const& y = pairs.route_stops_[b];
    auto const& z = is_from ? x : y;
    if (is_base(z)) {
      return {};
    }
    auto const& sp = pairs.stop_pairs_.at(pair{x.stop_, y.stop_});
    auto const n =
        tt.route_transport_ranges_[routes.base_route_[z.route_]].size();
    auto w = vector_map<route_transport_idx_t, std::optional<window_t>>(n);
    for (auto const& t : pairs.transfers_[p]) {
      auto& tw = w[is_from ? t.from_ : t.to_];
      if (!tw.has_value()) {
        tw = sp.window_;
      }
      narrow(*tw, t.gap_, t.gap_ >= t.expected_);
    }
    auto const max = a == b ? kMaxTransferTime : kMaxFootpath;
    auto const items = [&]() {
      auto v = std::vector<std::pair<route_transport_idx_t, window_t>>{};
      for (auto i = route_transport_idx_t{0U}; i != n; ++i) {
        if (w[i].has_value()) {
          v.emplace_back(i, *w[i]);
        }
      }
      utl::sort(v, [](auto const& l, auto const& r) {
        return std::pair{l.second.to_, l.first} <
               std::pair{r.second.to_, r.first};
      });
      return v;
    }();
    auto groups =
        std::vector<std::pair<window_t, std::vector<route_transport_idx_t>>>{};
    for (auto const& [t, tw] : items) {
      auto const both = groups.empty() ? tw : groups.back().first.intersect(tw);
      if (!groups.empty() &&
          pick_duration(both, sp.fallback_, max).has_value()) {
        groups.back().first = both;
        groups.back().second.push_back(t);
      } else {
        groups.push_back({tw, {t}});
      }
    }
    if (groups.size() < 2U) {
      return {};
    }
    auto moved =
        utl::to_vec(groups, [](auto& g) { return std::move(g.second); });
    moved.erase(std::ranges::max_element(
        moved, {}, [](auto const& g) { return g.size(); }));
    return moved;
  };
  auto work = std::vector<route_stop_pair_idx_t>{};
  for (auto p = route_stop_pair_idx_t{0U}; p != pairs.pairs_.size(); ++p) {
    if (!is_pair_admissible(p)) {
      work.push_back(p);
    }
  }
  auto n_split = 0U;
  auto n_unresolved = 0U;
  while (!work.empty()) {
    auto const p = work.back();
    work.pop_back();
    if (is_pair_admissible(p)) {
      continue;
    }
    auto const from = get_split(p, true);
    auto const to = get_split(p, false);
    if (from.empty() && to.empty()) {
      ++n_unresolved;
      continue;
    }
    auto const use_from =
        !from.empty() && (to.empty() || from.size() <= to.size());
    auto const [a, b] = pairs.pairs_[p];
    auto const r = pairs.route_stops_[use_from ? a : b].route_;
    auto const base_route = routes.base_route_[r];
    auto& route = routes.transport_route_[base_route];
    if (route.empty()) {
      route = vector_map<route_transport_idx_t, split_route_idx_t>(
          tt.route_transport_ranges_[base_route].size(),
          to_split_route(base_route));
    }
    auto const first_new_route = split_route_idx_t{routes.base_route_.size()};
    for (auto const& g : use_from ? from : to) {
      auto const new_route = split_route_idx_t{routes.base_route_.size()};
      routes.base_route_.push_back(base_route);
      pairs.of_route_.emplace_back();
      for (auto const t : g) {
        route[t] = new_route;
      }
    }
    auto const route_pairs = pairs.of_route_[r];
    for (auto const q : route_pairs) {
      repartition(q);
    }
    auto const recheck = [&](split_route_idx_t const q) {
      for (auto const pq : pairs.of_route_[q]) {
        if (!is_pair_admissible(pq)) {
          work.push_back(pq);
        }
      }
    };
    recheck(r);
    for (auto q = first_new_route;
         q != split_route_idx_t{routes.base_route_.size()}; ++q) {
      recheck(q);
    }
    work.push_back(p);
    ++n_split;
  }
  log(log_lvl::info, "loader.virtual_locations",
      "{} route stops, {} pairs, {} splits, {} unresolved",
      pairs.route_stops_.size(), pairs.pairs_.size(), n_split, n_unresolved);
  return routes;
}

route_stop_locations merge_locations(route_stop_pairs const& pairs) {
  auto locations = route_stop_locations{};
  auto& location = locations.location_;
  auto& lps = locations.pairs_;
  lps.resize(pairs.route_stops_.size());
  for (auto s = route_stop_idx_t{0U}; s != pairs.route_stops_.size(); ++s) {
    location.push_back(s);
  }
  for (auto p = route_stop_pair_idx_t{0U}; p != pairs.pairs_.size(); ++p) {
    auto const [a, b] = pairs.pairs_[p];
    lps[a].push_back({b, pairs.windows_[p], pairs.expected_[p]});
  }
  auto const by_to = [](location_pair const& a, location_pair const& b) {
    return a.to_ < b.to_;
  };
  for (auto& v : lps) {
    utl::sort(v, by_to);
  }
  auto partners = std::vector<route_stop_idx_t>{};
  auto folds = std::vector<pair<route_stop_pair_t, route_stop_pair_t>>{};
  auto merged = std::vector<pair<route_stop_idx_t, location_pair>>{};
  auto const locations_of = [&](std::vector<route_stop_idx_t> const& v) {
    partners.clear();
    for (auto const x : v) {
      partners.push_back(location[x]);
    }
    utl::erase_duplicates(partners);
  };
  auto const join = [&](route_stop_idx_t const s, route_stop_idx_t const l) {
    folds.clear();
    locations_of(pairs.out_[s]);
    for (auto const q : partners) {
      folds.push_back({{l, q == s ? l : q}, {s, q}});
    }
    locations_of(pairs.in_[s]);
    for (auto const q : partners) {
      if (q != s) {
        folds.push_back({{q, l}, {q, s}});
      }
    }
    utl::sort(folds);
    merged.clear();
    for (auto it = begin(folds); it != end(folds);) {
      auto const into = it->first;
      auto const* e = locations.find(into.first, into.second);
      auto m =
          location_pair{into.second, e != nullptr ? e->window_ : kAnyDuration,
                        e != nullptr ? e->expected_ : histogram_t{}};
      for (; it != end(folds) && it->first == into; ++it) {
        auto const* o = locations.find(it->second.first, it->second.second);
        utl::verify(o != nullptr, "location pair missing");
        m.window_ = m.window_.intersect(o->window_);
        for (auto const& [d, n] : o->expected_) {
          count(m.expected_, d, n);
        }
      }
      auto const [a, b] = into;
      if (pairs.is_base(a) && pairs.is_base(b)
              ? !m.window_.contains(pairs.stop_pairs_
                                        .at(pair{pairs.route_stops_[a].stop_,
                                                 pairs.route_stops_[b].stop_})
                                        .fallback_)
              : !is_admissible(m.window_, a == b)) {
        return false;
      }
      merged.push_back({into.first, std::move(m)});
    }
    for (auto it = begin(merged); it != end(merged);) {
      auto const from = it->first;
      auto& v = lps[from];
      auto const n = static_cast<std::ptrdiff_t>(v.size());
      for (; it != end(merged) && it->first == from; ++it) {
        auto const pos =
            std::lower_bound(begin(v), begin(v) + n, it->second, by_to);
        if (pos != begin(v) + n && pos->to_ == it->second.to_) {
          *pos = std::move(it->second);
        } else {
          v.push_back(std::move(it->second));
        }
      }
      std::inplace_merge(begin(v), begin(v) + n, end(v), by_to);
    }
    location[s] = l;
    return true;
  };
  auto at_stop = hash_map<location_idx_t, std::vector<route_stop_idx_t>>{};
  auto n_joined = 0U;
  auto n_locations = 0U;
  for (auto s = route_stop_idx_t{0U}; s != pairs.route_stops_.size(); ++s) {
    if (pairs.is_base(s)) {
      continue;
    }
    auto const st = pairs.route_stops_[s].stop_;
    auto& candidates = at_stop[st];
    if (candidates.empty()) {
      candidates.push_back(pairs.route_stop_of_.at(get_base_stop(st)));
    }
    if (utl::any_of(candidates,
                    [&](route_stop_idx_t const l) { return join(s, l); })) {
      ++n_joined;
    } else {
      candidates.push_back(s);
      ++n_locations;
    }
  }
  log(log_lvl::info, "loader.virtual_locations", "{} joined, {} locations",
      n_joined, n_locations);
  return locations;
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

duration_t to_duration(duration_t const d) {
  return d >= kNever ? footpath::kMaxDuration : d;
}

vector_map<route_stop_idx_t, location_idx_t> add_virts(
    timetable& tt,
    route_stop_pairs const& pairs,
    route_stop_locations const& locations) {
  auto virts = vector_map<route_stop_idx_t, location_idx_t>(
      pairs.route_stops_.size(), location_idx_t::invalid());
  auto n_virts = std::size_t{0U};
  for (auto s = route_stop_idx_t{0U}; s != pairs.route_stops_.size(); ++s) {
    if (locations.location_[s] != s) {
      continue;
    }
    auto const st = pairs.route_stops_[s].stop_;
    if (pairs.is_base(s)) {
      virts[s] = st;
      continue;
    }
    auto const fallback = get_walk(tt, st, st);
    auto const* self = locations.find(s, s);
    auto const d = self == nullptr
                       ? std::optional{fallback}
                       : pick_duration(self->window_, fallback,
                                       kMaxTransferTime, self->expected_);
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
  log(log_lvl::info, "loader.virtual_locations", "{} virtual locations",
      n_virts);
  return virts;
}

void add_footpaths(timetable const& tt,
                   route_stop_pairs const& pairs,
                   route_stop_locations const& locations,
                   vector_map<route_stop_idx_t, location_idx_t> const& virts,
                   mutable_fws_multimap<location_idx_t, footpath>& durations) {
  durations[location_idx_t{tt.n_locations() - 1U}];
  auto n_footpaths = std::size_t{0U};
  for (auto a = route_stop_idx_t{0U}; a != locations.pairs_.size(); ++a) {
    if (locations.location_[a] != a) {
      continue;
    }
    for (auto const& m : locations.pairs_[a]) {
      auto const c = m.to_;
      if (a == c || locations.location_[c] != c ||
          (pairs.is_base(a) && pairs.is_base(c))) {
        continue;
      }
      auto const fallback = pairs.stop_pairs_
                                .at(pair{pairs.route_stops_[a].stop_,
                                         pairs.route_stops_[c].stop_})
                                .fallback_;
      auto const d =
          pick_duration(m.window_, fallback, kMaxFootpath, m.expected_);
      utl::verify(d.has_value(), "location pair without duration");
      if (*d == fallback) {
        continue;
      }
      durations[virts[a]].push_back(footpath{virts[c], to_duration(*d)});
      ++n_footpaths;
    }
  }
  log(log_lvl::info, "loader.virtual_locations", "{} footpaths", n_footpaths);
}

void write_routes(timetable& tt,
                  route_stop_pairs const& pairs,
                  transport_routes const& routes,
                  route_stop_locations const& locations,
                  vector_map<route_stop_idx_t, location_idx_t> const& virts,
                  shapes_storage* shapes) {
  auto origin = vector_map<route_idx_t, route_idx_t>{};
  auto route_transports =
      vector_map<route_idx_t, std::vector<transport_idx_t>>{};
  auto const n_routes = tt.n_routes();
  for (auto r = route_idx_t{0U}; r != n_routes; ++r) {
    auto const transports = tt.route_transport_ranges_[r];
    origin.push_back(r);
    route_transports.emplace_back(begin(transports), end(transports));
  }
  auto const base_route_routes = [&]() {
    auto m = hash_map<route_idx_t, std::vector<split_route_idx_t>>{};
    for (auto r = split_route_idx_t{n_routes};
         r != split_route_idx_t{routes.base_route_.size()}; ++r) {
      m[routes.base_route_[r]].push_back(r);
    }
    return m;
  }();
  auto seqs = std::vector<stop_seq_t>{};
  auto n_regrouped = std::size_t{0U};
  for (auto r = route_idx_t{0U}; r != n_routes; ++r) {
    if (pairs.of_route_[to_split_route(r)].empty()) {
      continue;
    }
    auto const transports = tt.route_transport_ranges_[r];
    auto const seq = tt.route_location_seq_[r];
    auto const routes_of_base = [&]() {
      auto v = std::vector{to_split_route(r)};
      if (auto const it = base_route_routes.find(r);
          it != end(base_route_routes)) {
        v.insert(end(v), begin(it->second), end(it->second));
      }
      return v;
    }();
    seqs.assign(routes_of_base.size(), stop_seq_t{seq.begin(), seq.end()});
    for (auto k = 0U; k != routes_of_base.size(); ++k) {
      for (auto i = stop_idx_t{0U}; i != seq.size(); ++i) {
        auto const s = pairs.route_stop_of_.find(
            route_stop{routes_of_base[k], i, stop{seq[i]}.location_idx()});
        if (s != end(pairs.route_stop_of_)) {
          seqs[k][i] = stop{seq[i]}
                           .with_location(virts[locations.location_[s->second]])
                           .value();
        }
      }
    }
    auto groups =
        std::vector<std::pair<stop_seq_t, std::vector<transport_idx_t>>>{};
    for (auto t = route_transport_idx_t{0U}; t != transports.size(); ++t) {
      auto const route = routes.get_route(r, t);
      auto const& s = seqs[static_cast<std::size_t>(
          utl::find(routes_of_base, route) - begin(routes_of_base))];
      auto g = utl::find_if(
          groups, [&](auto const& group) { return group.first == s; });
      if (g == end(groups)) {
        g = groups.insert(end(groups), {s, {}});
      }
      g->second.push_back(transports.from_ + to_idx(t));
    }
    n_regrouped += routes_of_base.size() - groups.size();
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
      "{} routes -> {}, {} regrouped", n_routes, tt.n_routes(), n_regrouped);
}

bool add_virtual_locations(
    timetable& tt,
    shapes_storage* shapes,
    mutable_fws_multimap<location_idx_t, footpath>& durations) {
  if (tt.transfer_rules_.rules_.empty()) {
    return false;
  }
  auto const timer = scoped_timer{"loader.virtual_locations"};
  auto const sigs = get_signatures(tt);
  auto pairs = get_route_stop_pairs(tt, sigs);
  auto const routes = split_routes(tt, pairs);
  auto const locations = merge_locations(pairs);
  auto const n_locations = tt.n_locations();
  auto const virts = add_virts(tt, pairs, locations);
  if (tt.n_locations() == n_locations) {
    return false;
  }
  add_footpaths(tt, pairs, locations, virts, durations);
  write_routes(tt, pairs, routes, locations, virts, shapes);
  return true;
}

}  // namespace nigiri::loader
