#include "nigiri/loader/virtual_locations.h"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "utl/get_or_create.h"
#include "utl/helpers/algorithm.h"
#include "utl/lookup.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "nigiri/loader/register.h"
#include "nigiri/loader/transfer_rules.h"
#include "nigiri/loader/virt/signatures.h"
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
    cista::strong<std::uint16_t, struct route_transport_>;
using route_stop_idx_t = cista::strong<std::uint32_t, struct route_stop_>;
using route_stop_pair_idx_t =
    cista::strong<std::uint32_t, struct route_stop_pair_>;
using stop_seq_t = std::vector<stop::value_type>;
using route_stop_pair_t = pair<route_stop_idx_t, route_stop_idx_t>;
using window_t = interval<duration_t>;

constexpr auto const kBaseTransport = route_transport_idx_t{0U};
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
  route_idx_t route_;
  stop_idx_t idx_;
  location_idx_t stop_;
  signature_idx_t sig_{signature_idx_t::invalid()};
};

struct transfer_durations {
  duration_t fallback_;
  window_t window_;
};

using transfer_durations_t = hash_map<transfer_pair, transfer_durations>;

struct location_pair {
  route_stop_idx_t to_;
  window_t window_;
};

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
    auto const w = window_t{m, m + duration_t{1}};
    auto const [it, is_new] =
        rule_times.emplace(transfer_pair{rule.from_stop_, rule.to_stop_}, w);
    if (!is_new) {
      extend(it->second, w);
    }
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
                                        duration_t const max) {
  if (w.contains(fallback)) {
    return fallback;
  }
  if (w.contains(kNever)) {
    return kNever;
  }
  return w.from_ < std::min(w.to_, max + duration_t{1}) ? std::optional{w.from_}
                                                        : std::nullopt;
}

bool is_admissible(window_t const& w, bool const is_self) {
  return pick_duration(w, kNever, is_self ? kMaxTransferTime : kMaxFootpath)
      .has_value();
}

bool is_base(route_stop const& s) {
  return s.route_ == route_idx_t::invalid() &&
         s.sig_ == signature_idx_t::invalid();
}

bool has_route(route_stop const& s) {
  return s.route_ != route_idx_t::invalid();
}

route_stop get_base_stop(location_idx_t const l) {
  return {route_idx_t::invalid(), stop_idx_t{0U}, l};
}

struct transport_routes {
  route_idx_t get_route(route_idx_t const base_route,
                        route_transport_idx_t const t) const {
    auto const& route = transport_route_[base_route];
    return route.empty() ? base_route : route[t];
  }

  vector_map<route_idx_t, route_idx_t> base_route_;
  vector_map<route_idx_t, vector_map<route_transport_idx_t, route_idx_t>>
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
    return idx;
  }

  route_stop_idx_t get_or_create_route_stop(route_stop const& rs,
                                            signature_idx_t const sig) {
    if (auto const it = route_stop_of_.find(rs); it != end(route_stop_of_)) {
      return it->second;
    }
    auto const s = get_or_create_route_stop(
        route_stop{route_idx_t::invalid(), stop_idx_t{0U}, rs.stop_, sig});
    route_stop_of_.emplace(rs, s);
    return s;
  }

  route_stop_pair_idx_t get_or_create_pair(route_stop_idx_t const a,
                                           route_stop_idx_t const b,
                                           window_t const& window) {
    return utl::get_or_create(pair_of_, route_stop_pair_t{a, b}, [&]() {
      auto const p = route_stop_pair_idx_t{pairs_.size()};
      pairs_.push_back(route_stop_pair_t{a, b});
      transfers_.emplace_back();
      windows_.push_back(window);
      auto const& x = route_stops_[a];
      auto const& y = route_stops_[b];
      if (has_route(x)) {
        of_route_[x.route_].push_back(p);
      }
      if (has_route(y) && (!has_route(x) || x.route_ != y.route_)) {
        of_route_[y.route_].push_back(p);
      }
      return p;
    });
  }

  void add_transfer(route_stop_pair_idx_t const p, transfer const& t) {
    auto const [a, b] = pairs_[p];
    if (has_route(route_stops_[a]) || has_route(route_stops_[b])) {
      transfers_[p].push_back(t);
    }
    narrow(windows_[p], t.gap_, t.gap_ >= t.expected_);
  }

  vector_map<route_stop_idx_t, route_stop> route_stops_;
  hash_map<route_stop, route_stop_idx_t> route_stop_of_;
  vector_map<route_stop_pair_idx_t, route_stop_pair_t> pairs_;
  hash_map<route_stop_pair_t, route_stop_pair_idx_t> pair_of_;
  vector_map<route_stop_pair_idx_t, std::vector<transfer>> transfers_;
  vector_map<route_stop_pair_idx_t, window_t> windows_;
  vector_map<route_idx_t, std::vector<route_stop_pair_idx_t>> of_route_;
};

template <typename T>
auto* find_pair(T& location_pairs, route_stop_idx_t const to) {
  auto const it =
      std::ranges::lower_bound(location_pairs, to, {}, &location_pair::to_);
  return it != end(location_pairs) && it->to_ == to ? &*it : nullptr;
}

struct route_stop_locations {
  vector_map<route_stop_idx_t, route_stop_idx_t> location_;
  vector_map<route_stop_idx_t, std::vector<location_pair>> pairs_;
};

duration_t get_walk(timetable const& tt,
                    location_idx_t const from,
                    location_idx_t const to) {
  return routing::shortest_transfer(tt, nullptr, kDefaultProfile, from, to)
      .value_or(kNever);
}

transfer_durations_t get_transfer_durations(timetable const& tt,
                                            signatures const& sigs) {
  auto const& locs = tt.locations_;
  auto const rule_times = get_rule_times(tt);
  auto rule_stops = hash_set<location_idx_t>{};
  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    if (sigs.route_stop_sigs_[r].size() == 0U) {
      continue;
    }
    auto const seq = tt.route_location_seq_[r];
    for (auto i = 0U; i != seq.size(); ++i) {
      if (!sigs.route_stop_sigs_[r][i].empty()) {
        rule_stops.insert(stop{seq[i]}.location_idx());
      }
    }
  }

  auto durations_of = transfer_durations_t{};
  auto const add_pair = [&](location_idx_t const from,
                            location_idx_t const to) {
    if (!rule_stops.contains(from) && !rule_stops.contains(to)) {
      return;
    }
    utl::get_or_create(durations_of, transfer_pair{from, to}, [&]() {
      auto const fallback = get_walk(tt, from, to);
      auto window = window_t{fallback, fallback + duration_t{1}};
      for (auto const a : {from, locs.parents_[from]}) {
        for (auto const b : {to, locs.parents_[to]}) {
          if (auto const w = utl::lookup(rule_times, transfer_pair{a, b})) {
            extend(window, *w);
          }
        }
      }
      return transfer_durations{fallback, window};
    });
  };
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
              [&](location_idx_t const t) { add_pair(f, t); });
        });
  }
  return durations_of;
}

template <typename Fn>
void for_each_meeting(timetable const& tt,
                      route_idx_t const rx,
                      route_idx_t const ry,
                      std::span<delta const> arrivals,
                      std::span<delta const> departures,
                      window_t const& w,
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
  auto const lo = w.from_;
  auto const hi = w.to_ - duration_t{1};
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

struct location_route_stop {
  route_idx_t route_;
  stop_idx_t idx_;
  std::span<signature_idx_t const> sigs_;
  bool has_arr_;
  bool has_dep_;
  route_stop_idx_t route_stop_{route_stop_idx_t::invalid()};
};

using location_route_stops_t =
    hash_map<location_idx_t, std::vector<location_route_stop>>;

location_route_stops_t get_location_route_stops(
    timetable const& tt,
    signatures const& sigs,
    transfer_durations_t const& durations_of) {
  auto route_stops = location_route_stops_t{};
  for (auto const& [stops, _] : durations_of) {
    for (auto const l : {stops.from_, stops.to_}) {
      utl::get_or_create(route_stops, l, [&]() {
        auto v = std::vector<location_route_stop>{};
        for (auto const r : tt.location_routes_[l]) {
          auto const seq = tt.route_location_seq_[r];
          for (auto i = stop_idx_t{0U}; i != seq.size(); ++i) {
            if (stop{seq[i]}.location_idx() != l) {
              continue;
            }
            auto stop_sigs = std::span<signature_idx_t const>{};
            if (sigs.route_stop_sigs_[r].size() != 0U) {
              auto const b = sigs.route_stop_sigs_[r][i];
              stop_sigs = {b.begin(), b.end()};
            }
            v.push_back(location_route_stop{
                .route_ = r,
                .idx_ = i,
                .sigs_ = stop_sigs,
                .has_arr_ = has_event(tt, r, i, event_type::kArr),
                .has_dep_ = has_event(tt, r, i, event_type::kDep)});
          }
        }
        return v;
      });
    }
  }
  return route_stops;
}

signature_idx_t get_signature(location_route_stop const& s,
                              route_transport_idx_t const u) {
  return s.sigs_.empty()        ? kEmptySig
         : s.sigs_.size() == 1U ? s.sigs_.front()
                                : s.sigs_[to_idx(u)];
}

route_stop_idx_t get_route_stop(route_stop_pairs& pairs,
                                location_route_stop& s,
                                location_idx_t const l) {
  if (s.route_stop_ == route_stop_idx_t::invalid()) {
    auto const rs = route_stop{s.route_, s.idx_, l};
    s.route_stop_ = s.sigs_.size() == 1U
                        ? pairs.get_or_create_route_stop(rs, s.sigs_.front())
                    : !s.sigs_.empty()
                        ? pairs.get_or_create_route_stop(rs)
                        : pairs.get_or_create_route_stop(get_base_stop(l));
  }
  return s.route_stop_;
}

struct expected_durations {
  void reset(transfer_pair const stops, duration_t const fallback) {
    stops_ = stops;
    fallback_ = fallback;
    cache_.clear();
  }

  duration_t operator()(signature_idx_t const from, signature_idx_t const to) {
    return utl::get_or_create(cache_, pair{from, to}, [&]() {
      auto const rule = get_change_rule(
          tt_, sigs_.rules_[from], sigs_.rules_[to], stops_.from_, stops_.to_);
      return rule == transfer_rule_idx_t::invalid()
                 ? fallback_
                 : get_minutes(tt_.transfer_rules_.rules_[rule]);
    });
  }

  timetable const& tt_;
  signatures const& sigs_;
  transfer_pair stops_{};
  duration_t fallback_{};
  hash_map<pair<signature_idx_t, signature_idx_t>, duration_t> cache_{};
};

void add_base_transfers(timetable const& tt,
                        transfer_pair const stops,
                        transfer_durations const& pair_durations,
                        std::vector<location_route_stop>& at_from,
                        std::vector<location_route_stop>& at_to,
                        expected_durations& expected,
                        route_stop_pairs& pairs) {
  for (auto const is_arr : {true, false}) {
    auto const l = is_arr ? stops.from_ : stops.to_;
    for (auto& s : is_arr ? at_from : at_to) {
      if (s.sigs_.empty() || !(is_arr ? s.has_arr_ : s.has_dep_)) {
        continue;
      }
      auto const x =
          is_arr ? get_route_stop(pairs, s, l)
                 : pairs.get_or_create_route_stop(get_base_stop(stops.from_));
      auto const y =
          is_arr ? pairs.get_or_create_route_stop(get_base_stop(stops.to_))
                 : get_route_stop(pairs, s, l);
      auto const p = pairs.get_or_create_pair(x, y, pair_durations.window_);
      auto const n = static_cast<std::uint32_t>(
          tt.route_transport_ranges_[s.route_].size());
      auto const is_single = s.sigs_.size() == 1U;
      for (auto t = route_transport_idx_t{0U};
           t != route_transport_idx_t{is_single ? 1U : n}; ++t) {
        auto const e = is_arr ? expected(get_signature(s, t), kEmptySig)
                              : expected(kEmptySig, get_signature(s, t));
        for (auto const gap : {e, e - duration_t{1}}) {
          pairs.add_transfer(p, transfer{is_arr ? t : kBaseTransport,
                                         is_arr ? kBaseTransport : t, gap, e});
        }
      }
    }
  }
}

void add_meeting_transfers(timetable const& tt,
                           transfer_pair const stops,
                           transfer_durations const& pair_durations,
                           std::vector<location_route_stop>& at_from,
                           std::vector<location_route_stop>& at_to,
                           expected_durations& expected,
                           route_stop_pairs& pairs) {
  for (auto& x : at_from) {
    if (!x.has_arr_) {
      continue;
    }
    for (auto& y : at_to) {
      if (!y.has_dep_ || (x.sigs_.empty() && y.sigs_.empty())) {
        continue;
      }
      auto const has_transport_sigs =
          x.sigs_.size() > 1U || y.sigs_.size() > 1U;
      auto single_expected = std::optional<duration_t>{};
      auto p = route_stop_pair_idx_t::invalid();
      for_each_meeting(
          tt, x.route_, y.route_,
          tt.event_times_at_stop(x.route_, x.idx_, event_type::kArr),
          tt.event_times_at_stop(y.route_, y.idx_, event_type::kDep),
          pair_durations.window_,
          [&](route_transport_idx_t const a, route_transport_idx_t const b,
              duration_t const gap) {
            if (p == route_stop_pair_idx_t::invalid()) {
              auto const from = get_route_stop(pairs, x, stops.from_);
              auto const to = get_route_stop(pairs, y, stops.to_);
              p = pairs.get_or_create_pair(from, to, pair_durations.window_);
            }
            if (!has_transport_sigs && !single_expected.has_value()) {
              single_expected =
                  expected(get_signature(x, a), get_signature(y, b));
            }
            auto const e = has_transport_sigs ? expected(get_signature(x, a),
                                                         get_signature(y, b))
                                              : *single_expected;
            pairs.add_transfer(p, transfer{a, b, gap, e});
          });
    }
  }
}

transport_routes split_routes(timetable const& tt,
                              transfer_durations_t const& durations_of,
                              route_stop_pairs& pairs) {
  auto routes = transport_routes{};
  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    routes.base_route_.push_back(r);
  }
  routes.transport_route_.resize(tt.n_routes());
  auto const is_pair_admissible = [&](route_stop_pair_idx_t const p) {
    auto const [a, b] = pairs.pairs_[p];
    return is_admissible(pairs.windows_[p], a == b);
  };
  auto targets = std::vector<route_stop_pair_idx_t>{};
  auto const repartition = [&](route_stop_pair_idx_t const p) {
    auto const [a, b] = pairs.pairs_[p];
    auto const x = pairs.route_stops_[a];
    auto const y = pairs.route_stops_[b];
    auto const route_of = [&](route_stop const& s,
                              route_transport_idx_t const t) {
      return !has_route(s) ? s.route_
                           : routes.get_route(routes.base_route_[s.route_], t);
    };
    auto const window =
        durations_of.at(transfer_pair{x.stop_, y.stop_}).window_;
    targets.clear();
    auto const get_target = [&](route_idx_t const rx, route_idx_t const ry) {
      for (auto const q : targets) {
        auto const [qa, qb] = pairs.pairs_[q];
        if (pairs.route_stops_[qa].route_ == rx &&
            pairs.route_stops_[qb].route_ == ry) {
          return q;
        }
      }
      auto const a2 = pairs.get_or_create_route_stop(
          route_stop{rx, x.idx_, x.stop_, x.sig_});
      auto const b2 = pairs.get_or_create_route_stop(
          route_stop{ry, y.idx_, y.stop_, y.sig_});
      auto const q = pairs.get_or_create_pair(a2, b2, window);
      targets.push_back(q);
      return q;
    };
    auto transfers = std::move(pairs.transfers_[p]);
    auto kept = begin(transfers);
    for (auto const& t : transfers) {
      auto const rx = route_of(x, t.from_);
      auto const ry = route_of(y, t.to_);
      if (rx == x.route_ && ry == y.route_) {
        *kept++ = t;
      } else {
        pairs.add_transfer(get_target(rx, ry), t);
      }
    }
    if (kept == end(transfers)) {
      pairs.transfers_[p] = std::move(transfers);
      return;
    }
    auto& w = pairs.windows_[p];
    w = window;
    for (auto it = begin(transfers); it != kept; ++it) {
      narrow(w, it->gap_, it->gap_ >= it->expected_);
    }
    pairs.transfers_[p] = std::vector<transfer>(begin(transfers), kept);
  };
  auto const get_groups = [&](route_stop_pair_idx_t const p, bool const is_from)
      -> std::vector<std::vector<route_transport_idx_t>> {
    auto const [a, b] = pairs.pairs_[p];
    auto const& x = pairs.route_stops_[a];
    auto const& y = pairs.route_stops_[b];
    auto const& z = is_from ? x : y;
    if (!has_route(z)) {
      return {};
    }
    auto const& pair_durations =
        durations_of.at(transfer_pair{x.stop_, y.stop_});
    auto const n = route_transport_idx_t{
        tt.route_transport_ranges_[routes.base_route_[z.route_]].size()};
    auto w =
        vector_map<route_transport_idx_t, std::optional<window_t>>(to_idx(n));
    for (auto const& t : pairs.transfers_[p]) {
      auto& tw = w[is_from ? t.from_ : t.to_];
      if (!tw.has_value()) {
        tw = pair_durations.window_;
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
          pick_duration(both, pair_durations.fallback_, max).has_value()) {
        groups.back().first = both;
        groups.back().second.push_back(t);
      } else {
        groups.push_back({tw, {t}});
      }
    }
    if (groups.size() < 2U) {
      return {};
    }
    return utl::to_vec(groups, [](auto& g) { return std::move(g.second); });
  };
  auto pair_groups =
      std::vector<std::vector<std::vector<route_transport_idx_t>>>{};
  auto groups_of =
      vector_map<route_transport_idx_t,
                 std::vector<pair<std::uint32_t, std::uint32_t>>>{};
  auto route_groups = std::vector<std::vector<std::uint32_t>>{};
  auto new_route_of = vector_map<route_transport_idx_t, std::uint32_t>{};
  auto route_sizes = std::vector<std::uint32_t>{};
  constexpr auto const kNoGroup = std::numeric_limits<std::uint32_t>::max();
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
    auto const from = get_groups(p, true);
    auto const to = get_groups(p, false);
    if (from.empty() && to.empty()) {
      ++n_unresolved;
      continue;
    }
    auto const use_from =
        !from.empty() && (to.empty() || from.size() <= to.size());
    auto const [a, b] = pairs.pairs_[p];
    auto const r = pairs.route_stops_[use_from ? a : b].route_;
    auto const base_route = routes.base_route_[r];
    auto const n_transports =
        route_transport_idx_t{tt.route_transport_ranges_[base_route].size()};

    pair_groups.clear();
    for (auto const q : pairs.of_route_[r]) {
      if (is_pair_admissible(q)) {
        continue;
      }
      auto const [qa, qb] = pairs.pairs_[q];
      for (auto const is_from : {true, false}) {
        if (pairs.route_stops_[is_from ? qa : qb].route_ == r) {
          auto g = get_groups(q, is_from);
          if (!g.empty()) {
            pair_groups.push_back(std::move(g));
          }
        }
      }
    }

    groups_of.clear();
    groups_of.resize(to_idx(n_transports));
    for (auto i = 0U; i != pair_groups.size(); ++i) {
      for (auto g = 0U; g != pair_groups[i].size(); ++g) {
        for (auto const t : pair_groups[i][g]) {
          groups_of[t].push_back(pair{i, g});
        }
      }
    }
    route_groups.clear();
    route_sizes.clear();
    new_route_of.clear();
    new_route_of.resize(to_idx(n_transports), kNoGroup);
    for (auto t = route_transport_idx_t{0U}; t != n_transports; ++t) {
      if (groups_of[t].empty() || routes.get_route(base_route, t) != r) {
        continue;
      }
      auto const fits = [&](std::vector<std::uint32_t> const& groups) {
        return utl::all_of(groups_of[t], [&](auto const& x) {
          return groups[x.first] == kNoGroup || groups[x.first] == x.second;
        });
      };
      auto const it = utl::find_if(route_groups, fits);
      auto const k = static_cast<std::uint32_t>(it - begin(route_groups));
      if (it == end(route_groups)) {
        route_groups.emplace_back(pair_groups.size(), kNoGroup);
        route_sizes.push_back(0U);
      }
      for (auto const& [i, g] : groups_of[t]) {
        route_groups[k][i] = g;
      }
      new_route_of[t] = k;
      ++route_sizes[k];
    }
    auto const kept_route = static_cast<std::uint32_t>(
        std::ranges::max_element(route_sizes) - begin(route_sizes));

    auto& route = routes.transport_route_[base_route];
    if (route.empty()) {
      route = vector_map<route_transport_idx_t, route_idx_t>(
          to_idx(n_transports), base_route);
    }
    auto const first_new_route = route_idx_t{routes.base_route_.size()};
    for (auto k = 0U; k != route_sizes.size(); ++k) {
      if (k == kept_route) {
        continue;
      }
      auto const new_route = route_idx_t{routes.base_route_.size()};
      routes.base_route_.push_back(base_route);
      pairs.of_route_.emplace_back();
      for (auto t = route_transport_idx_t{0U}; t != n_transports; ++t) {
        if (new_route_of[t] == k) {
          route[t] = new_route;
        }
      }
    }
    auto const route_pairs = pairs.of_route_[r];
    for (auto const q : route_pairs) {
      repartition(q);
    }
    auto const recheck = [&](route_idx_t const q) {
      for (auto const pq : pairs.of_route_[q]) {
        if (!is_pair_admissible(pq)) {
          work.push_back(pq);
        }
      }
    };
    recheck(r);
    for (auto q = first_new_route; q != route_idx_t{routes.base_route_.size()};
         ++q) {
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

route_stop_locations merge_locations(transfer_durations_t const& durations_of,
                                     route_stop_pairs const& pairs) {
  auto locations = route_stop_locations{};
  auto& location = locations.location_;
  auto& lps = locations.pairs_;
  lps.resize(pairs.route_stops_.size());
  for (auto s = route_stop_idx_t{0U}; s != pairs.route_stops_.size(); ++s) {
    location.push_back(s);
  }
  for (auto p = route_stop_pair_idx_t{0U}; p != pairs.pairs_.size(); ++p) {
    auto const [a, b] = pairs.pairs_[p];
    lps[a].push_back({b, pairs.windows_[p]});
  }
  for (auto& v : lps) {
    std::ranges::sort(v, {}, &location_pair::to_);
  }
  auto const get_partners = [&](bool const is_out) {
    auto n = vector_map<route_stop_idx_t, std::uint32_t>(
        pairs.route_stops_.size(), 0U);
    for (auto const [a, b] : pairs.pairs_) {
      ++n[is_out ? a : b];
    }
    auto partners = vecvec<route_stop_idx_t, route_stop_idx_t>{};
    for (auto const k : n) {
      partners.add_back_sized(k);
    }
    std::ranges::fill(n, 0U);
    for (auto const [a, b] : pairs.pairs_) {
      auto const x = is_out ? a : b;
      partners[x][n[x]++] = is_out ? b : a;
    }
    return partners;
  };
  auto const out = get_partners(true);
  auto const in = get_partners(false);
  auto out_partners = std::vector<route_stop_idx_t>{};
  auto in_partners = std::vector<route_stop_idx_t>{};
  auto added = std::vector<location_pair>{};
  auto is_partner = bitvec_map<route_stop_idx_t>{};
  is_partner.resize(pairs.route_stops_.size());
  auto const locations_of = [&](auto const& v,
                                std::vector<route_stop_idx_t>& partners) {
    partners.clear();
    for (auto const x : v) {
      auto const q = location[x];
      if (!is_partner.test(q)) {
        is_partner.set(q);
        partners.push_back(q);
      }
    }
    for (auto const q : partners) {
      is_partner.set(q, false);
    }
  };
  auto const find = [&](route_stop_idx_t const from,
                        route_stop_idx_t const to) {
    return find_pair(lps[from], to);
  };
  auto const window_of = [&](route_stop_idx_t const from,
                             route_stop_idx_t const to) {
    auto const* e = find(from, to);
    return e != nullptr ? e->window_ : kAnyDuration;
  };
  auto const get = [&](route_stop_idx_t const from,
                       route_stop_idx_t const to) -> location_pair const& {
    auto const* o = find(from, to);
    utl::verify(o != nullptr, "location pair missing");
    return *o;
  };
  auto const fold = [](location_pair& m, location_pair const& o) {
    m.window_ = m.window_.intersect(o.window_);
  };
  auto const is_ok = [&](route_stop_idx_t const a, route_stop_idx_t const b,
                         window_t const& w) {
    return pairs.is_base(a) && pairs.is_base(b)
               ? w.contains(durations_of
                                .at(transfer_pair{pairs.route_stops_[a].stop_,
                                                  pairs.route_stops_[b].stop_})
                                .fallback_)
               : is_admissible(w, a == b);
  };
  auto const join = [&](route_stop_idx_t const s, route_stop_idx_t const l) {
    locations_of(out[s], out_partners);
    locations_of(in[s], in_partners);
    auto has_self = false;
    auto has_in_self = false;
    auto self = window_of(l, l);
    for (auto const q : out_partners) {
      if (q == s || q == l) {
        has_self = true;
        self = self.intersect(get(s, q).window_);
      } else if (!is_ok(l, q, window_of(l, q).intersect(get(s, q).window_))) {
        return false;
      }
    }
    for (auto const q : in_partners) {
      if (q == l) {
        has_self = true;
        has_in_self = true;
        self = self.intersect(get(l, s).window_);
      } else if (q != s &&
                 !is_ok(q, l, window_of(q, l).intersect(get(q, s).window_))) {
        return false;
      }
    }
    if (has_self && !is_ok(l, l, self)) {
      return false;
    }
    added.clear();
    for (auto const q : out_partners) {
      if (q == s || q == l) {
        continue;
      }
      if (auto* const m = find(l, q); m != nullptr) {
        fold(*m, get(s, q));
      } else {
        added.push_back(location_pair{q, kAnyDuration});
        fold(added.back(), get(s, q));
      }
    }
    for (auto const q : in_partners) {
      if (q == s || q == l) {
        continue;
      }
      if (auto* const m = find(q, l); m != nullptr) {
        fold(*m, get(q, s));
      } else {
        auto created = location_pair{l, kAnyDuration};
        fold(created, get(q, s));
        auto& v = lps[q];
        v.insert(std::ranges::lower_bound(v, l, {}, &location_pair::to_),
                 std::move(created));
      }
    }
    if (has_self) {
      auto* const m = find(l, l);
      auto created = location_pair{l, kAnyDuration};
      auto& target = m != nullptr ? *m : created;
      for (auto const q : out_partners) {
        if (q == s || q == l) {
          fold(target, get(s, q));
        }
      }
      if (has_in_self) {
        fold(target, get(l, s));
      }
      if (m == nullptr) {
        added.push_back(std::move(created));
      }
    }
    if (!added.empty()) {
      std::ranges::sort(added, {}, &location_pair::to_);
      auto& v = lps[l];
      auto const n = static_cast<std::ptrdiff_t>(v.size());
      v.insert(end(v), std::make_move_iterator(begin(added)),
               std::make_move_iterator(end(added)));
      std::ranges::inplace_merge(v, begin(v) + n, {}, &location_pair::to_);
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

void rewrite_routes(
    timetable& tt,
    shapes_storage* shapes,
    vector_map<route_idx_t, route_idx_t> const& base_route,
    vecvec<route_idx_t, transport_idx_t> const& route_transports) {
  auto const n_routes = tt.route_transport_ranges_.size();
  auto ranges = vector_map<route_idx_t, interval<transport_idx_t>>{};
  auto time_ranges = vector_map<route_idx_t, interval<std::uint32_t>>{};
  auto times = vector<delta>{};
  auto const& starts = route_transports.bucket_starts_;
  for (auto r = route_idx_t{0U}; r != base_route.size(); ++r) {
    auto const o = base_route[r];
    auto const transports = route_transports[r];
    ranges.emplace_back(transport_idx_t{starts[to_idx(r)]},
                        transport_idx_t{starts[to_idx(r) + 1U]});

    auto const begin_times = times.size();
    auto const n_stops =
        static_cast<stop_idx_t>(tt.route_location_seq_[r].size());
    for (auto i = stop_idx_t{1U}; i != n_stops; ++i) {
      for (auto const t : transports) {
        times.push_back(tt.event_mam(o, t, static_cast<stop_idx_t>(i - 1U),
                                     event_type::kDep));
      }
      for (auto const t : transports) {
        times.push_back(tt.event_mam(o, t, i, event_type::kArr));
      }
    }
    time_ranges.emplace_back(static_cast<std::uint32_t>(begin_times),
                             static_cast<std::uint32_t>(times.size()));

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
  tt.route_transport_ranges_ = std::move(ranges);
  tt.route_stop_time_ranges_ = std::move(time_ranges);
  tt.route_stop_times_ = std::move(times);

  auto const& order = route_transports.data_;
  auto inverse = vector_map<transport_idx_t, transport_idx_t>{};
  inverse.resize(order.size());
  for (auto t = transport_idx_t{0U}; t != order.size(); ++t) {
    inverse[order[to_idx(t)]] = t;
  }
  auto const permute = [&](auto& v) {
    auto copy = v;
    for (auto t = transport_idx_t{0U}; t != order.size(); ++t) {
      copy[t] = v[order[to_idx(t)]];
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
    auto const* self = find_pair(locations.pairs_[s], s);
    auto const d = self == nullptr ? std::optional{fallback}
                                   : pick_duration(self->window_, fallback,
                                                   kMaxTransferTime);
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
                   transfer_durations_t const& durations_of,
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
      auto const fallback = durations_of
                                .at(transfer_pair{pairs.route_stops_[a].stop_,
                                                  pairs.route_stops_[c].stop_})
                                .fallback_;
      auto const d = pick_duration(m.window_, fallback, kMaxFootpath);
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
                  signatures const& sigs,
                  route_stop_pairs const& pairs,
                  transport_routes const& routes,
                  route_stop_locations const& locations,
                  vector_map<route_stop_idx_t, location_idx_t> const& virts,
                  shapes_storage* shapes) {
  auto const n_routes = tt.n_routes();
  auto route_transports = vecvec<route_idx_t, transport_idx_t>{};
  auto transports = std::vector<transport_idx_t>{};
  for (auto q = route_idx_t{0U}; q != routes.base_route_.size(); ++q) {
    auto const r = routes.base_route_[q];
    auto const range = tt.route_transport_ranges_[r];
    transports.clear();
    for (auto t = route_transport_idx_t{0U};
         t != route_transport_idx_t{range.size()}; ++t) {
      if (routes.get_route(r, t) == q) {
        transports.push_back(range.from_ + to_idx(t));
      }
    }
    route_transports.emplace_back(transports);
  }
  auto seq = stop_seq_t{};
  auto const get_seq = [&](route_idx_t const q) -> stop_seq_t const& {
    auto const r = routes.base_route_[q];
    auto const base_seq = tt.route_location_seq_[r];
    seq.assign(base_seq.begin(), base_seq.end());
    for (auto i = stop_idx_t{0U}; i != seq.size(); ++i) {
      auto const l = stop{seq[i]}.location_idx();
      auto s = pairs.route_stop_of_.find(route_stop{q, i, l});
      if (s == end(pairs.route_stop_of_) && q != r) {
        auto const b = pairs.route_stop_of_.find(route_stop{r, i, l});
        if (b != end(pairs.route_stop_of_) &&
            !has_route(pairs.route_stops_[b->second])) {
          s = b;
        }
      }
      if (s != end(pairs.route_stop_of_)) {
        seq[i] = stop{seq[i]}
                     .with_location(virts[locations.location_[s->second]])
                     .value();
      }
    }
    return seq;
  };
  for (auto q = route_idx_t{n_routes}; q != routes.base_route_.size(); ++q) {
    tt.route_location_seq_.emplace_back(get_seq(q));
  }
  for (auto r = route_idx_t{0U}; r != n_routes; ++r) {
    if (pairs.of_route_[r].empty() && sigs.route_stop_sigs_[r].size() == 0U) {
      continue;
    }
    std::ranges::copy(get_seq(r), tt.route_location_seq_[r].begin());
  }
  rewrite_routes(tt, shapes, routes.base_route_, route_transports);
  log(log_lvl::info, "loader.virtual_locations", "{} routes -> {}", n_routes,
      tt.n_routes());
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
  auto const durations_of = get_transfer_durations(tt, sigs);
  auto pairs = route_stop_pairs{};
  pairs.of_route_.resize(tt.n_routes());
  {
    auto location_route_stops =
        get_location_route_stops(tt, sigs, durations_of);
    auto expected = expected_durations{.tt_ = tt, .sigs_ = sigs};
    for (auto const& [stops, pair_durations] : durations_of) {
      expected.reset(stops, pair_durations.fallback_);
      auto& at_from = location_route_stops.at(stops.from_);
      auto& at_to = location_route_stops.at(stops.to_);
      add_base_transfers(tt, stops, pair_durations, at_from, at_to, expected,
                         pairs);
      add_meeting_transfers(tt, stops, pair_durations, at_from, at_to, expected,
                            pairs);
    }
  }

  auto const routes = split_routes(tt, durations_of, pairs);
  auto const locations = merge_locations(durations_of, pairs);
  auto const n_locations = tt.n_locations();
  auto const virts = add_virts(tt, pairs, locations);
  if (tt.n_locations() == n_locations) {
    return false;
  }
  add_footpaths(tt, durations_of, pairs, locations, virts, durations);
  write_routes(tt, sigs, pairs, routes, locations, virts, shapes);
  return true;
}

}  // namespace nigiri::loader
