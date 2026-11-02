#include "nigiri/rt/rt_transfer_rules.h"

#include <algorithm>
#include <map>
#include <optional>
#include <span>
#include <tuple>
#include <vector>

#include "utl/erase_duplicates.h"
#include "utl/get_or_create.h"
#include "utl/helpers/algorithm.h"
#include "utl/lookup.h"

#include "nigiri/footpath.h"
#include "nigiri/logging.h"
#include "nigiri/routing/search_location.h"
#include "nigiri/routing/transfers.h"
#include "nigiri/rt/frun.h"
#include "nigiri/transfer_rule_sides.h"

namespace nigiri::rt {

using side_t = transfer_rule_side_idx;
using stop_event = pair<rt_transport_idx_t, stop_idx_t>;

struct change_end {
  rt::run r_;
  stop_idx_t stop_;
  unixtime_t time_;
  location_idx_t location_;
};

struct rule_event {
  unixtime_t time_;
  transport t_;
  stop_idx_t stop_;
};

struct rule_events_key {
  trip_idx_t trip_;
  source_idx_t src_;
  route_id_idx_t route_;
  location_idx_t stop_;
  event_type ev_;
};

struct change_side {
  change_end e_;
  bool is_out_;
  std::vector<side_t> signature_;
  location_idx_t base_;
};

struct rule_partner {
  change_end y_;
  bool is_out_;
  duration_t gap_;
  duration_t expected_;
};

struct rule_conflicts {
  interval<unixtime_t> horizon_;
  vector_map<location_idx_t, duration_t> rule_times_;
  bitvec_map<location_idx_t> is_virt_or_rule_stop_;
  bitvec_map<trip_idx_t> rule_trips_;
  hash_map<location_idx_t, std::vector<location_idx_t>> to_stops_, from_stops_;
  hash_map<rule_events_key, std::vector<rule_event>> events_;
  hash_map<pair<location_idx_t, location_idx_t>, duration_t> durations_;
  std::map<location_idx_t, std::vector<footpath>> new_footpaths_out_,
      new_footpaths_in_;
  hash_map<pair<transport_idx_t, pair<stop_idx_t, location_idx_t>>,
           std::vector<side_t>>
      signatures_;
  hash_map<stop_event, std::uint32_t> placements_;
  std::uint32_t n_unresolved_{0U};
};

std::optional<change_end> get_change_end(rt_timetable const& rtt,
                                         rt::frun const& fr,
                                         stop_idx_t const i,
                                         event_type const ev) {
  if (!fr.is_rt() && !rtt.is_transport_active(fr.t_.t_idx_, fr.t_.day_)) {
    return std::nullopt;
  }
  if (fr.is_cancelled() ||
      (ev == event_type::kArr ? i == 0U : i + 1U >= fr.size())) {
    return std::nullopt;
  }
  auto const stp = fr[i];
  if (!(ev == event_type::kArr ? stp.out_allowed() : stp.in_allowed())) {
    return std::nullopt;
  }
  return change_end{
      .r_ = fr,
      .stop_ = i,
      .time_ = stp.time(ev),
      .location_ = routing::search_location(kDefaultProfile, stp)};
}

bool is_same_run(change_end const& a, change_end const& b) {
  return a.r_.t_ == b.r_.t_ && a.r_.rt_ == b.r_.rt_;
}

duration_t get_duration(timetable const& tt,
                        rt_timetable const& rtt,
                        rule_conflicts& st,
                        bool const is_out,
                        location_idx_t const l,
                        location_idx_t const other) {
  auto const key = is_out ? pair{l, other} : pair{other, l};
  return utl::get_or_create(st.durations_, key, [&]() {
    auto d = routing::shortest_transfer(tt, &rtt, kDefaultProfile, key.first,
                                        key.second)
                 .value_or(footpath::kBan);
    if (auto const it = st.new_footpaths_out_.find(key.first);
        it != end(st.new_footpaths_out_)) {
      for (auto const fp : it->second) {
        if (fp.target() == key.second) {
          d = std::min(d, fp.duration());
        }
      }
    }
    return d;
  });
}

std::vector<side_t> const& get_signature(timetable const& tt,
                                         rt_timetable const& rtt,
                                         rule_conflicts& st,
                                         change_end const& x) {
  static auto const no_sides = std::vector<side_t>{};
  auto const t = x.r_.t_.t_idx_;
  auto const base = rtt.base(x.location_);
  if (t == transport_idx_t::invalid() ||
      x.stop_ >= tt.route_location_seq_[tt.transport_route_[t]].size()) {
    return no_sides;
  }
  return utl::get_or_create(st.signatures_, pair{t, pair{x.stop_, base}},
                            [&]() {
                              auto sig = std::vector<side_t>{};
                              get_stop_signature(tt, t, x.stop_, base, sig);
                              return sig;
                            });
}

duration_t get_expected(timetable const& tt,
                        rt_timetable const& rtt,
                        rule_conflicts& st,
                        change_side const& x,
                        std::span<side_t const> other_sig,
                        location_idx_t const other_base) {
  auto const rule =
      x.is_out_
          ? get_change_rule(tt, x.signature_, other_sig, x.base_, other_base)
          : get_change_rule(tt, other_sig, x.signature_, other_base, x.base_);
  if (rule != transfer_rule_idx_t::invalid()) {
    return tt.transfer_rules_.rules_[rule].duration_;
  }
  return get_duration(tt, rtt, st, x.is_out_, x.base_, other_base);
}

bool is_possible(duration_t const gap, duration_t const d) {
  return d < footpath::kBan && gap >= d;
}

duration_t get_gap(change_side const& x, change_end const& y) {
  return std::chrono::duration_cast<duration_t>(
      x.is_out_ ? y.time_ - x.e_.time_ : x.e_.time_ - y.time_);
}

std::vector<location_idx_t> get_locations(timetable const& tt,
                                          rt_timetable const& rtt,
                                          location_idx_t const base) {
  auto locations = std::vector<location_idx_t>{base};
  tt.locations_.for_each_virt(
      base, [&](location_idx_t const v) { locations.push_back(v); });
  rtt.for_each_rt_virt(base,
                       [&](location_idx_t const w) { locations.push_back(w); });
  return locations;
}

std::vector<location_idx_t> get_partner_stops(rule_conflicts const& st,
                                              change_side const& x) {
  auto stops = std::vector<location_idx_t>{x.base_};
  auto const& m = x.is_out_ ? st.to_stops_ : st.from_stops_;
  if (auto const it = m.find(x.base_); it != end(m)) {
    stops.insert(end(stops), begin(it->second), end(it->second));
  }
  utl::erase_duplicates(stops);
  return stops;
}

day_idx_t get_day(timetable const& tt, unixtime_t const t) {
  auto const& iv = tt.internal_interval();
  return tt.day_idx_mam(std::clamp(t, iv.from_, iv.to_ - duration_t{1})).first;
}

template <typename Fn>
void for_each_event_at(timetable const& tt,
                       rt_timetable const& rtt,
                       location_idx_t const l,
                       event_type const ev,
                       interval<unixtime_t> const window,
                       Fn&& fn) {
  if (!rtt.is_rt_location(l)) {
    for (auto const r : tt.location_routes_[l]) {
      auto const seq = tt.route_location_seq_[r];
      for (auto j = stop_idx_t{0U}; j != seq.size(); ++j) {
        if (stop{seq[j]}.location_idx() != l) {
          continue;
        }
        for (auto const t : tt.route_transport_ranges_[r]) {
          auto const mam = tt.event_mam(t, j, ev).as_duration();
          for (auto d = get_day(tt, window.from_ - mam);
               d <= get_day(tt, window.to_ - mam); ++d) {
            auto const tr = transport{t, d};
            if (!window.contains(tt.event_time(tr, j, ev)) ||
                rtt.resolve_rt(tr) != rt_transport_idx_t::invalid()) {
              continue;
            }
            auto const e =
                get_change_end(rtt, rt::frun::from_t(tt, &rtt, tr), j, ev);
            if (e.has_value() && window.contains(e->time_)) {
              fn(*e);
            }
          }
        }
      }
    }
  }
  for (auto const rt_t : rtt.location_rt_transports_[l]) {
    auto const n =
        static_cast<stop_idx_t>(rtt.rt_transport_location_seq_[rt_t].size());
    for (auto j = stop_idx_t{0U}; j != n; ++j) {
      if (rtt.stop_location(rt_t, j) != l) {
        continue;
      }
      auto const e =
          get_change_end(rtt, rt::frun::from_rt(tt, &rtt, rt_t), j, ev);
      if (e.has_value() && window.contains(e->time_)) {
        fn(*e);
      }
    }
  }
}

template <typename SortedPairs, typename Key>
auto values_of(SortedPairs const& v, Key const key) {
  using entry = std::ranges::range_value_t<SortedPairs>;
  return std::ranges::equal_range(v, key, {}, &entry::first) |
         std::views::transform(&entry::second);
}

std::vector<rule_event> get_rule_events(timetable const& tt,
                                        rt_timetable const& rtt,
                                        interval<unixtime_t> const horizon,
                                        rule_events_key const& key) {
  auto const ev = key.ev_;
  auto events = std::vector<rule_event>{};
  auto positions = std::vector<stop_idx_t>{};
  auto const add_trip = [&](trip_idx_t const trip) {
    for (auto const [t, stops] : tt.trip_transport_ranges_[trip]) {
      auto const seq = tt.route_location_seq_[tt.transport_route_[t]];
      auto const mam = [&](stop_idx_t const j) {
        return tt.event_mam(t, j, j == 0U ? event_type::kDep : event_type::kArr)
            .as_duration();
      };
      auto const is_rule_stop = [&](stop_idx_t const j,
                                    location_idx_t const l) {
        return (ev == event_type::kArr ? j != 0U : j + 1U != seq.size()) &&
               tt.locations_.is_self_or_parent(key.stop_, l);
      };
      positions.clear();
      for (auto const j : stops) {
        auto const stp = stop{seq[j]};
        if ((ev == event_type::kArr ? stp.out_allowed() : stp.in_allowed()) &&
            is_rule_stop(j, tt.base(stp.location_idx()))) {
          positions.push_back(j);
        }
      }
      auto const last_day = get_day(tt, horizon.to_ - mam(stops.from_));
      for (auto d =
               get_day(tt, horizon.from_ - date::days{1} -
                               mam(static_cast<stop_idx_t>(stops.to_ - 1U)));
           d <= last_day; ++d) {
        auto const rt_t = rtt.resolve_rt({t, d});
        if (rt_t == rt_transport_idx_t::invalid()) {
          if (!rtt.is_transport_active(t, d)) {
            continue;
          }
          for (auto const j : positions) {
            auto const time = tt.event_time({t, d}, j, ev);
            if (horizon.contains(time)) {
              events.push_back({time, {t, d}, j});
            }
          }
        } else {
          for (auto const j : stops) {
            if (!is_rule_stop(j, rtt.base(rtt.stop_location(rt_t, j)))) {
              continue;
            }
            auto const e =
                get_change_end(rtt, rt::frun::from_t(tt, &rtt, {t, d}), j, ev);
            if (e.has_value() && horizon.contains(e->time_)) {
              events.push_back({e->time_, {t, d}, j});
            }
          }
        }
      }
    }
  };
  if (key.trip_ != trip_idx_t::invalid()) {
    add_trip(key.trip_);
  } else {
    for (auto const trip :
         tt.route_ids_[key.src_].route_id_trips_[key.route_]) {
      add_trip(trip);
    }
  }
  std::ranges::sort(events, {}, &rule_event::time_);
  return events;
}

std::vector<rule_partner> get_rule_partners(timetable const& tt,
                                            rt_timetable const& rtt,
                                            rule_conflicts& st,
                                            change_side const& x,
                                            duration_t const window) {
  auto const& tr = tt.transfer_rules_;
  auto const ev = x.is_out_ ? event_type::kArr : event_type::kDep;
  auto const partner_ev = flip(ev);
  auto sides = std::vector<side_t>{};
  for (auto const s : x.signature_) {
    if (s.ev_type() == ev && tr.rules_[s.rule()].is_qualified(partner_ev)) {
      sides.push_back(side_t{s.rule(), partner_ev});
    }
  }
  for (auto const l : {x.base_, tt.locations_.parents_[x.base_]}) {
    if (l == location_idx_t::invalid()) {
      continue;
    }
    for (auto const s : values_of(tr.stop_rules_, l)) {
      if (s.ev_type() == ev && tr.rules_[s.rule()].is_qualified(partner_ev)) {
        sides.push_back(side_t{s.rule(), partner_ev});
      }
    }
  }
  utl::erase_duplicates(sides);

  auto partners = std::vector<rule_partner>{};
  auto const from = x.is_out_ ? x.e_.time_ : x.e_.time_ - window;
  auto const to = x.is_out_ ? x.e_.time_ + window : x.e_.time_;
  for (auto const q : sides) {
    auto const& rule = tr.rules_[q.rule()];
    auto const key = rule_events_key{rule.trip(partner_ev), rule.src_,
                                     rule.route(partner_ev),
                                     rule.stop(partner_ev), partner_ev};
    auto const& events = utl::get_or_create(st.events_, key, [&]() {
      return get_rule_events(tt, rtt, st.horizon_, key);
    });
    for (auto it =
             std::ranges::lower_bound(events, from, {}, &rule_event::time_);
         it != end(events) && it->time_ <= to; ++it) {
      auto const y = get_change_end(rtt, rt::frun::from_t(tt, &rtt, it->t_),
                                    it->stop_, partner_ev);
      if (!y.has_value() || is_same_run(x.e_, *y)) {
        continue;
      }
      auto const& y_sig = get_signature(tt, rtt, st, *y);
      if (!std::ranges::binary_search(y_sig, q)) {
        continue;
      }
      partners.push_back(rule_partner{
          .y_ = *y,
          .is_out_ = x.is_out_,
          .gap_ = get_gap(x, *y),
          .expected_ =
              get_expected(tt, rtt, st, x, y_sig, rtt.base(y->location_))});
    }
  }
  return partners;
}

bool plain_partners_fit(timetable const& tt,
                        rt_timetable const& rtt,
                        rule_conflicts& st,
                        change_side const& x,
                        location_idx_t const s,
                        location_idx_t const q,
                        duration_t const d,
                        duration_t const plain) {
  if (d == plain) {
    return true;
  }
  auto const lo = std::min(d, plain);
  auto const hi = std::max(d, plain);
  auto const window = x.is_out_ ? interval{x.e_.time_ + lo, x.e_.time_ + hi}
                                : interval{x.e_.time_ - hi + duration_t{1},
                                           x.e_.time_ - lo + duration_t{1}};
  auto is_ok = true;
  for_each_event_at(
      tt, rtt, q, x.is_out_ ? event_type::kDep : event_type::kArr, window,
      [&](change_end const& y) {
        if (is_ok && !is_same_run(x.e_, y)) {
          auto const gap = get_gap(x, y);
          is_ok =
              is_possible(gap, d) ==
              is_possible(gap, get_expected(tt, rtt, st, x,
                                            get_signature(tt, rtt, st, y), s));
        }
      });
  return is_ok;
}

bool fits(timetable const& tt,
          rt_timetable const& rtt,
          rule_conflicts& st,
          std::span<change_side const> sides,
          std::span<rule_partner const> partners,
          location_idx_t const l) {
  for (auto const& p : partners) {
    auto const d = get_duration(tt, rtt, st, p.is_out_, l, p.y_.location_);
    if (is_possible(p.gap_, d) != is_possible(p.gap_, p.expected_)) {
      return false;
    }
  }
  for (auto const& x : sides) {
    for (auto const s : get_partner_stops(st, x)) {
      auto const plain = get_expected(tt, rtt, st, x, {}, s);
      if (get_duration(tt, rtt, st, x.is_out_, l, s) != plain) {
        return false;
      }
      for (auto const q : get_locations(tt, rtt, s)) {
        if (q != s && !plain_partners_fit(
                          tt, rtt, st, x, s, q,
                          get_duration(tt, rtt, st, x.is_out_, l, q), plain)) {
          return false;
        }
      }
    }
  }
  return true;
}

std::pair<location_idx_t, std::vector<change_end>> add_location(
    timetable const& tt,
    rt_timetable& rtt,
    rule_conflicts& st,
    location_idx_t const base,
    std::span<change_side const> sides,
    std::span<rule_partner const> partners) {
  auto const v = rtt.add_rt_location(base);
  auto moved = std::vector<change_end>{};
  for (auto const is_out : {true, false}) {
    auto durations = hash_map<location_idx_t, duration_t>{};
    for (auto const fp : routing::get_transfers(
             tt, nullptr, kDefaultProfile,
             is_out ? direction::kForward : direction::kBackward, base)) {
      auto const it = durations.emplace(fp.target(), fp.duration()).first;
      it->second = std::min(it->second, fp.duration());
    }
    durations[base] = duration_t{rtt.transfer_time(base)};
    rtt.for_each_rt_virt([&](location_idx_t const w, rt_location_idx_t) {
      if (auto const d = utl::lookup(durations, rtt.base(w));
          w != v && d.has_value()) {
        durations.emplace(w, *d);
      }
    });
    for (auto const& x : sides) {
      if (x.is_out_ != is_out) {
        continue;
      }
      for (auto const s : get_partner_stops(st, x)) {
        auto const plain = get_expected(tt, rtt, st, x, {}, s);
        for (auto const q : get_locations(tt, rtt, s)) {
          if (q == v) {
            continue;
          }
          auto d = plain;
          if (q != s) {
            auto const at_q = [&](rule_partner const& p) {
              return p.is_out_ == is_out && p.y_.location_ == q;
            };
            auto const fit = [&](duration_t const c) {
              return utl::all_of(partners,
                                 [&](rule_partner const& p) {
                                   return !at_q(p) ||
                                          is_possible(p.gap_, c) ==
                                              is_possible(p.gap_, p.expected_);
                                 }) &&
                     plain_partners_fit(tt, rtt, st, x, s, q, c, plain);
            };
            if (!fit(plain)) {
              for (auto const& p : partners) {
                if (at_q(p) && fit(p.expected_)) {
                  d = p.expected_;
                  break;
                }
              }
            }
          }
          durations[q] = d;
        }
      }
    }
    for (auto const& p : partners) {
      if (p.is_out_ != is_out) {
        continue;
      }
      auto const d =
          utl::lookup(durations, p.y_.location_).value_or(footpath::kBan);
      if (is_possible(p.gap_, d) != is_possible(p.gap_, p.expected_)) {
        moved.push_back(p.y_);
      }
    }
    auto& out = is_out ? st.new_footpaths_out_ : st.new_footpaths_in_;
    auto& in = is_out ? st.new_footpaths_in_ : st.new_footpaths_out_;
    for (auto const& [l, d] : durations) {
      if (d != footpath::kBan) {
        out[v].emplace_back(l, d);
        in[l].emplace_back(v, d);
      }
    }
  }
  return {v, moved};
}

std::vector<stop_event> get_changed_stops(timetable const& tt,
                                          rt_timetable const& rtt,
                                          rule_conflicts const& st) {
  auto const has_rules = [&](transport_idx_t const t) {
    return utl::any_of(tt.transport_to_trip_section_[t], [&](auto const s) {
      return utl::any_of(tt.merged_trips_[s], [&](trip_idx_t const trip) {
        return st.rule_trips_.test(trip);
      });
    });
  };
  auto changed = std::vector<stop_event>{};
  for (auto rt_t = rt_transport_idx_t{0U}; rt_t != rtt.n_rt_transports();
       ++rt_t) {
    auto const fr = rt::frun::from_rt(tt, &rtt, rt_t);
    if (!fr.is_scheduled()) {
      continue;
    }
    auto is_rule_transport = std::optional<bool>{};
    auto const seq = tt.route_location_seq_[tt.transport_route_[fr.t_.t_idx_]];
    auto const n = fr.size();
    for (auto i = stop_idx_t{0U}; i != n; ++i) {
      auto const l = rtt.stop_location(rt_t, i);
      if (!rtt.is_rt_location(l)) {
        if (!st.is_virt_or_rule_stop_.test(l)) {
          continue;
        }
        if (!tt.locations_.is_virt(l)) {
          if (!is_rule_transport.has_value()) {
            is_rule_transport = has_rules(fr.t_.t_idx_);
          }
          if (!*is_rule_transport) {
            continue;
          }
        }
      }
      auto const stp = fr[i];
      if (i >= seq.size() || l != stop{seq[i]}.location_idx() ||
          (i != 0U && stp.time(event_type::kArr) !=
                          stp.scheduled_time(event_type::kArr)) ||
          (i + 1U != n && stp.time(event_type::kDep) !=
                              stp.scheduled_time(event_type::kDep))) {
        changed.emplace_back(rt_t, i);
      }
    }
  }
  return changed;
}

void resolve_conflicts(timetable const& tt, rt_timetable& rtt) {
  auto const& rules = tt.transfer_rules_.rules_;
  if (rules.empty()) {
    return;
  }
  auto const timer = scoped_timer{"rt.resolve_conflicts"};

  auto st = rule_conflicts{};
  st.rule_times_.resize(tt.n_locations(), duration_t{0});
  auto const for_each_covered = [&](location_idx_t const s, auto&& fn) {
    fn(s);
    for (auto const c : tt.locations_.children_[s]) {
      if (!tt.locations_.is_virt(c)) {
        fn(c);
      }
    }
  };
  auto prev = pair{location_idx_t::invalid(), location_idx_t::invalid()};
  for (auto const& r : rules) {
    auto const d = std::max(r.duration_, duration_t{1});
    for (auto const l : {r.from_stop_, r.to_stop_}) {
      st.rule_times_[l] = std::max(st.rule_times_[l], d);
    }
    if (!r.is_qualified() || pair{r.from_stop_, r.to_stop_} == prev) {
      continue;
    }
    prev = pair{r.from_stop_, r.to_stop_};
    for_each_covered(r.from_stop_, [&](location_idx_t const f) {
      for_each_covered(r.to_stop_, [&](location_idx_t const t) {
        st.to_stops_[f].push_back(t);
        st.from_stops_[t].push_back(f);
      });
    });
  }
  for (auto* m : {&st.to_stops_, &st.from_stops_}) {
    for (auto& [_, stops] : *m) {
      utl::erase_duplicates(stops);
    }
  }
  st.is_virt_or_rule_stop_.resize(tt.n_locations());
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    auto const parent = tt.locations_.parents_[l];
    if (tt.locations_.is_virt(l) || st.rule_times_[l] != duration_t{0} ||
        (parent != location_idx_t::invalid() &&
         st.rule_times_[parent] != duration_t{0})) {
      st.is_virt_or_rule_stop_.set(l);
    }
  }
  auto const& tr = tt.transfer_rules_;
  st.rule_trips_.resize(tt.n_trips());
  for (auto const& s : tr.trip_rules_) {
    st.rule_trips_.set(s.qualifier_);
  }
  for (auto it = begin(tr.route_rules_); it != end(tr.route_rules_);) {
    auto const [src, route] = it->qualifier_;
    for (auto const trip : tt.route_ids_[src].route_id_trips_[route]) {
      st.rule_trips_.set(trip);
    }
    it = std::find_if(it, end(tr.route_rules_), [&](auto const& x) {
      return x.qualifier_ != it->qualifier_;
    });
  }

  auto queue = get_changed_stops(tt, rtt, st);
  auto const n_changed = queue.size();
  st.horizon_ = interval{unixtime_t::max(), unixtime_t::min()};
  for (auto const [rt_t, i] : queue) {
    auto const t = rtt.unix_event_time(
        rt_t, i, i == 0U ? event_type::kDep : event_type::kArr);
    st.horizon_ = st.horizon_.extend({t - footpath::kMax, t + footpath::kMax});
  }

  auto const n_transports = rtt.n_rt_transports();
  auto n_checked = 0U;
  auto n_moves = 0U;
  auto sides = std::vector<change_side>{};
  auto partners = std::vector<rule_partner>{};
  while (!queue.empty()) {
    auto const [rt_t, i] = queue.back();
    queue.pop_back();
    ++n_checked;
    auto const fr = rt::frun::from_rt(tt, &rtt, rt_t);
    sides.clear();
    for (auto const ev : {event_type::kArr, event_type::kDep}) {
      if (auto const e = get_change_end(rtt, fr, i, ev); e.has_value()) {
        sides.push_back(
            change_side{.e_ = *e,
                        .is_out_ = ev == event_type::kArr,
                        .signature_ = get_signature(tt, rtt, st, *e),
                        .base_ = rtt.base(e->location_)});
      }
    }
    if (sides.empty()) {
      continue;
    }
    auto const current = sides.front().e_.location_;
    auto const base = sides.front().base_;
    auto const parent = tt.locations_.parents_[base];
    auto const window =
        std::max(st.rule_times_[base], parent == location_idx_t::invalid()
                                           ? duration_t{0}
                                           : st.rule_times_[parent]);
    partners.clear();
    for (auto const& x : sides) {
      auto p = get_rule_partners(tt, rtt, st, x, window);
      partners.insert(end(partners), begin(p), end(p));
    }
    if (fits(tt, rtt, st, sides, partners, current)) {
      continue;
    }
    auto to = location_idx_t::invalid();
    auto moved = std::vector<change_end>{};
    for (auto const l : get_locations(tt, rtt, base)) {
      if (l != current && fits(tt, rtt, st, sides, partners, l)) {
        to = l;
        break;
      }
    }
    if (to == location_idx_t::invalid()) {
      std::tie(to, moved) = add_location(tt, rtt, st, base, sides, partners);
    }
    if (++st.placements_[stop_event{rt_t, i}] > 3U) {
      ++st.n_unresolved_;
      continue;
    }
    rtt.update_stop_location(rt_t, i, to);
    ++n_moves;
    for (auto const& y : moved) {
      auto y_rt = y.r_.is_rt() ? y.r_.rt_ : rtt.resolve_rt(y.r_.t_);
      if (y_rt == rt_transport_idx_t::invalid()) {
        y_rt = rtt.add_rt_transport(rt::frun{tt, &rtt, y.r_}.id().src_, tt,
                                    y.r_.t_);
      }
      queue.emplace_back(y_rt, y.stop_);
    }
  }

  auto const write_rt_footpaths =
      [&](vecvec<location_idx_t, footpath>& fps,
          std::map<location_idx_t, std::vector<footpath>> const& added) {
        if (added.empty()) {
          return;
        }
        auto next = vecvec<location_idx_t, footpath>{};
        auto const copy_until = [&](location_idx_t const to) {
          if (fps.data_.empty()) {
            next.resize(to_idx(to));
            return;
          }
          for (auto l = location_idx_t{next.size()}; l != to; ++l) {
            next.emplace_back(fps[l]);
          }
        };
        for (auto const& [l, l_fps] : added) {
          copy_until(l);
          next.emplace_back(fps[l]);
          for (auto const& fp : l_fps) {
            next.back().push_back(fp);
          }
        }
        copy_until(location_idx_t{rtt.n_locations()});
        fps = std::move(next);
      };
  write_rt_footpaths(rtt.rt_footpaths_out_, st.new_footpaths_out_);
  write_rt_footpaths(rtt.rt_footpaths_in_, st.new_footpaths_in_);

  log(log_lvl::info, "rt.resolve_conflicts",
      "{} changed stops, {} checked, {} moves, {} unresolved, {} new "
      "real-time transports, {} real-time locations",
      n_changed, n_checked, n_moves, st.n_unresolved_,
      rtt.n_rt_transports() - n_transports, rtt.n_rt_locations());
}

}  // namespace nigiri::rt
