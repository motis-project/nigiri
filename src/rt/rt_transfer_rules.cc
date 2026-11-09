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
  day_idx_t day_;
};

struct change_side {
  change_end e_;
  event_type ev_;
  std::vector<transfer_rule_side_idx> signature_;
  location_idx_t base_;
};

struct rule_partner {
  change_end y_;
  event_type ev_;
  duration_t gap_;
  duration_t expected_;
};

using event_cache_t = hash_map<rule_events_key, std::vector<rule_event>>;
using duration_cache_t =
    hash_map<pair<location_idx_t, location_idx_t>, duration_t>;
using new_footpaths_t = std::map<location_idx_t, std::vector<footpath>>;

std::optional<change_end> get_change_end(rt_timetable const& rtt,
                                         rt::frun const& fr,
                                         stop_idx_t const i,
                                         event_type const ev) {
  if ((!fr.is_rt() && !rtt.is_transport_active(fr.t_.t_idx_, fr.t_.day_)) ||
      fr.is_cancelled() ||
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
                        duration_cache_t& durations,
                        new_footpaths_t const& new_footpaths_out,
                        event_type const ev,
                        location_idx_t const l,
                        location_idx_t const other) {
  auto const key = ev == event_type::kArr ? pair{l, other} : pair{other, l};
  return utl::get_or_create(durations, key, [&]() {
    auto d = routing::shortest_transfer(tt, &rtt, kDefaultProfile, key.first,
                                        key.second)
                 .value_or(footpath::kBan);
    if (auto const it = new_footpaths_out.find(key.first);
        it != end(new_footpaths_out)) {
      for (auto const fp : it->second) {
        if (fp.target() == key.second) {
          d = std::min(d, fp.duration());
        }
      }
    }
    return d;
  });
}

std::vector<transfer_rule_side_idx> get_signature(timetable const& tt,
                                                  rt_timetable const& rtt,
                                                  change_end const& x) {
  auto sig = std::vector<transfer_rule_side_idx>{};
  if (x.r_.t_.t_idx_ != transport_idx_t::invalid()) {
    get_stop_signature(tt, x.r_.t_.t_idx_, x.stop_, rtt.base(x.location_), sig);
  }
  return sig;
}

duration_t get_expected(timetable const& tt,
                        rt_timetable const& rtt,
                        duration_cache_t& durations,
                        new_footpaths_t const& new_footpaths_out,
                        change_side const& x,
                        std::span<transfer_rule_side_idx const> other_sig,
                        location_idx_t const other_base) {
  auto const rule =
      x.ev_ == event_type::kArr
          ? get_change_rule(tt, x.signature_, other_sig, x.base_, other_base)
          : get_change_rule(tt, other_sig, x.signature_, other_base, x.base_);
  if (rule != transfer_rule_idx_t::invalid()) {
    return tt.transfer_rules_.rules_[rule].duration_;
  }
  return get_duration(tt, rtt, durations, new_footpaths_out, x.ev_, x.base_,
                      other_base);
}

bool is_possible(duration_t const gap, duration_t const d) {
  return d < footpath::kBan && gap >= d;
}

duration_t get_gap(change_side const& x, change_end const& y) {
  return std::chrono::duration_cast<duration_t>(
      x.ev_ == event_type::kArr ? y.time_ - x.e_.time_ : x.e_.time_ - y.time_);
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

std::vector<location_idx_t> get_partner_locations(timetable const& tt,
                                                  change_side const& x) {
  auto const& tr = tt.transfer_rules_;
  auto locations = std::vector<location_idx_t>{x.base_};
  for (auto const l : {x.base_, tt.locations_.parents_[x.base_]}) {
    if (l == location_idx_t::invalid()) {
      continue;
    }
    for (auto const s : values_of(
             x.ev_ == event_type::kArr ? tr.to_stops_ : tr.from_stops_, l)) {
      locations.push_back(s);
      for (auto const c : tt.locations_.children_[s]) {
        if (!tt.locations_.is_virt(c)) {
          locations.push_back(c);
        }
      }
    }
  }
  utl::erase_duplicates(locations);
  return locations;
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

std::vector<rule_event> get_rule_events(timetable const& tt,
                                        rt_timetable const& rtt,
                                        rule_events_key const& key) {
  auto const ev = key.ev_;
  auto const window =
      interval{tt.to_unixtime(key.day_, minutes_after_midnight_t{0}),
               tt.to_unixtime(day_idx_t{to_idx(key.day_) + 1U},
                              minutes_after_midnight_t{0})};
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
      auto const last_day =
          get_day(tt, window.to_ + footpath::kMax - mam(stops.from_));
      for (auto d =
               get_day(tt, window.from_ - date::days{1} -
                               mam(static_cast<stop_idx_t>(stops.to_ - 1U)));
           d <= last_day; ++d) {
        auto const rt_t = rtt.resolve_rt({t, d});
        if (rt_t == rt_transport_idx_t::invalid()) {
          if (!rtt.is_transport_active(t, d)) {
            continue;
          }
          for (auto const j : positions) {
            auto const time = tt.event_time({t, d}, j, ev);
            if (window.contains(time)) {
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
            if (e.has_value() && window.contains(e->time_)) {
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

std::vector<rule_partner> get_rule_partners(
    timetable const& tt,
    rt_timetable const& rtt,
    event_cache_t& events,
    duration_cache_t& durations,
    new_footpaths_t const& new_footpaths_out,
    change_side const& x,
    duration_t const window) {
  auto const& tr = tt.transfer_rules_;
  auto const partner_ev = flip(x.ev_);
  auto rules = std::vector<transfer_rule_side_idx>{};
  for (auto const s : x.signature_) {
    if (s.ev_type() == x.ev_ && tr.rules_[s.rule()].is_qualified(partner_ev)) {
      rules.push_back(transfer_rule_side_idx{s.rule(), partner_ev});
    }
  }
  for (auto const l : {x.base_, tt.locations_.parents_[x.base_]}) {
    if (l == location_idx_t::invalid()) {
      continue;
    }
    for (auto const s : values_of(tr.stop_rules_, l)) {
      if (s.ev_type() == x.ev_ &&
          tr.rules_[s.rule()].is_qualified(partner_ev)) {
        rules.push_back(transfer_rule_side_idx{s.rule(), partner_ev});
      }
    }
  }
  utl::erase_duplicates(rules);

  auto partners = std::vector<rule_partner>{};
  auto const is_out = x.ev_ == event_type::kArr;
  auto const from = is_out ? x.e_.time_ : x.e_.time_ - window;
  auto const to = is_out ? x.e_.time_ + window : x.e_.time_;
  for (auto const q : rules) {
    auto const& rule = tr.rules_[q.rule()];
    for (auto d = get_day(tt, from); d <= get_day(tt, to); ++d) {
      auto const key = rule_events_key{
          rule.trip(partner_ev), rule.src_,  rule.route(partner_ev),
          rule.stop(partner_ev), partner_ev, d};
      auto const& side_events = utl::get_or_create(
          events, key, [&]() { return get_rule_events(tt, rtt, key); });
      for (auto it = std::ranges::lower_bound(side_events, from, {},
                                              &rule_event::time_);
           it != end(side_events) && it->time_ <= to; ++it) {
        auto const y = get_change_end(rtt, rt::frun::from_t(tt, &rtt, it->t_),
                                      it->stop_, partner_ev);
        if (!y.has_value() || is_same_run(x.e_, *y)) {
          continue;
        }

        auto const& y_sig = get_signature(tt, rtt, *y);
        if (!std::ranges::binary_search(y_sig, q)) {
          continue;
        }

        partners.push_back(rule_partner{
            .y_ = *y,
            .ev_ = x.ev_,
            .gap_ = get_gap(x, *y),
            .expected_ = get_expected(tt, rtt, durations, new_footpaths_out, x,
                                      y_sig, rtt.base(y->location_))});
      }
    }
  }
  return partners;
}

bool plain_partners_fit(timetable const& tt,
                        rt_timetable const& rtt,
                        duration_cache_t& durations,
                        new_footpaths_t const& new_footpaths_out,
                        change_side const& x,
                        location_idx_t const s,
                        location_idx_t const q,
                        duration_t const d,
                        duration_t const base_expected) {
  if (d == base_expected) {
    return true;
  }
  auto const lo = std::min(d, base_expected);
  auto const hi = std::max(d, base_expected);
  auto const window = x.ev_ == event_type::kArr
                          ? interval{x.e_.time_ + lo, x.e_.time_ + hi}
                          : interval{x.e_.time_ - hi + duration_t{1},
                                     x.e_.time_ - lo + duration_t{1}};
  auto is_ok = true;
  for_each_event_at(tt, rtt, q, flip(x.ev_), window, [&](change_end const& y) {
    if (is_ok && !is_same_run(x.e_, y)) {
      auto const gap = get_gap(x, y);
      is_ok =
          is_possible(gap, d) ==
          is_possible(gap, get_expected(tt, rtt, durations, new_footpaths_out,
                                        x, get_signature(tt, rtt, y), s));
    }
  });
  return is_ok;
}

bool fits(timetable const& tt,
          rt_timetable const& rtt,
          duration_cache_t& durations,
          new_footpaths_t const& new_footpaths_out,
          std::span<change_side const> sides,
          std::span<rule_partner const> partners,
          location_idx_t const l) {
  for (auto const& p : partners) {
    auto const d = get_duration(tt, rtt, durations, new_footpaths_out, p.ev_, l,
                                p.y_.location_);
    if (is_possible(p.gap_, d) != is_possible(p.gap_, p.expected_)) {
      return false;
    }
  }
  for (auto const& x : sides) {
    for (auto const s : get_partner_locations(tt, x)) {
      auto const base_expected =
          get_expected(tt, rtt, durations, new_footpaths_out, x, {}, s);
      if (get_duration(tt, rtt, durations, new_footpaths_out, x.ev_, l, s) !=
          base_expected) {
        return false;
      }
      for (auto const q : get_locations(tt, rtt, s)) {
        if (q != s &&
            !plain_partners_fit(tt, rtt, durations, new_footpaths_out, x, s, q,
                                get_duration(tt, rtt, durations,
                                             new_footpaths_out, x.ev_, l, q),
                                base_expected)) {
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
    duration_cache_t& durations,
    new_footpaths_t& new_footpaths_out,
    new_footpaths_t& new_footpaths_in,
    location_idx_t const base,
    std::span<change_side const> sides,
    std::span<rule_partner const> partners) {
  auto const v = rtt.add_rt_location(base);
  auto moved = std::vector<change_end>{};
  for (auto const ev : {event_type::kArr, event_type::kDep}) {
    auto const is_out = ev == event_type::kArr;
    auto v_durations = hash_map<location_idx_t, duration_t>{};
    for (auto const fp : routing::get_transfers(
             tt, nullptr, kDefaultProfile,
             is_out ? direction::kForward : direction::kBackward, base)) {
      auto const it = v_durations.emplace(fp.target(), fp.duration()).first;
      it->second = std::min(it->second, fp.duration());
    }
    v_durations[base] = duration_t{rtt.transfer_time(base)};
    rtt.for_each_rt_virt([&](location_idx_t const w, rt_location_idx_t) {
      if (auto const d = utl::lookup(v_durations, rtt.base(w));
          w != v && d.has_value()) {
        v_durations.emplace(w, *d);
      }
    });
    for (auto const& x : sides) {
      if (x.ev_ != ev) {
        continue;
      }
      for (auto const s : get_partner_locations(tt, x)) {
        auto const base_expected =
            get_expected(tt, rtt, durations, new_footpaths_out, x, {}, s);
        for (auto const q : get_locations(tt, rtt, s)) {
          if (q == v) {
            continue;
          }
          auto d = base_expected;
          if (q != s) {
            auto const at_q = [&](rule_partner const& p) {
              return p.ev_ == ev && p.y_.location_ == q;
            };
            auto const fit = [&](duration_t const c) {
              return utl::all_of(partners,
                                 [&](rule_partner const& p) {
                                   return !at_q(p) ||
                                          is_possible(p.gap_, c) ==
                                              is_possible(p.gap_, p.expected_);
                                 }) &&
                     plain_partners_fit(tt, rtt, durations, new_footpaths_out,
                                        x, s, q, c, base_expected);
            };
            if (!fit(base_expected)) {
              for (auto const& p : partners) {
                if (at_q(p) && fit(p.expected_)) {
                  d = p.expected_;
                  break;
                }
              }
            }
          }
          v_durations[q] = d;
        }
      }
    }
    for (auto const& p : partners) {
      if (p.ev_ != ev) {
        continue;
      }
      auto const d =
          utl::lookup(v_durations, p.y_.location_).value_or(footpath::kBan);
      if (is_possible(p.gap_, d) != is_possible(p.gap_, p.expected_)) {
        moved.push_back(p.y_);
      }
    }
    auto& out = is_out ? new_footpaths_out : new_footpaths_in;
    auto& in = is_out ? new_footpaths_in : new_footpaths_out;
    for (auto const& [l, d] : v_durations) {
      if (d != footpath::kBan) {
        out[v].emplace_back(l, d);
        in[l].emplace_back(v, d);
      }
    }
  }
  return {v, moved};
}

std::vector<stop_event> get_changed_events(timetable const& tt,
                                           rt_timetable const& rtt) {
  auto changed = std::vector<stop_event>{};
  for (auto rt_t = rt_transport_idx_t{0U}; rt_t != rtt.n_rt_transports();
       ++rt_t) {
    auto const fr = rt::frun::from_rt(tt, &rtt, rt_t);
    if (!fr.is_scheduled()) {
      continue;
    }

    auto const is_rule_transport = has_transfer_rules(tt, fr.t_.t_idx_);
    auto const seq = tt.route_location_seq_[tt.transport_route_[fr.t_.t_idx_]];
    auto const n = fr.size();
    for (auto i = stop_idx_t{0U}; i != n; ++i) {
      auto const l = rtt.stop_location(rt_t, i);
      if (!rtt.is_rt_location(l) &&
          (!tt.transfer_rules_.is_rule_location_.test(l) ||
           (!tt.locations_.is_virt(l) && !is_rule_transport))) {
        continue;
      }

      auto const stp = fr[i];
      if (l != stop{seq[i]}.location_idx() ||
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
  auto const& tr = tt.transfer_rules_;
  if (tr.rules_.empty()) {
    return;
  }
  auto const timer = scoped_timer{"rt.resolve_conflicts"};

  // Get all updated events with
  // - rules and
  // - times that differ from scheduled times
  auto queue = get_changed_events(tt, rtt);
  auto const n_changed = queue.size();

  auto events = event_cache_t{};
  auto durations = duration_cache_t{};
  auto new_footpaths_out = new_footpaths_t{};
  auto new_footpaths_in = new_footpaths_t{};

  auto const n_transports = rtt.n_rt_transports();
  auto n_checked = 0U;
  auto n_moves = 0U;
  auto n_unresolved = 0U;
  auto placements = hash_map<stop_event, std::uint32_t>{};
  auto sides = std::vector<change_side>{};
  auto partners = std::vector<rule_partner>{};
  while (!queue.empty()) {
    auto const [rt_t, i] = queue.back();
    queue.pop_back();
    ++n_checked;

    // Get arrival + departure change sides including signatures.
    auto const fr = rt::frun::from_rt(tt, &rtt, rt_t);
    sides.clear();
    for (auto const ev : {event_type::kArr, event_type::kDep}) {
      if (auto const e = get_change_end(rtt, fr, i, ev); e.has_value()) {
        sides.push_back(change_side{.e_ = *e,
                                    .ev_ = ev,
                                    .signature_ = get_signature(tt, rtt, *e),
                                    .base_ = rtt.base(e->location_)});
      }
    }
    if (sides.empty()) {
      continue;
    }

    auto const l = sides.front().e_.location_;
    auto const base = sides.front().base_;
    auto const rule_window = tr.max_rule_duration_[base];
    partners.clear();
    for (auto const& x : sides) {
      auto window = rule_window;
      for (auto const s : get_partner_locations(tt, x)) {
        window =
            std::max(window, get_duration(tt, rtt, durations, new_footpaths_out,
                                          x.ev_, x.base_, s));
      }
      auto p = get_rule_partners(tt, rtt, events, durations, new_footpaths_out,
                                 x, window);
      partners.insert(end(partners), begin(p), end(p));
    }
    if (fits(tt, rtt, durations, new_footpaths_out, sides, partners, l)) {
      continue;
    }
    auto to = location_idx_t::invalid();
    auto moved = std::vector<change_end>{};
    for (auto const c : get_locations(tt, rtt, base)) {
      if (c != l &&
          fits(tt, rtt, durations, new_footpaths_out, sides, partners, c)) {
        to = c;
        break;
      }
    }
    if (to == location_idx_t::invalid()) {
      std::tie(to, moved) =
          add_location(tt, rtt, durations, new_footpaths_out, new_footpaths_in,
                       base, sides, partners);
    }
    if (++placements[stop_event{rt_t, i}] > 3U) {
      ++n_unresolved;
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
  write_rt_footpaths(rtt.rt_footpaths_out_, new_footpaths_out);
  write_rt_footpaths(rtt.rt_footpaths_in_, new_footpaths_in);

  log(log_lvl::info, "rt.resolve_conflicts",
      "{} changed events, {} checked, {} moves, {} unresolved, {} new "
      "real-time transports, {} real-time locations",
      n_changed, n_checked, n_moves, n_unresolved,
      rtt.n_rt_transports() - n_transports, rtt.n_rt_locations());
}

}  // namespace nigiri::rt
