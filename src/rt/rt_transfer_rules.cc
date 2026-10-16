#include "nigiri/rt/rt_transfer_rules.h"

#include <cassert>
#include <algorithm>
#include <span>
#include <vector>

#include "utl/erase_duplicates.h"
#include "utl/get_or_create.h"
#include "utl/helpers/algorithm.h"
#include "utl/lookup.h"
#include "utl/to_vec.h"

#include "nigiri/footpath.h"
#include "nigiri/logging.h"
#include "nigiri/routing/search_location.h"
#include "nigiri/routing/transfers.h"
#include "nigiri/rt/frun.h"
#include "nigiri/transfer_rule_sides.h"

namespace nigiri::rt {

using side_t = transfer_rule_side_idx;

void add_rt_transfers(timetable const& tt,
                      rt_timetable& rtt,
                      location_idx_t const v) {
  auto const& tr = tt.transfer_rules_;
  auto const& loc = tt.locations_;
  auto const p = rtt.base(v);
  auto const change_time = to_fp_duration(loc.transfer_time_[p]);

  auto const for_each_location = [&](side_t const side, auto&& fn) {
    for_each_side_location(tt, side, fn);
    rtt.for_each_rt_virt([&](location_idx_t const w,
                             rt_location_idx_t const i) {
      auto const rules = rtt.rt_virt_rules_[i];
      if (is_applicable(tt, side, {begin(rules), end(rules)}, rtt.base(w))) {
        fn(w);
      }
    });
  };

  for (auto const dir : {direction::kForward, direction::kBackward}) {
    auto const ev =
        dir == direction::kForward ? event_type::kArr : event_type::kDep;

    auto durations = hash_map<location_idx_t, duration_t>{};
    for (auto const fp :
         routing::get_transfers(tt, nullptr, kDefaultProfile, dir, p)) {
      auto const it = durations.emplace(fp.target(), fp.duration()).first;
      it->second = std::min(it->second, fp.duration());
    }
    durations.erase(p);
    if (change_time != footpath::kMaxDuration) {
      durations.emplace(p, change_time);
    }
    rtt.for_each_rt_virt([&](location_idx_t const w, rt_location_idx_t) {
      if (auto const d = utl::lookup(durations, rtt.base(w));
          w != v && d.has_value()) {
        durations.emplace(w, *d);
      }
    });

    auto most_specific = hash_map<location_idx_t, transfer_rule_idx_t>{};
    auto const compete = [&](side_t const s) {
      if (s.ev_type() != ev) {
        return;
      }
      for_each_location(
          side_t{s.rule(), flip(ev)}, [&](location_idx_t const y) {
            if (y != v) {
              auto const [it, is_new] = most_specific.emplace(y, s.rule());
              if (!is_new) {
                it->second = get_more_specific(tt, it->second, s.rule());
              }
            }
          });
    };
    for (auto const s : rtt.rt_virt_rules_[rtt.to_rt_location(v)]) {
      compete(s);
    }
    for (auto const stop : {p, loc.parents_[p]}) {
      if (stop != location_idx_t::invalid()) {
        for (auto const s : values_of(tr.stop_rules_, stop)) {
          compete(s);
        }
      }
    }

    for (auto const& [y, rule] : most_specific) {
      if (auto const d = tr.rules_[rule].duration_;
          d == footpath::kMaxDuration) {
        durations.erase(y);
      } else {
        durations[y] = d;
      }
    }
    auto& out =
        ev == event_type::kArr ? rtt.rt_footpaths_out_ : rtt.rt_footpaths_in_;
    auto& in =
        ev == event_type::kArr ? rtt.rt_footpaths_in_ : rtt.rt_footpaths_out_;
    for (auto const& [y, d] : durations) {
      out[v].emplace_back(y, d);
      in[y].emplace_back(v, d);
    }
  }
}

location_idx_t get_or_create_location(timetable const& tt,
                                      rt_timetable& rtt,
                                      location_idx_t const base,
                                      std::span<side_t const> sides) {
  assert(!tt.locations_.is_virt(base));
  if (sides.empty()) {
    return base;
  }
  auto found = location_idx_t::invalid();
  tt.locations_.for_each_virt(base, [&](location_idx_t const c) {
    if (std::ranges::equal(values_of(tt.transfer_rules_.virt_rules_, c),
                           sides)) {
      found = c;
    }
  });
  rtt.for_each_rt_virt([&](location_idx_t const w, rt_location_idx_t const i) {
    if (rtt.base(w) == base &&
        std::ranges::equal(rtt.rt_virt_rules_[i], sides)) {
      found = w;
    }
  });
  if (found != location_idx_t::invalid()) {
    return found;
  }
  auto const v = rtt.add_rt_location(
      base, to_transfer_time(get_transfer_time(tt, sides, base)), sides);
  add_rt_transfers(tt, rtt, v);
  return v;
}

location_idx_t get_or_create_location(timetable const& tt,
                                      rt_timetable& rtt,
                                      rt_transport_idx_t const rt_t,
                                      stop_idx_t const stop_idx,
                                      location_idx_t const base) {
  auto const t = rtt.resolve_static(rt_t).t_idx_;
  if (tt.transfer_rules_.empty() || t == transport_idx_t::invalid()) {
    return base;
  }
  auto sig = std::vector<side_t>{};
  get_stop_signature(tt, t, stop_idx, base, sig);
  return get_or_create_location(tt, rtt, base, sig);
}

namespace {

struct change_end {
  rt::run r_;
  stop_idx_t stop_;
  unixtime_t time_;
  duration_t delay_;
  location_idx_t location_;
  std::vector<side_t> signature_;
  std::vector<side_t> sides_;
};

struct move {
  change_end x_;
  location_idx_t to_;
};

struct event {
  unixtime_t time_;
  transport t_;
  stop_idx_t stop_;
};

struct signature_key {
  CISTA_COMPARABLE()
  transport_idx_t t_;
  stop_idx_t stop_;
  location_idx_t base_;
};

struct visit {
  CISTA_COMPARABLE()
  transport t_;
  stop_idx_t stop_;
  location_idx_t location_;
};

struct state {
  interval<unixtime_t> horizon_;
  vector_map<location_idx_t, duration_t> rule_times_;
  hash_map<side_t, std::vector<event>> events_;
  hash_map<signature_key, std::vector<side_t>> signatures_;
  hash_map<location_idx_t, std::vector<side_t>> sides_;
  hash_map<pair<location_idx_t, location_idx_t>, duration_t> change_times_;
  hash_set<visit> visited_;
  std::uint32_t n_unresolved_{0U};
};

using stop_event = pair<rt_transport_idx_t, stop_idx_t>;

rt::run get_run(rt_timetable const& rtt, rt_transport_idx_t const rt_t) {
  return {.t_ = rtt.resolve_static(rt_t),
          .stop_range_ = {stop_idx_t{0U},
                          static_cast<stop_idx_t>(
                              rtt.rt_transport_location_seq_[rt_t].size())},
          .rt_ = rt_t};
}

rt::run get_run(timetable const& tt,
                rt_timetable const& rtt,
                transport const t) {
  return {
      .t_ = t,
      .stop_range_ =
          {stop_idx_t{0U},
           static_cast<stop_idx_t>(
               tt.route_location_seq_[tt.transport_route_[t.t_idx_]].size())},
      .rt_ = rtt.resolve_rt(t)};
}

std::optional<change_end> get_change_end(timetable const& tt,
                                         rt_timetable const& rtt,
                                         rt::run const& r,
                                         stop_idx_t const i,
                                         event_type const ev) {
  if (!r.is_rt() && !rtt.is_transport_active(r.t_.t_idx_, r.t_.day_)) {
    return std::nullopt;
  }
  auto const fr = rt::frun{tt, &rtt, r};
  if (fr.is_cancelled() ||
      (ev == event_type::kArr ? i == 0U : i + 1U >= fr.size())) {
    return std::nullopt;
  }
  auto const stp = fr[i];
  if (!(ev == event_type::kArr ? stp.out_allowed() : stp.in_allowed())) {
    return std::nullopt;
  }
  return change_end{
      .r_ = r,
      .stop_ = i,
      .time_ = stp.time(ev),
      .delay_ =
          std::max(duration_t{0}, std::chrono::duration_cast<duration_t>(
                                      stp.time(ev) - stp.scheduled_time(ev))),
      .location_ = routing::search_location(kDefaultProfile, stp),
      .signature_ = {},
      .sides_ = {}};
}

void add_sides(timetable const& tt,
               rt_timetable const& rtt,
               state& st,
               change_end& x) {
  auto const l = x.location_;
  x.sides_ = utl::get_or_create(st.sides_, l, [&]() {
    return rtt.is_rt_location(l)
               ? utl::to_vec(rtt.rt_virt_rules_[rtt.to_rt_location(l)])
               : utl::to_vec(values_of(tt.transfer_rules_.virt_rules_, l));
  });
  auto const t = x.r_.t_.t_idx_;
  if (t != transport_idx_t::invalid() &&
      x.stop_ < tt.route_location_seq_[tt.transport_route_[t]].size()) {
    auto const key = signature_key{t, x.stop_, rtt.base(l)};
    x.signature_ = utl::get_or_create(st.signatures_, key, [&]() {
      auto sig = std::vector<side_t>{};
      get_stop_signature(tt, t, x.stop_, key.base_, sig);
      return sig;
    });
  }
}

std::vector<move> get_moves(timetable const& tt,
                            rt_timetable& rtt,
                            state& st,
                            change_end const& a,
                            change_end const& b) {
  auto const gap = b.time_ - a.time_;
  if (gap < duration_t{0} || (a.r_.t_ == b.r_.t_ && a.r_.rt_ == b.r_.rt_)) {
    return {};
  }
  auto const change_time = [&](location_idx_t const from,
                               location_idx_t const to) {
    return utl::get_or_create(st.change_times_, pair{from, to}, [&]() {
      return routing::shortest_transfer(tt, &rtt, kDefaultProfile, from, to)
          .value_or(footpath::kMaxDuration);
    });
  };
  auto const own_sides = [](change_end const& x) {
    auto own = std::vector<side_t>{};
    std::ranges::set_intersection(x.sides_, x.signature_,
                                  std::back_inserter(own));
    return own;
  };

  auto const& rules = tt.transfer_rules_.rules_;
  auto const base_a = rtt.base(a.location_);
  auto const base_b = rtt.base(b.location_);
  auto const expected =
      get_change_rule(tt, a.signature_, b.signature_, base_a, base_b);
  auto const expected_time = expected == transfer_rule_idx_t::invalid()
                                 ? change_time(base_a, base_b)
                                 : rules[expected].duration_;
  auto const actual_time = change_time(a.location_, b.location_);
  auto const is_possible = [&](duration_t const d) {
    return d < footpath::kMaxDuration && gap >= d;
  };
  if (is_possible(expected_time) == is_possible(actual_time)) {
    return {};
  }

  auto moves = std::vector<move>{};
  auto const add = [&](change_end const& x, std::vector<side_t> const& sides) {
    auto const to =
        get_or_create_location(tt, rtt, rtt.base(x.location_), sides);
    if (to != x.location_ &&
        !st.visited_.contains(visit{x.r_.t_, x.stop_, to})) {
      moves.push_back({x, to});
    }
  };
  if (expected != transfer_rule_idx_t::invalid()) {
    for (auto const& [x, ev] :
         {pair{&a, event_type::kArr}, pair{&b, event_type::kDep}}) {
      auto const side = side_t{expected, ev};
      if (rules[expected].is_qualified(ev) &&
          !std::ranges::binary_search(x->sides_, side)) {
        auto sides = own_sides(*x);
        sides.insert(std::ranges::upper_bound(sides, side), side);
        add(*x, sides);
      }
    }
  }
  for (auto const* x : {&a, &b}) {
    if (auto const own = own_sides(*x);
        moves.empty() && own.size() != x->sides_.size()) {
      add(*x, own);
    }
  }
  if (moves.empty()) {
    ++st.n_unresolved_;
  }
  return moves;
}

std::vector<event> get_events(timetable const& tt,
                              rt_timetable const& rtt,
                              interval<unixtime_t> const horizon,
                              side_t const q) {
  auto const& tr = tt.transfer_rules_;
  auto const& rule = tr.rules_[q.rule()];
  auto const ev = q.ev_type();
  auto events = std::vector<event>{};
  auto const day = [&](unixtime_t const t) {
    auto const& iv = tt.internal_interval();
    return tt.day_idx_mam(std::clamp(t, iv.from_, iv.to_ - duration_t{1}))
        .first;
  };
  auto const add = [&](transport_idx_t const t, stop_idx_t const j) {
    auto const n = tt.route_location_seq_[tt.transport_route_[t]].size();
    if (ev == event_type::kArr ? j == 0U : j + 1U == n) {
      return;
    }
    auto const mam = tt.event_mam(t, j, ev).as_duration();
    for (auto d = day(horizon.from_ - date::days{1} - mam);
         d <= day(horizon.to_ - mam); ++d) {
      auto const e = get_change_end(tt, rtt, get_run(tt, rtt, {t, d}), j, ev);
      if (e.has_value() && horizon.contains(e->time_)) {
        events.push_back({e->time_, {t, d}, j});
      }
    }
  };

  auto const add_trip = [&](trip_idx_t const trip) {
    for (auto const [t, stops] : tt.trip_transport_ranges_[trip]) {
      auto const seq = tt.route_location_seq_[tt.transport_route_[t]];
      for (auto const j : stops) {
        if (tt.locations_.is_self_or_parent(
                rule.stop(ev), tt.base(stop{seq[j]}.location_idx()))) {
          add(t, j);
        }
      }
    }
  };
  if (auto const rule_trip = rule.trip(ev);
      rule_trip != trip_idx_t::invalid()) {
    add_trip(rule_trip);
  } else if (rule.route(ev) == route_id_idx_t::invalid()) {
    for (auto const& [trip, s] : tr.trip_rules_) {  // trip set (HRDF)
      if (s == q) {
        add_trip(trip);
      }
    }
  } else {
    for (auto const trip :
         tt.route_ids_[rule.src_].route_id_trips_[rule.route(ev)]) {
      add_trip(trip);
    }
  }
  for (auto const v : values_of(tr.rule_virts_, q)) {
    for (auto const r : tt.location_routes_[v]) {
      auto const seq = tt.route_location_seq_[r];
      for (auto j = stop_idx_t{0U}; j != seq.size(); ++j) {
        if (stop{seq[j]}.location_idx() == v) {
          for (auto const t : tt.route_transport_ranges_[r]) {
            add(t, j);
          }
        }
      }
    }
  }

  std::ranges::sort(events, {}, &event::time_);
  return events;
}

std::vector<move> find_conflict(timetable const& tt,
                                rt_timetable& rtt,
                                state& st,
                                change_end const& x,
                                event_type const ev) {
  auto const& tr = tt.transfer_rules_;
  auto const partner_ev = flip(ev);
  auto const is_arr = ev == event_type::kArr;
  auto const base = rtt.base(x.location_);
  auto const parent = tt.locations_.parents_[base];
  auto window =
      std::max(st.rule_times_[base], parent == location_idx_t::invalid()
                                         ? duration_t{0}
                                         : st.rule_times_[parent]);
  for (auto const l : {x.location_, base}) {
    for (auto const fp : routing::get_transfers(
             tt, &rtt, kDefaultProfile,
             is_arr ? direction::kForward : direction::kBackward, l)) {
      window = std::max(window, fp.duration());
    }
    // where changes are banned, an exception holds at any gap
    auto const t = routing::get_transfer_time(tt, &rtt, l);
    window = std::max(window, t == kNoTransferAllowed ? footpath::kMaxDuration
                                                      : duration_t{t.count()});
  }
  auto const from = is_arr ? x.time_ : x.time_ - window - x.delay_;
  auto const to = is_arr ? x.time_ + window : x.time_;

  auto relevant = x.signature_;
  relevant.insert(end(relevant), begin(x.sides_), end(x.sides_));
  utl::erase_duplicates(relevant);
  for (auto const s : relevant) {
    if (s.ev_type() != ev || !tr.rules_[s.rule()].is_qualified(partner_ev)) {
      continue;
    }
    auto const q = side_t{s.rule(), partner_ev};
    auto const& events = utl::get_or_create(
        st.events_, q, [&]() { return get_events(tt, rtt, st.horizon_, q); });
    for (auto it = std::ranges::lower_bound(events, from, {}, &event::time_);
         it != end(events) && it->time_ <= to; ++it) {
      auto y = get_change_end(tt, rtt, get_run(tt, rtt, it->t_), it->stop_,
                              partner_ev);
      if (!y.has_value()) {
        continue;
      }
      add_sides(tt, rtt, st, *y);
      if (!std::ranges::binary_search(y->signature_, q) &&
          !std::ranges::binary_search(y->sides_, q)) {
        continue;
      }
      if (auto moves = is_arr ? get_moves(tt, rtt, st, x, *y)
                              : get_moves(tt, rtt, st, *y, x);
          !moves.empty()) {
        return moves;
      }
    }
  }
  return {};
}

std::vector<stop_event> get_changed_stops(
    timetable const& tt,
    rt_timetable const& rtt,
    vector_map<location_idx_t, duration_t> const& rule_times) {
  auto const is_rule_stop = [&](location_idx_t const l) {
    auto const parent = tt.locations_.parents_[l];
    return rule_times[l] != duration_t{0} ||
           (parent != location_idx_t::invalid() &&
            rule_times[parent] != duration_t{0});
  };
  auto changed = std::vector<stop_event>{};
  for (auto rt_t = rt_transport_idx_t{0U}; rt_t != rtt.n_rt_transports();
       ++rt_t) {
    auto const fr = rt::frun{tt, &rtt, get_run(rtt, rt_t)};
    if (!fr.is_scheduled()) {
      continue;
    }
    auto const is_rule_transport = has_transfer_rules(tt, fr.t_.t_idx_);
    auto const seq = tt.route_location_seq_[tt.transport_route_[fr.t_.t_idx_]];
    for (auto i = stop_idx_t{0U}; i != fr.size(); ++i) {
      auto const stp = fr[i];
      auto const l = rtt.stop_location(rt_t, i);
      if (((is_rule_transport && is_rule_stop(rtt.base(l))) ||
           rtt.is_rt_location(l) || tt.locations_.is_virt(l)) &&
          (i >= seq.size() || l != stop{seq[i]}.location_idx() ||
           (i != 0U && stp.time(event_type::kArr) !=
                           stp.scheduled_time(event_type::kArr)) ||
           (i + 1U != fr.size() && stp.time(event_type::kDep) !=
                                       stp.scheduled_time(event_type::kDep)))) {
        changed.emplace_back(rt_t, i);
      }
    }
  }
  return changed;
}

}  // namespace

void resolve_conflicts(timetable const& tt, rt_timetable& rtt) {
  if (tt.transfer_rules_.rules_.empty()) {
    return;
  }
  auto const timer = scoped_timer{"rt.resolve_conflicts"};

  auto st = state{};
  st.rule_times_.resize(tt.n_locations(), duration_t{0});
  for (auto const& r : tt.transfer_rules_.rules_) {
    // A ban holds at any gap; a timed rule (0 min) decides changes too.
    auto const d = std::max(r.duration_, duration_t{1});
    for (auto const l : {r.from_stop_, r.to_stop_}) {
      if (l != location_idx_t::invalid()) {  // global rules: not in RT
        st.rule_times_[l] = std::max(st.rule_times_[l], d);
      }
    }
  }

  auto queue = get_changed_stops(tt, rtt, st.rule_times_);
  auto const n_changed = queue.size();
  st.horizon_ = interval{unixtime_t::max(), unixtime_t::min()};
  for (auto const [rt_t, i] : queue) {
    auto const t = rtt.unix_event_time(
        rt_t, i, i == 0U ? event_type::kDep : event_type::kArr);
    st.horizon_.from_ = std::min(st.horizon_.from_, t - footpath::kMaxDuration);
    st.horizon_.to_ = std::max(st.horizon_.to_, t + footpath::kMaxDuration);
  }

  auto const n_transports = rtt.n_rt_transports();
  auto n_checked = 0U;
  auto n_moves = 0U;
  while (!queue.empty()) {
    auto const [rt_t, i] = queue.back();
    queue.pop_back();
    ++n_checked;
    for (auto const ev : {event_type::kArr, event_type::kDep}) {
      auto x = get_change_end(tt, rtt, get_run(rtt, rt_t), i, ev);
      if (!x.has_value()) {
        continue;
      }
      add_sides(tt, rtt, st, *x);
      auto const moves = find_conflict(tt, rtt, st, *x, ev);
      n_moves += moves.size();
      for (auto const& [y, to] : moves) {
        auto const moved =
            y.r_.is_rt() ? y.r_.rt_
                         : rtt.add_rt_transport(
                               rt::frun{tt, &rtt, y.r_}.id().src_, tt, y.r_.t_);
        st.visited_.emplace(visit{y.r_.t_, y.stop_, to});
        rtt.update_stop_location(moved, y.stop_, to);
        queue.emplace_back(moved, y.stop_);
      }
      if (!moves.empty()) {
        queue.emplace_back(rt_t, i);
        break;
      }
    }
  }
  log(log_lvl::info, "rt.resolve_conflicts",
      "{} changed stops, {} checked, {} moves, {} unresolved, {} new "
      "real-time transports, {} real-time locations",
      n_changed, n_checked, n_moves, st.n_unresolved_,
      rtt.n_rt_transports() - n_transports, rtt.n_rt_locations());
}

}  // namespace nigiri::rt
