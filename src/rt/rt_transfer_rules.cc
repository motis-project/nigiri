#include "nigiri/rt/rt_transfer_rules.h"

#include <algorithm>
#include <array>
#include <map>
#include <optional>
#include <ranges>
#include <span>
#include <tuple>
#include <vector>

#include "utl/concat.h"
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
#include "nigiri/transfer_rules.h"

namespace nigiri::rt {

using rt_stop = pair<rt_transport_idx_t, stop_idx_t>;

struct run_event {
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
  route_id_idx_t route_;
  location_idx_t stop_;
  event_type ev_;
  source_idx_t src_;
  day_idx_t day_;
};

struct changed_event {
  run_event e_;
  event_type ev_;
  std::vector<transfer_rule_side_idx> signature_;
  location_idx_t base_;
};

struct rule_transfer {
  run_event other_;
  event_type ev_;
  duration_t gap_;
  duration_t expected_;
};

using event_cache_t = hash_map<rule_events_key, std::vector<rule_event>>;
using duration_cache_t =
    hash_map<pair<location_idx_t, location_idx_t>, duration_t>;
using rt_fps_t = std::map<location_idx_t, std::vector<footpath>>;

std::optional<run_event> get_run_event(rt_timetable const& rtt,
                                       rt::frun const& fr,
                                       stop_idx_t const i,
                                       event_type const ev) {
  if ((!fr.is_rt() && !rtt.is_transport_active(fr.t_.t_idx_, fr.t_.day_)) ||
      fr.is_cancelled() ||
      (ev == event_type::kArr ? i == 0U : i + 1U >= fr.size())) {
    return std::nullopt;  // does not exist or cancelled
  }

  auto const stp = fr[i];
  if (!(ev == event_type::kArr ? stp.out_allowed() : stp.in_allowed())) {
    return std::nullopt;  // no transfer possible
  }

  return run_event{.r_ = fr,
                   .stop_ = i,
                   .time_ = stp.time(ev),
                   .location_ = routing::search_location(kDefaultProfile, stp)};
}

bool is_same_run(run_event const& a, run_event const& b) {
  return a.r_.t_ == b.r_.t_ && a.r_.rt_ == b.r_.rt_;
}

duration_t get_duration(timetable const& tt,
                        rt_timetable const& rtt,
                        duration_cache_t& durations,
                        rt_fps_t const& rt_fps,
                        event_type const ev,
                        location_idx_t const l,
                        location_idx_t const other) {
  auto const key = ev == event_type::kArr ? pair{l, other} : pair{other, l};
  return utl::get_or_create(durations, key, [&]() {
    auto d = routing::shortest_transfer(tt, &rtt, kDefaultProfile, key.first,
                                        key.second)
                 .value_or(kNever);
    if (auto const it = rt_fps.find(key.first); it != end(rt_fps)) {
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
                                                  run_event const& x) {
  auto sig = std::vector<transfer_rule_side_idx>{};
  if (x.r_.t_.t_idx_ != transport_idx_t::invalid()) {
    get_stop_signature(tt, x.r_.t_.t_idx_, x.stop_, rtt.base(x.location_), sig);
  }
  return sig;
}

duration_t get_expected(timetable const& tt,
                        rt_timetable const& rtt,
                        duration_cache_t& durations,
                        rt_fps_t const& rt_fps,
                        changed_event const& x,
                        std::span<transfer_rule_side_idx const> other_sig,
                        location_idx_t const other_base) {
  auto const rule =
      x.ev_ == event_type::kArr
          ? get_change_rule(tt, x.signature_, other_sig, x.base_, other_base)
          : get_change_rule(tt, other_sig, x.signature_, other_base, x.base_);
  if (rule != transfer_rule_idx_t::invalid()) {
    auto const& r = tt.transfer_rules_.rules_[rule];
    return r.is_forbidden() ? kNever : r.duration_;
  }
  return get_duration(tt, rtt, durations, rt_fps, x.ev_, x.base_, other_base);
}

duration_t get_gap(changed_event const& x, run_event const& y) {
  return std::chrono::duration_cast<duration_t>(
      x.ev_ == event_type::kArr ? y.time_ - x.e_.time_ : x.e_.time_ - y.time_);
}

auto get_virts(timetable const& tt,
               rt_timetable const& rtt,
               location_idx_t const base) {
  // Static virts range (virts are at the end of children).
  auto static_virts = std::span<location_idx_t const>{};
  if (to_idx(base) < tt.locations_.children_.size()) {
    auto const children = tt.locations_.children_[base];
    auto const first_virt = std::ranges::partition_point(
        children,
        [&](location_idx_t const c) { return !tt.locations_.is_virt(c); });
    static_virts = {first_virt, children.end()};
    assert(std::ranges::all_of(static_virts, [&](location_idx_t const c) {
      return tt.locations_.is_virt(c);
    }));
  }

  // Rt-virts.
  auto const rt_virts_it = rtt.rt_virts_.find(base);
  auto const rt_virts =
      rt_virts_it == end(rtt.rt_virts_)
          ? std::span<location_idx_t const>{}
          : std::span<location_idx_t const>{rt_virts_it->second};

  // Join view over both.
  return std::array{static_virts, rt_virts} | std::views::join;
}

std::vector<location_idx_t> get_rule_partner_bases(timetable const& tt,
                                                   changed_event const& x) {
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

interval<day_idx_t> get_days(timetable const& tt,
                             interval<unixtime_t> const window,
                             duration_t const mam_first,
                             duration_t const mam_last) {
  return {
      get_day(tt, window.from_ - date::days{1} - mam_last),
      day_idx_t{to_idx(get_day(tt, window.to_ + footpath::kMax - mam_first)) +
                1U}};
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
          for (auto const d : get_days(tt, window, mam, mam)) {
            auto const tr = transport{t, d};
            if (!window.contains(tt.event_time(tr, j, ev)) ||
                rtt.resolve_rt(tr) != rt_transport_idx_t::invalid()) {
              continue;
            }
            auto const e =
                get_run_event(rtt, rt::frun::from_t(tt, &rtt, tr), j, ev);
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
          get_run_event(rtt, rt::frun::from_rt(tt, &rtt, rt_t), j, ev);
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
      for (auto const d :
           get_days(tt, window, mam(stops.from_),
                    mam(static_cast<stop_idx_t>(stops.to_ - 1U)))) {
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
                get_run_event(rtt, rt::frun::from_t(tt, &rtt, {t, d}), j, ev);
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

std::vector<rule_transfer> get_qualified_rule_transfers(
    timetable const& tt,
    rt_timetable const& rtt,
    event_cache_t& events,
    duration_cache_t& durations,
    rt_fps_t const& rt_fps,
    changed_event const& x,
    duration_t const window) {
  auto const& tr = tt.transfer_rules_;

  // The other side
  // => departure if x is an arrival / arrival if x is a departure
  auto const ev = flip(x.ev_);

  // Collect trip/route qualified rules that apply to x (x.signature_)
  // AND are also qualified on the other side.
  auto rules = std::vector<transfer_rule_side_idx>{};
  for (auto const s : x.signature_) {
    if (s.ev_type() == x.ev_ && tr.rules_[s.rule()].is_qualified(ev)) {
      rules.push_back(transfer_rule_side_idx{s.rule(), ev});
    }
  }

  // Collect: unqualified on x's stop/parent AND qualified on the other side.
  for (auto const l : {x.base_, tt.locations_.parents_[x.base_]}) {
    if (l == location_idx_t::invalid()) {
      continue;  // no parent
    }
    for (auto const s : values_of(tr.stop_rules_, l)) {
      if (s.ev_type() == x.ev_ && tr.rules_[s.rule()].is_qualified(ev)) {
        rules.push_back(transfer_rule_side_idx{s.rule(), ev});
      }
    }
  }

  utl::erase_duplicates(rules);

  // Collect all (event -> event) transfers in `window` covered by those rules.
  auto transfers = std::vector<rule_transfer>{};
  auto const is_out = x.ev_ == event_type::kArr;
  for (auto const q : rules) {
    auto const& rule = tr.rules_[q.rule()];
    auto const horizon = rule.is_forbidden() ? kNever : window;
    auto const from = is_out ? x.e_.time_ : x.e_.time_ - horizon;
    auto const to = is_out ? x.e_.time_ + horizon : x.e_.time_;
    for (auto d = get_day(tt, from); d <= get_day(tt, to); ++d) {
      auto const key = rule_events_key{
          rule.trip(ev), rule.route(ev), rule.stop(ev), ev, rule.src_, d};
      auto const& side_events = utl::get_or_create(
          events, key, [&]() { return get_rule_events(tt, rtt, key); });
      for (auto it = std::ranges::lower_bound(side_events, from, {},
                                              &rule_event::time_);
           it != end(side_events) && it->time_ <= to; ++it) {
        auto const y = get_run_event(rtt, rt::frun::from_t(tt, &rtt, it->t_),
                                     it->stop_, ev);
        if (!y.has_value() || is_same_run(x.e_, *y)) {
          continue;
        }

        auto const& y_sig = get_signature(tt, rtt, *y);
        if (!std::ranges::binary_search(y_sig, q)) {
          continue;
        }

        transfers.push_back(rule_transfer{
            .other_ = *y,
            .ev_ = x.ev_,
            .gap_ = get_gap(x, *y),
            // This might not be the most specific rule for this transfer
            // -> compute expected
            .expected_ = get_expected(tt, rtt, durations, rt_fps, x, y_sig,
                                      rtt.base(y->location_)),
        });
      }
    }
  }

  return transfers;
}

window_t get_window(timetable const& tt,
                    rt_timetable const& rtt,
                    changed_event const& x,
                    std::span<rule_transfer const> transfers,
                    location_idx_t const other,
                    duration_t const base_expected,
                    interval<duration_t> const gaps) {
  auto w = window_t{duration_t{0}, kNever + duration_t{1}};
  auto has_never = base_expected == kNever;

  // Narrow transfer window with rules that are qualified on the partner side.
  for (auto const& p : transfers) {
    if (p.ev_ == x.ev_ && p.other_.location_ == other) {
      narrow(w, p.gap_, p.expected_);
      has_never |= p.expected_ == kNever;
    }
  }
  if (!has_never) {
    w.to_ = std::min(w.to_, footpath::kBan);
  }

  // Iterate all events within `gaps` at `other` to cover partners no  qualified
  // rule names.
  auto const to = std::min(gaps.to_, footpath::kBan);
  if (gaps.from_ >= to) {
    return w;
  }
  for_each_event_at(tt, rtt, other, flip(x.ev_),
                    x.ev_ == event_type::kArr
                        ? interval{x.e_.time_ + gaps.from_, x.e_.time_ + to}
                        : interval{x.e_.time_ - to + duration_t{1},
                                   x.e_.time_ - gaps.from_ + duration_t{1}},
                    [&](run_event const& y) {
                      if (!is_same_run(x.e_, y) &&
                          utl::none_of(transfers, [&](rule_transfer const& p) {
                            return p.ev_ == x.ev_ &&
                                   p.other_.stop_ == y.stop_ &&
                                   is_same_run(p.other_, y);
                          })) {
                        narrow(w, get_gap(x, y), base_expected);
                      }
                    });
  return w;
}

bool can_join(timetable const& tt,
              rt_timetable const& rtt,
              duration_cache_t& durations,
              rt_fps_t const& rt_fps,
              std::span<changed_event const> stop_events,
              std::span<rule_transfer const> transfers,
              location_idx_t const l) {
  for (auto const& x : stop_events) {
    for (auto const other_base : get_rule_partner_bases(tt, x)) {
      auto const base_expected =
          get_expected(tt, rtt, durations, rt_fps, x, {}, other_base);
      auto const fits = [&](location_idx_t const other) {
        auto const d =
            get_duration(tt, rtt, durations, rt_fps, x.ev_, l, other);
        return get_window(
                   tt, rtt, x, transfers, other, base_expected,
                   {std::min(d, base_expected), std::max(d, base_expected)})
            .contains(d);
      };
      if (get_duration(tt, rtt, durations, rt_fps, x.ev_, l, other_base) !=
              base_expected ||
          !fits(other_base) ||
          !utl::all_of(get_virts(tt, rtt, other_base), fits)) {
        return false;
      }
    }
  }
  return true;
}

std::pair<location_idx_t, std::vector<run_event>> add_location(
    timetable const& tt,
    rt_timetable& rtt,
    duration_cache_t& durations,
    rt_fps_t& rt_fps,
    rt_fps_t& rt_fps_in,
    location_idx_t const base,
    std::span<changed_event const> stop_events,
    std::span<rule_transfer const> transfers) {
  auto const v = rtt.add_rt_location(base);
  auto moved = std::vector<run_event>{};
  for (auto const ev : {event_type::kArr, event_type::kDep}) {
    auto const is_out = ev == event_type::kArr;
    auto v_durations = hash_map<location_idx_t, duration_t>{};
    for (auto const fp : routing::get_transfers(
             tt, nullptr, kDefaultProfile,
             is_out ? direction::kForward : direction::kBackward, base)) {
      auto const it = v_durations.emplace(fp.target(), fp.duration()).first;
      it->second = std::min(it->second, fp.duration());
    }
    auto const transfer_time = rtt.transfer_time(base);
    v_durations[base] = transfer_time == kNoTransferAllowed
                            ? kNever
                            : duration_t{transfer_time.count()};
    rtt.for_each_rt_virt([&](location_idx_t const w, rt_location_idx_t) {
      if (auto const d = utl::lookup(v_durations, rtt.base(w));
          w != v && d.has_value()) {
        v_durations.emplace(w, *d);
      }
    });
    for (auto const& x : stop_events) {
      if (x.ev_ != ev) {
        continue;
      }
      for (auto const other_base : get_rule_partner_bases(tt, x)) {
        auto const base_expected =
            get_expected(tt, rtt, durations, rt_fps, x, {}, other_base);
        v_durations[other_base] = base_expected;
        for (auto const other : get_virts(tt, rtt, other_base)) {
          if (other != v) {
            v_durations[other] =
                pick_duration(
                    get_window(tt, rtt, x, transfers, other, base_expected,
                               {duration_t{0}, footpath::kBan}),
                    base_expected)
                    .value_or(base_expected);
          }
        }
      }
    }
    for (auto const& p : transfers) {
      if (p.ev_ != ev) {
        continue;
      }
      auto const d =
          utl::lookup(v_durations, p.other_.location_).value_or(kNever);
      if ((p.gap_ >= d) != (p.gap_ >= p.expected_)) {
        moved.push_back(p.other_);
      }
    }
    auto& out = is_out ? rt_fps : rt_fps_in;
    auto& in = is_out ? rt_fps_in : rt_fps;
    for (auto const& [l, d] : v_durations) {
      if (d != kNever) {
        out[v].emplace_back(l, d);
        in[l].emplace_back(v, d);
      }
    }
  }
  return {v, moved};
}

std::vector<rt_stop> get_changed_events(timetable const& tt,
                                        rt_timetable const& rtt) {
  auto changed = std::vector<rt_stop>{};
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
  auto rt_fps = rt_fps_t{};
  auto rt_fps_in = rt_fps_t{};

  auto const n_transports = rtt.n_rt_transports();
  auto n_checked = 0U;
  auto n_moves = 0U;
  auto n_unresolved = 0U;
  auto placements = hash_map<rt_stop, std::uint32_t>{};
  auto stop_events = std::vector<changed_event>{};
  auto transfers = std::vector<rule_transfer>{};
  while (!queue.empty()) {
    auto const [rt_t, i] = queue.back();
    queue.pop_back();
    ++n_checked;

    // Get arrival + departure events including signatures.
    auto const fr = rt::frun::from_rt(tt, &rtt, rt_t);
    stop_events.clear();
    for (auto const ev : {event_type::kArr, event_type::kDep}) {
      if (auto const e = get_run_event(rtt, fr, i, ev); e.has_value()) {
        stop_events.push_back(
            changed_event{.e_ = *e,
                          .ev_ = ev,
                          .signature_ = get_signature(tt, rtt, *e),
                          .base_ = rtt.base(e->location_)});
      }
    }
    if (stop_events.empty()) {
      continue;
    }

    auto const l = stop_events.front().e_.location_;
    auto const base = stop_events.front().base_;
    auto const rule_window = tr.max_rule_duration_[base];
    transfers.clear();
    for (auto const& x : stop_events) {
      auto window = rule_window;
      for (auto const other_base : get_rule_partner_bases(tt, x)) {
        window = std::max(window, get_duration(tt, rtt, durations, rt_fps,
                                               x.ev_, x.base_, other_base));
      }
      utl::concat(transfers,
                  get_qualified_rule_transfers(tt, rtt, events, durations,
                                               rt_fps, x, window));
    }

    // Check if stop events can stay where they are.
    if (can_join(tt, rtt, durations, rt_fps, stop_events, transfers, l)) {
      continue;
    }
    if (++placements[rt_stop{rt_t, i}] > 3U) {
      ++n_unresolved;
      continue;
    }

    // Conflict -> check if stop events can join any existing virt location.
    auto const can_move_to = [&](location_idx_t const c) {
      return c != l &&
             can_join(tt, rtt, durations, rt_fps, stop_events, transfers, c);
    };
    auto virts = get_virts(tt, rtt, base);
    auto const it = std::ranges::find_if(virts, can_move_to);
    auto to = can_move_to(base)   ? base
              : it != virts.end() ? *it
                                  : location_idx_t::invalid();
    auto moved = std::vector<run_event>{};
    if (to == location_idx_t::invalid()) {
      std::tie(to, moved) = add_location(tt, rtt, durations, rt_fps, rt_fps_in,
                                         base, stop_events, transfers);
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
  write_rt_footpaths(rtt.rt_footpaths_out_, rt_fps);
  write_rt_footpaths(rtt.rt_footpaths_in_, rt_fps_in);

  log(log_lvl::info, "rt.resolve_conflicts",
      "{} changed events, {} checked, {} moves, {} unresolved, {} new "
      "real-time transports, {} real-time locations",
      n_changed, n_checked, n_moves, n_unresolved,
      rtt.n_rt_transports() - n_transports, rtt.n_rt_locations());
}

}  // namespace nigiri::rt
