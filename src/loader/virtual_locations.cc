#include "nigiri/loader/virtual_locations.h"

#include <algorithm>
#include <iterator>
#include <limits>
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

#include "nigiri/loader/transfer_rules.h"
#include "nigiri/logging.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/transfers.h"
#include "nigiri/shapes_storage.h"
#include "nigiri/stop.h"
#include "nigiri/timetable.h"
#include "nigiri/transfer_rule_sides.h"

namespace nigiri::loader {

namespace {
// Index within a route.
using route_transport_idx_t =
    cista::strong<std::uint32_t, struct route_transport_>;
using sides_t = std::vector<transfer_rule_side_idx>;
using stop_seq_t = std::vector<stop::value_type>;

constexpr auto const kNever = i32_minutes{routing::kMaxTravelTime};
constexpr auto const kNone = std::numeric_limits<std::uint32_t>::max();

struct route_stop {
  CISTA_COMPARABLE()
  route_idx_t route_;
  stop_idx_t idx_;
};

// Transports of a route stop with the same signature and facts: the sides they
// need and the sides they must not inherit. Their group (the sides they need
// in any case) only shares locations with the same group.
struct member {
  CISTA_COMPARABLE()
  std::uint32_t signature_{0U};
  std::uint32_t group_{0U};
  sides_t required_, forbidden_;
};

struct stop_state {
  std::uint32_t location_{0U};
  vector_map<route_transport_idx_t, std::uint32_t> member_;
  std::vector<member> members_;
  std::vector<sides_t> sides_;  // per route part: the sides of its location
  std::vector<sides_t> required_;  // per route part: the sides it needs
  std::vector<sides_t> forbidden_;  // per route part: sides it must not have
  std::vector<sides_t> unshared_;  // per route part: sides it must not share
  std::vector<std::pair<i32_minutes, route_transport_idx_t>> departures_;
  std::uint64_t changed_{1U};  // checked in the first pass
};

// The transports of a route part share the location at each stop: a route
// with more than one part is split.
struct route_state {
  hash_map<stop_idx_t, stop_state> stops_;
  std::vector<std::uint32_t> part_;
  std::uint32_t n_parts_{1U};
};

// The signatures and route stops at a stop location.
struct location_state {
  location_idx_t l_;
  std::vector<sides_t> signatures_;
  std::vector<route_stop> stops_;
};

struct state {
  vector_map<route_idx_t, route_state> routes_;
  std::vector<location_state> locations_;
};

struct minutes_range {
  void extend(i32_minutes const m) {
    lo_ = std::min(lo_, m);
    hi_ = std::max(hi_, m);
  }
  i32_minutes lo_, hi_;
};

i32_minutes get_minutes(transfer_rule const& r) {
  return r.duration_ == footpath::kMaxDuration ? kNever
                                               : i32_minutes{r.duration_};
}

bool intersects(sides_t const& a, sides_t const& b) {
  for (auto i = begin(a), j = begin(b); i != end(a) && j != end(b);) {
    if (*i < *j) {
      ++i;
    } else if (*j < *i) {
      ++j;
    } else {
      return true;
    }
  }
  return false;
}

void unite(sides_t& a, sides_t const& b) {
  a.insert(end(a), begin(b), end(b));
  utl::erase_duplicates(a);
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

state get_state(timetable const& tt) {
  auto const& tr = tt.transfer_rules_;
  auto const& parents = tt.locations_.parents_;
  auto const global_rules = get_global_rules(tt);

  // the sides of a section's trips per stop and event type
  using section_sides_t = hash_map<pair<location_idx_t, event_type>, sides_t>;
  auto by_section = hash_map<merged_trips_idx_t, section_sides_t>{};
  auto const get_sides = [&](merged_trips_idx_t const m) -> auto const& {
    return utl::get_or_create(by_section, m, [&]() {
      auto sides = section_sides_t{};
      auto const add = [&](transfer_rule_side_idx const s) {
        sides[pair{tr.rules_[s.rule()].stop(s.ev_type()), s.ev_type()}]
            .push_back(s);
      };
      for (auto const trip : tt.merged_trips_[m]) {
        for (auto const s : values_of(tr.trip_rules_, trip)) {
          add(s);
        }
        auto const src = tt.trip_id_src_[tt.trip_ids_[trip].front()];
        for (auto const s :
             values_of(tr.route_rules_, tt.trip_route_id_[trip])) {
          if (tr.rules_[s.rule()].src_ == src) {
            add(s);
          }
        }
      }
      return sides;
    });
  };

  auto st = state{};
  st.routes_.resize(tt.n_routes());
  auto location_idx = hash_map<location_idx_t, std::uint32_t>{};
  auto signatures = std::vector<hash_map<sides_t, std::uint32_t>>{};
  auto const get_signature = [&](std::uint32_t const l, sides_t const& s) {
    return utl::get_or_create(signatures[l], s, [&]() {
      st.locations_[l].signatures_.push_back(s);
      return static_cast<std::uint32_t>(st.locations_[l].signatures_.size() -
                                        1U);
    });
  };

  auto sig = sides_t{};
  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    auto const transports = tt.route_transport_ranges_[r];
    auto const seq = tt.route_location_seq_[r];
    auto& route = st.routes_[r];
    route.part_.resize(transports.size(), 0U);
    for (auto const t : transports) {
      if (!has_transfer_rules(tt, t)) {
        continue;
      }
      by_section.clear();
      auto const sections = tt.transport_to_trip_section_[t];
      for (auto i = stop_idx_t{0U}; i != seq.size(); ++i) {
        auto const stp = stop{seq[i]};
        auto const base = stp.location_idx();
        auto const globals =
            global_rules.find(tt.locations_.get_root_idx(base));
        auto const add = [&](stop_idx_t const section, event_type const ev) {
          auto const& sides =
              get_sides(sections[sections.size() == 1U ? 0U : section]);
          for (auto const l : {base, parents[base]}) {
            if (auto const it = sides.find(pair{l, ev});
                l != location_idx_t::invalid() && it != end(sides)) {
              sig.insert(end(sig), begin(it->second), end(it->second));
            }
          }
          if (auto const it = sides.find(pair{location_idx_t::invalid(), ev});
              it != end(sides) && globals != end(global_rules)) {
            for (auto const s : it->second) {
              if (std::ranges::binary_search(globals->second, s.rule())) {
                sig.push_back(s);
              }
            }
          }
        };
        sig.clear();
        if (i != 0U && stp.out_allowed()) {
          add(static_cast<stop_idx_t>(i - 1U), event_type::kArr);
        }
        if (i + 1U != seq.size() && stp.in_allowed()) {
          add(i, event_type::kDep);
        }
        utl::erase_duplicates(sig);
        if (sig.empty()) {
          continue;
        }
        auto& c = utl::get_or_create(route.stops_, i, [&]() {
          auto x = stop_state{};
          x.location_ = utl::get_or_create(location_idx, base, [&]() {
            st.locations_.emplace_back().l_ = base;
            signatures.emplace_back();
            return static_cast<std::uint32_t>(st.locations_.size() - 1U);
          });
          st.locations_[x.location_].stops_.push_back({r, i});
          x.member_.resize(transports.size(), kNone);
          return x;
        });
        c.member_[route_transport_idx_t{to_idx(t - transports.from_)}] =
            get_signature(c.location_, sig);
      }
    }

    // members: one per signature (transports without sides: the empty one)
    for (auto& [_, c] : route.stops_) {
      auto by_signature = hash_map<std::uint32_t, std::uint32_t>{};
      for (auto& m : c.member_) {
        auto const s = m == kNone ? get_signature(c.location_, {}) : m;
        m = utl::get_or_create(by_signature, s, [&]() {
          c.members_.emplace_back().signature_ = s;
          return static_cast<std::uint32_t>(c.members_.size() - 1U);
        });
      }
    }
  }
  return st;
}

// The transport pairs that change under another rule than they should.
void find_conflicts(timetable const& tt,
                    hash_map<transfer_pair, minutes_range> const& rule_times,
                    state const& st,
                    route_stop const x,
                    route_stop const y,
                    std::vector<std::array<std::uint32_t, 4>>& conflicts) {
  conflicts.clear();
  auto const& rules = tt.transfer_rules_.rules_;
  auto const& parents = tt.locations_.parents_;
  auto const& rx = st.routes_[x.route_];
  auto const& ry = st.routes_[y.route_];
  auto const& cx = rx.stops_.at(x.idx_);
  auto const& cy = ry.stops_.at(y.idx_);
  auto const& px = st.locations_[cx.location_];
  auto const& py = st.locations_[cy.location_];
  auto const x_transports = tt.route_transport_ranges_[x.route_];
  auto const y_transports = tt.route_transport_ranges_[y.route_];
  auto const lx = px.l_;
  auto const ly = py.l_;

  auto const walk =
      routing::shortest_transfer(tt, nullptr, kDefaultProfile, lx, ly);
  auto const fallback = walk.has_value() ? i32_minutes{*walk} : kNever;
  auto window = minutes_range{fallback, fallback};
  auto const extend = [&](location_idx_t const from, location_idx_t const to) {
    if (auto const r = utl::lookup(rule_times, transfer_pair{from, to})) {
      window.extend(r->lo_);
      window.extend(r->hi_);
    }
  };
  for (auto const from : {lx, parents[lx]}) {
    for (auto const to : {ly, parents[ly]}) {
      extend(from, to);
    }
  }
  if (auto const station = tt.locations_.get_root_idx(lx);
      station == tt.locations_.get_root_idx(ly) &&
      !tt.transfer_rules_.is_global_excluded(station)) {
    extend(location_idx_t::invalid(), location_idx_t::invalid());
  }
  auto const get_time = [&](transfer_rule_idx_t const r) {
    return r == transfer_rule_idx_t::invalid() ? fallback
                                               : get_minutes(rules[r]);
  };
  auto const traffic_days = [&](interval<transport_idx_t> const transports,
                                route_transport_idx_t const u) {
    return tt
        .bitfields_[tt.transport_traffic_days_[transports.from_ + to_idx(u)]];
  };
  auto via_cache =
      hash_map<pair<std::uint32_t, std::uint32_t>, transfer_rule_idx_t>{};
  auto expected_cache =
      hash_map<pair<std::uint32_t, std::uint32_t>, transfer_rule_idx_t>{};
  auto const& departures = cy.departures_;
  for (auto a = route_transport_idx_t{0U}; a != cx.member_.size(); ++a) {
    auto const sx = cx.members_[cx.member_[a]].signature_;
    auto const& a_days = traffic_days(x_transports, a);
    auto const arr =
        i32_minutes{tt.event_mam(x.route_, x_transports.from_ + to_idx(a),
                                 x.idx_, event_type::kArr)
                        .as_duration()};
    for (auto offset = std::chrono::floor<date::days>(arr + window.lo_ -
                                                      departures.back().first);
         offset <= std::chrono::floor<date::days>(arr + window.hi_ -
                                                  departures.front().first);
         ++offset) {
      auto const arrival = arr - offset;
      auto const n = offset.count();
      for (auto it = std::lower_bound(
               begin(departures), end(departures),
               std::pair{arrival + window.lo_, route_transport_idx_t{0U}});
           it != end(departures) && it->first < arrival + window.hi_; ++it) {
        auto const b = it->second;
        if (x.route_ == y.route_ && a == b) {
          continue;
        }
        auto const sy = cy.members_[cy.member_[b]].signature_;
        auto const expected =
            utl::get_or_create(expected_cache, pair{sx, sy}, [&]() {
              return get_change_rule(tt, px.signatures_[sx], py.signatures_[sy],
                                     lx, ly);
            });
        auto const parts = pair{rx.part_[to_idx(a)], ry.part_[to_idx(b)]};
        auto const via = utl::get_or_create(via_cache, parts, [&]() {
          return get_change_rule(tt, cx.sides_[parts.first],
                                 cy.sides_[parts.second], lx, ly);
        });
        if (expected == via) {
          continue;
        }
        auto const days = n >= 0 ? a_days << static_cast<std::size_t>(n)
                                 : a_days >> static_cast<std::size_t>(-n);
        auto const [lo, hi] = std::minmax({get_time(expected), get_time(via)});
        auto const gap = it->first - arrival;
        if (gap >= lo && gap < hi &&
            (days & traffic_days(y_transports, b)).any()) {
          conflicts.push_back(
              {to_idx(a), to_idx(b), to_idx(expected), to_idx(via)});
        }
      }
    }
  }
}

// A route part at a stop is at the location (the stop itself without sides,
// else a virtual location) for the sides of its class: route parts with the
// same group at the stop location share it unless one forbids a side another
// one needs. Sides of global rules (the same at every station) are only
// shared with route parts that have them, too. Transports that must not
// inherit a side another one of their route needs at some stop are in
// different route parts. Facts (needed and forbidden sides) are added from
// conflicts until there are none.
void resolve(timetable const& tt, state& st) {
  auto const& rules = tt.transfer_rules_.rules_;
  auto rule_times = hash_map<transfer_pair, minutes_range>{};
  for (auto const& rule : rules) {
    auto const m = get_minutes(rule);
    rule_times
        .emplace(transfer_pair{rule.from_stop_, rule.to_stop_},
                 minutes_range{m, m})
        .first->second.extend(m);
  }
  auto const get_stop = [&](route_stop const s) -> stop_state& {
    return st.routes_[s.route_].stops_.at(s.idx_);
  };
  auto const is_global = [&](transfer_rule_side_idx const s) {
    return rules[s.rule()].is_global();
  };
  auto const is_required = [&](member const& m,
                               transfer_rule_side_idx const s) {
    return !rules[s.rule()].is_qualified(s.ev_type()) ||
           std::ranges::binary_search(m.required_, s);
  };

  // Needed in any case: sides whose other side is unqualified.
  auto groups = hash_map<sides_t, std::uint32_t>{};
  for (auto& route : st.routes_) {
    for (auto& [_, c] : route.stops_) {
      for (auto& m : c.members_) {
        std::ranges::copy_if(
            st.locations_[c.location_].signatures_[m.signature_],
            std::back_inserter(m.required_),
            [&](transfer_rule_side_idx const s) {
              return !rules[s.rule()].is_qualified(flip(s.ev_type()));
            });
        m.group_ = utl::get_or_create(groups, m.required_, [&]() {
          return static_cast<std::uint32_t>(groups.size());
        });
      }
    }
  }

  // Route stop pairs a rule connects (a global rule: at the same station):
  // those of the signatures sorted, those of inherited sides appended.
  auto side_stops = hash_map<pair<transfer_rule_side_idx, location_idx_t>,
                             std::vector<route_stop>>{};
  auto connected = std::vector<pair<route_stop, route_stop>>{};
  auto checked = std::vector<std::uint64_t>{};  // the stamp when checked
  auto n_sorted = kNone;
  auto appended = hash_set<pair<route_stop, route_stop>>{};
  auto registered = hash_set<pair<route_stop, transfer_rule_side_idx>>{};
  auto const connect = [&](route_stop const rs,
                           transfer_rule_side_idx const side) {
    auto const seq = tt.route_location_seq_[rs.route_];
    auto const stp = stop{seq[rs.idx_]};
    if (!(side.ev_type() == event_type::kArr
              ? rs.idx_ != 0U && stp.out_allowed()
              : rs.idx_ + 1U != seq.size() && stp.in_allowed()) ||
        !registered.emplace(rs, side).second) {
      return;
    }
    auto const at = is_global(side)
                        ? tt.locations_.get_root_idx(stp.location_idx())
                        : location_idx_t::invalid();
    side_stops[pair{side, at}].push_back(rs);
    auto const partners = side_stops.find(
        pair{transfer_rule_side_idx{side.rule(), flip(side.ev_type())}, at});
    if (partners != end(side_stops)) {
      for (auto const& o : partners->second) {
        auto const p =
            side.ev_type() == event_type::kArr ? pair{rs, o} : pair{o, rs};
        if (n_sorted == kNone ||
            (!std::binary_search(begin(connected), begin(connected) + n_sorted,
                                 p) &&
             appended.emplace(p).second)) {
          connected.push_back(p);
          checked.push_back(0U);
        }
      }
    }
  };
  for (auto r = route_idx_t{0U}; r != st.routes_.size(); ++r) {
    for (auto& [k, c] : st.routes_[r].stops_) {
      for (auto const& m : c.members_) {
        for (auto const s :
             st.locations_[c.location_].signatures_[m.signature_]) {
          connect({r, k}, s);
        }
      }
      if (k + 1U != tt.route_location_seq_[r].size()) {
        auto const transports = tt.route_transport_ranges_[r];
        c.departures_ = utl::to_vec(
            interval{route_transport_idx_t{0U},
                     route_transport_idx_t{c.member_.size()}},
            [&](route_transport_idx_t const u) {
              return std::pair{
                  i32_minutes{tt.event_mam(r, transports.from_ + to_idx(u), k,
                                           event_type::kDep)
                                  .as_duration()},
                  u};
            });
        utl::sort(c.departures_);
      }
    }
  }
  utl::erase_duplicates(connected);
  n_sorted = static_cast<std::uint32_t>(connected.size());
  checked.resize(connected.size());

  // Without sharing, a route part at a stop is at the location for the sides
  // its transports need.
  auto stamp = std::uint64_t{1U};
  auto part_members = std::vector<std::pair<std::uint32_t, std::uint32_t>>{};
  auto const refresh = [&](route_state const& route, stop_state& c) {
    part_members.clear();
    for (auto a = 0U; a != static_cast<std::uint32_t>(route.part_.size());
         ++a) {
      part_members.emplace_back(route.part_[a],
                                c.member_[route_transport_idx_t{a}]);
    }
    utl::erase_duplicates(part_members);
    c.required_.assign(route.n_parts_, {});
    c.forbidden_.assign(route.n_parts_, {});
    c.unshared_.resize(route.n_parts_);
    for (auto const& [part, m] : part_members) {
      unite(c.required_[part], c.members_[m].required_);
      unite(c.forbidden_[part], c.members_[m].forbidden_);
    }
    if (c.sides_ != c.required_) {
      c.sides_ = c.required_;
      c.changed_ = ++stamp;
    }
  };

  // Route parts: colors of the route's transport types (the same members at
  // each stop) such that types with a fact against each other at a stop (or
  // different groups) differ. DSatur: the type with the most different colors
  // among those it is apart from is colored next.
  auto stops = std::vector<stop_state*>{};
  auto types = hash_map<std::vector<std::uint32_t>, std::uint32_t>{};
  auto type_of = std::vector<std::uint32_t>{};
  auto apart = std::vector<std::vector<std::uint32_t>>{};
  auto saturation = std::vector<std::vector<std::uint32_t>>{};
  auto colors = std::vector<std::uint32_t>{};
  auto const recolor = [&](route_state& route) {
    stops.clear();
    for (auto& [_, c] : route.stops_) {
      stops.push_back(&c);
    }
    types.clear();
    type_of.clear();
    auto key = std::vector<std::uint32_t>{};
    for (auto a = route_transport_idx_t{0U};
         a != static_cast<std::uint32_t>(route.part_.size()); ++a) {
      key.clear();
      for (auto const c : stops) {
        key.push_back(c->member_[a]);
      }
      type_of.push_back(utl::get_or_create(types, key, [&]() {
        return static_cast<std::uint32_t>(types.size());
      }));
    }
    auto const n = types.size();
    auto members = std::vector<std::vector<std::uint32_t>>(n);
    for (auto const& [k, t] : types) {
      members[t] = k;
    }
    apart.assign(n, {});
    for (auto t = 0U; t != n; ++t) {
      for (auto o = t + 1U; o != n; ++o) {
        for (auto j = 0U; j != stops.size(); ++j) {
          auto const& m = stops[j]->members_[members[t][j]];
          auto const& x = stops[j]->members_[members[o][j]];
          if (m.group_ != x.group_ || intersects(m.forbidden_, x.required_) ||
              intersects(m.required_, x.forbidden_)) {
            apart[t].push_back(o);
            apart[o].push_back(t);
            break;
          }
        }
      }
    }
    colors.assign(n, kNone);
    saturation.assign(n, {});
    for (auto step = 0U; step != n; ++step) {
      auto best = kNone;
      for (auto t = 0U; t != n; ++t) {
        if (colors[t] == kNone &&
            (best == kNone ||
             std::pair{saturation[t].size(), apart[t].size()} >
                 std::pair{saturation[best].size(), apart[best].size()})) {
          best = t;
        }
      }
      auto color = 0U;
      while (utl::find(saturation[best], color) != end(saturation[best])) {
        ++color;
      }
      colors[best] = color;
      for (auto const o : apart[best]) {
        if (utl::find(saturation[o], color) == end(saturation[o])) {
          saturation[o].push_back(color);
        }
      }
    }
    auto is_changed = false;
    route.n_parts_ = 0U;
    for (auto a = 0U; a != route.part_.size(); ++a) {
      is_changed |= route.part_[a] != colors[type_of[a]];
      route.part_[a] = colors[type_of[a]];
      route.n_parts_ = std::max(route.n_parts_, route.part_[a] + 1U);
    }
    for (auto const c : stops) {
      c->changed_ = is_changed ? ++stamp : c->changed_;
      refresh(route, *c);
    }
  };

  // Facts take effect: the route stops they touched get the sides their parts
  // need, a route is recolored if one of its parts conflicts now.
  auto touched = std::vector<route_stop>{};
  auto const apply = [&]() {
    utl::erase_duplicates(touched);
    for (auto const rs : touched) {
      auto& route = st.routes_[rs.route_];
      auto& c = get_stop(rs);
      refresh(route, c);
      for (auto part = 0U; part != route.n_parts_; ++part) {
        if (intersects(c.required_[part], c.forbidden_[part])) {
          recolor(route);
          break;
        }
      }
    }
    touched.clear();
  };

  // A fact for a transport: transports with the same facts share a member.
  auto const add_fact = [&](route_stop const rs, route_transport_idx_t const t,
                            transfer_rule_side_idx const side,
                            bool const is_required_fact) {
    auto& c = get_stop(rs);
    auto m = c.members_[c.member_[t]];
    if (!(is_required_fact ? m.required_ : m.forbidden_).empty() &&
        std::ranges::binary_search(
            is_required_fact ? m.required_ : m.forbidden_, side)) {
      return false;
    }
    unite(is_required_fact ? m.required_ : m.forbidden_, {side});
    if (is_required_fact) {
      std::erase(m.forbidden_, side);
    }
    auto const it = utl::find(c.members_, m);
    c.member_[t] = static_cast<std::uint32_t>(
        it == end(c.members_)
            ? (c.members_.push_back(std::move(m)), c.members_.size() - 1U)
            : static_cast<std::size_t>(std::distance(begin(c.members_), it)));
    touched.push_back(rs);
    return true;
  };

  // A side of a global rule is needed by all transports at the stop location
  // with it (it applies to them anyway): the same sides, fewer locations.
  auto const add_required = [&](route_stop const rs,
                                route_transport_idx_t const t,
                                transfer_rule_side_idx const side) {
    auto is_added = add_fact(rs, t, side, true);
    if (is_global(side)) {
      auto const& p = st.locations_[get_stop(rs).location_];
      for (auto const& o : p.stops_) {
        auto const& oc = get_stop(o);
        for (auto u = route_transport_idx_t{0U}; u != oc.member_.size(); ++u) {
          if (std::ranges::binary_search(
                  p.signatures_[oc.members_[oc.member_[u]].signature_], side)) {
            is_added |= add_fact(o, u, side, true);
          }
        }
      }
    }
    return is_added;
  };

  // Sharing: a route part at a stop joins the class at the stop location of
  // its group that has none of the sides it must not have, forbids none of its
  // own and grows the least (not by global sides). Route parts that need the
  // same sides at the stop share their class (the first one stands for them).
  struct location_class {
    std::uint32_t group_;
    sides_t sides_, forbidden_;
    std::vector<pair<route_stop, std::uint32_t>> members_;
  };
  auto location_classes =
      std::vector<std::vector<location_class>>(st.locations_.size());
  auto grow = sides_t{};
  auto const first = [](stop_state const& c, std::uint32_t const part) {
    return static_cast<std::uint32_t>(std::distance(
        begin(c.required_), utl::find(c.required_, c.required_[part])));
  };
  auto const get_forbidden = [](stop_state const& c, std::uint32_t const part) {
    auto forbidden = c.unshared_[part];
    for (auto p = 0U; p != c.required_.size(); ++p) {
      if (c.required_[p] == c.required_[part]) {
        unite(forbidden, c.forbidden_[p]);
      }
    }
    return forbidden;
  };
  auto const update = [&](location_class const& k) {
    for (auto const& [m_rs, m_part] : k.members_) {
      auto& m = get_stop(m_rs);
      for (auto p = 0U; p != m.required_.size(); ++p) {
        if (m.required_[p] == m.required_[m_part] && m.sides_[p] != k.sides_) {
          m.sides_[p] = k.sides_;
          m.changed_ = ++stamp;
          for (auto const s : k.sides_) {
            if (!std::ranges::binary_search(m.required_[p], s)) {
              connect(m_rs, s);
            }
          }
        }
      }
    }
  };
  auto const place = [&](route_stop const rs, std::uint32_t const part) {
    auto const& route = st.routes_[rs.route_];
    auto& c = get_stop(rs);
    auto const& required = c.required_[part];
    auto const forbidden = get_forbidden(c, part);
    auto const group =
        c.members_[c.member_[route_transport_idx_t{std::distance(
                       begin(route.part_), utl::find(route.part_, part))}]]
            .group_;
    auto& ks = location_classes[c.location_];
    for (auto& k : ks) {
      if (std::erase(k.members_, pair{rs, part}) != 0U) {
        k.sides_.clear();
        k.forbidden_.clear();
        for (auto const& [m_rs, m_part] : k.members_) {
          auto const& m = get_stop(m_rs);
          unite(k.sides_, m.required_[m_part]);
          unite(k.forbidden_, get_forbidden(m, m_part));
        }
        update(k);
      }
    }
    auto best = end(ks);
    auto best_cost = std::numeric_limits<std::size_t>::max();
    for (auto it = begin(ks); it != end(ks); ++it) {
      grow.clear();
      std::ranges::set_difference(required, it->sides_,
                                  std::back_inserter(grow));
      auto const cost =
          it->sides_.size() + grow.size() * (it->members_.size() + 1U);
      if (it->group_ == group && !intersects(it->sides_, forbidden) &&
          !intersects(grow, it->forbidden_) && utl::none_of(grow, is_global) &&
          cost < best_cost) {
        best = it;
        best_cost = cost;
      }
    }
    if (best == end(ks)) {
      best = ks.insert(end(ks), {group, required, {}, {}});
    }
    unite(best->sides_, required);
    unite(best->forbidden_, forbidden);
    best->members_.push_back({rs, part});
    update(*best);
  };

  // Whether the transport fits into another part of its route if it must not
  // have the side at the stop.
  auto const can_move = [&](route_stop const rs, route_transport_idx_t const t,
                            transfer_rule_side_idx const side) {
    auto const& route = st.routes_[rs.route_];
    for (auto part = 0U; part != route.n_parts_; ++part) {
      auto const o = utl::find(route.part_, part);
      if (part == route.part_[to_idx(t)] || o == end(route.part_)) {
        continue;
      }
      auto const u =
          route_transport_idx_t{std::distance(begin(route.part_), o)};
      if (utl::all_of(route.stops_, [&](auto const& entry) {
            auto const& [i, c] = entry;
            auto const& m = c.members_[c.member_[t]];
            return m.group_ == c.members_[c.member_[u]].group_ &&
                   !intersects(m.forbidden_, c.required_[part]) &&
                   !intersects(m.required_, c.forbidden_[part]) &&
                   (i != rs.idx_ ||
                    !std::ranges::binary_search(c.required_[part], side));
          })) {
        return true;
      }
    }
    return false;
  };

  auto const has = [&](route_stop const rs, route_transport_idx_t const t,
                       transfer_rule_side_idx const s) {
    auto const& c = get_stop(rs);
    return is_required(c.members_[c.member_[t]], s);
  };

  // Sharing: the route part of the transport must not inherit the side at the
  // stop: it moves to another class.
  auto const unshare = [&](route_stop const rs, route_transport_idx_t const t,
                           transfer_rule_side_idx const s) {
    auto& c = get_stop(rs);
    auto const part = first(c, st.routes_[rs.route_].part_[to_idx(t)]);
    if (!std::ranges::binary_search(c.sides_[part], s) ||
        std::ranges::binary_search(c.required_[part], s) ||
        std::ranges::binary_search(c.unshared_[part], s)) {
      return false;
    }
    unite(c.unshared_[part], {s});
    place(rs, part);
    return true;
  };

  // Resolve conflicts until there are none: first without sharing, then with
  // sharing. A pair whose route stops kept their sides since it was checked
  // has no conflict. A missing expected rule: both transports need its sides.
  // Else, the inherited side that applies is forbidden: for the transport
  // (without sharing) or its route part at the stop (sharing). Sides inherited
  // at both ends wait until no other fact is left: then they are forbidden
  // where the transport fits into another part of its route (the departure if
  // both or none do).
  auto conflicts = std::vector<std::array<std::uint32_t, 4>>{};
  auto n_pass = 0U;
  auto n_stuck = std::size_t{0U};
  for (auto& route : st.routes_) {
    recolor(route);
  }
  for (auto const is_sharing : {false, true}) {
    if (is_sharing) {
      auto parts = std::vector<
          std::tuple<std::uint32_t, std::size_t, route_stop, std::uint32_t>>{};
      for (auto r = route_idx_t{0U}; r != st.routes_.size(); ++r) {
        for (auto const& [k, c] : st.routes_[r].stops_) {
          for (auto part = 0U; part != st.routes_[r].n_parts_; ++part) {
            if (!c.required_[part].empty() && first(c, part) == part) {
              parts.emplace_back(c.location_, kNone - c.required_[part].size(),
                                 route_stop{r, k}, part);
            }
          }
        }
      }
      utl::sort(parts);
      for (auto const& [loc, n_required, rs, part] : parts) {
        place(rs, part);
      }
    }
    for (auto is_changed = true, allow_both = false; is_changed; ++n_pass) {
      auto const is_allow_pass = allow_both;
      auto n_facts = std::size_t{0U};
      auto n_pending = std::size_t{0U};
      n_stuck = 0U;
      for (auto i = std::size_t{0U}; i != connected.size(); ++i) {
        auto const [x, y] = connected[i];
        if (get_stop(x).changed_ <= checked[i] &&
            get_stop(y).changed_ <= checked[i]) {
          continue;
        }
        for (auto is_again = true; is_again;) {
          auto const n_before = n_facts;
          checked[i] = stamp;
          find_conflicts(tt, rule_times, st, x, y, conflicts);
          for (auto const& [a_idx, b_idx, expected_idx, via_idx] : conflicts) {
            auto const a = route_transport_idx_t{a_idx};
            auto const b = route_transport_idx_t{b_idx};
            auto const expected = transfer_rule_idx_t{expected_idx};
            auto const via = transfer_rule_idx_t{via_idx};
            auto const from = transfer_rule_side_idx{via, event_type::kArr};
            auto const to = transfer_rule_side_idx{via, event_type::kDep};
            auto const is_missing = [&](route_stop const rs,
                                        route_transport_idx_t const t,
                                        event_type const ev) {
              return expected != transfer_rule_idx_t::invalid() &&
                     !has(rs, t, {expected, ev});
            };
            auto const is_from = is_missing(x, a, event_type::kArr);
            auto const is_to = is_missing(y, b, event_type::kDep);
            auto const n = n_facts;
            if (is_sharing) {
              n_facts += via != transfer_rule_idx_t::invalid() &&
                                 (unshare(x, a, from) || unshare(y, b, to))
                             ? 1U
                             : 0U;
            } else if (is_from || is_to) {
              n_facts +=
                  is_from && add_required(x, a, {expected, event_type::kArr})
                      ? 1U
                      : 0U;
              n_facts +=
                  is_to && add_required(y, b, {expected, event_type::kDep})
                      ? 1U
                      : 0U;
            } else if (via != transfer_rule_idx_t::invalid()) {
              auto is_dep = !has(y, b, to);
              if (is_dep && !has(x, a, from)) {
                if (!is_allow_pass) {
                  checked[i] = 0U;
                  ++n_pending;
                  continue;
                }
                is_dep = can_move(y, b, to) || !can_move(x, a, from);
              }
              n_facts += add_fact(is_dep ? y : x, is_dep ? b : a,
                                  is_dep ? to : from, false)
                             ? 1U
                             : 0U;
            }
            n_stuck += n == n_facts ? 1U : 0U;
          }
          apply();
          is_again = n_facts != n_before;
        }
      }
      allow_both = !is_allow_pass && n_facts == 0U && n_pending != 0U;
      is_changed = n_facts != 0U || allow_both || is_allow_pass;
    }
  }
  if (n_stuck != 0U) {
    log(log_lvl::error, "loader.virtual_locations", "{} unresolved conflicts",
        n_stuck);
  }
  log(log_lvl::info, "loader.virtual_locations",
      "{} route stop pairs, {} passes", connected.size(), n_pass);
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
  if (tt.initial_day_offset_.size() == order.size()) {
    permute(tt.initial_day_offset_);
  }
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

bool write(timetable& tt, state const& st, shapes_storage* shapes) {
  auto& tr = tt.transfer_rules_;
  auto virts = hash_map<pair<location_idx_t, sides_t>, location_idx_t>{};
  auto origin = vector_map<route_idx_t, route_idx_t>{};
  auto route_transports =
      vector_map<route_idx_t, std::vector<transport_idx_t>>{};
  auto const n_routes = tt.n_routes();
  for (auto r = route_idx_t{0U}; r != n_routes; ++r) {
    auto const transports = tt.route_transport_ranges_[r];
    origin.push_back(r);
    route_transports.emplace_back(begin(transports), end(transports));
  }

  auto seqs = std::vector<stop_seq_t>{};
  for (auto r = route_idx_t{0U}; r != n_routes; ++r) {
    auto const& route = st.routes_[r];
    if (route.stops_.empty()) {
      continue;
    }
    auto const transports = tt.route_transport_ranges_[r];
    seqs.assign(transports.size(), stop_seq_t{tt.route_location_seq_[r].begin(),
                                              tt.route_location_seq_[r].end()});
    for (auto const& [k, c] : route.stops_) {
      auto const base = st.locations_[c.location_].l_;
      for (auto u = 0U; u != seqs.size(); ++u) {
        auto const& sides = c.sides_[route.part_[u]];
        seqs[u][k] =
            stop{seqs[u][k]}
                .with_location(sides.empty()
                                   ? base
                                   : get_or_create_virt(tt, virts, base, sides))
                .value();
      }
    }

    auto parts =
        std::vector<std::pair<stop_seq_t, std::vector<transport_idx_t>>>{};
    for (auto u = 0U; u != seqs.size(); ++u) {
      auto it = utl::find_if(
          parts, [&](auto const& part) { return part.first == seqs[u]; });
      if (it == end(parts)) {
        it = parts.insert(end(parts), {seqs[u], {}});
      }
      it->second.push_back(transports.from_ + u);
    }
    route_transports[r] = std::move(parts.front().second);
    std::ranges::copy(parts.front().first, tt.route_location_seq_[r].begin());
    for (auto& [part_seq, part_transports] : std::span{parts}.subspan(1U)) {
      origin.push_back(r);
      route_transports.push_back(std::move(part_transports));
      tt.route_location_seq_.emplace_back(part_seq);
    }
  }
  if (virts.empty()) {
    return false;
  }
  split_routes(tt, shapes, origin, route_transports);

  utl::erase_duplicates(tr.rule_virts_);
  utl::erase_duplicates(tr.virt_rules_);

  log(log_lvl::info, "loader.virtual_locations",
      "{} virtual locations, {} routes -> {}", virts.size(), n_routes,
      tt.n_routes());
  return true;
}

}  // namespace

bool add_virtual_locations(timetable& tt, shapes_storage* shapes) {
  if (tt.transfer_rules_.rules_.empty()) {
    return false;
  }
  auto const timer = scoped_timer{"loader.virtual_locations"};
  auto st = get_state(tt);
  resolve(tt, st);
  return write(tt, st, shapes);
}

}  // namespace nigiri::loader
