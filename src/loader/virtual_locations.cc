#include "nigiri/loader/virtual_locations.h"

#include <chrono>
#include <algorithm>
#include <bit>
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

using group_idx_t = cista::strong<std::uint32_t, struct group_>;

constexpr auto const kNever = i32_minutes{routing::kMaxTravelTime};

using stop_signature_t = std::vector<transfer_rule_side_idx>;
using stop_seq_t = std::vector<stop::value_type>;

struct route_stop {
  CISTA_COMPARABLE()
  route_idx_t route_;
  stop_idx_t idx_;
};

using unit_idx_t = cista::strong<std::uint32_t, struct unit_>;
using member_idx_t = cista::strong<std::uint32_t, struct member_>;
using part_idx_t = cista::strong<std::uint32_t, struct part_>;

struct change {
  route_transport_idx_t from_;
  route_transport_idx_t to_;
  transfer_rule_idx_t expected_;
  transfer_rule_idx_t via_;
};

// The transports of a route stop with the same signature and facts: the sides
// they need and the sides they must not inherit.
struct member {
  unit_idx_t unit_;
  group_idx_t group_{0U};
  stop_signature_t required_;
  stop_signature_t forbidden_;
};

// A route stop: per transport its member, per route part the facts of its
// transports and the sides of the location it is at (the stop itself without
// sides, else a virtual location).
struct stop_classes {
  vector_map<route_transport_idx_t, member_idx_t> member_;
  vector_map<member_idx_t, member> members_;
  vector_map<part_idx_t, stop_signature_t> required_;
  vector_map<part_idx_t, stop_signature_t> forbidden_;
  vector_map<part_idx_t, stop_signature_t> sides_;
  std::vector<std::pair<i32_minutes, route_transport_idx_t>> departures_;
  std::uint64_t changed_{0U};
  std::uint32_t location_{0U};
};

// The transports of a route part share a location at each stop. A route with
// more than one part is split.
struct route_classes {
  hash_map<stop_idx_t, stop_classes> stops_;
  vector_map<route_transport_idx_t, part_idx_t> part_;
  std::uint32_t n_parts_{1U};
};

// The distinct signatures at a stop location and their transports.
struct location_classes {
  location_idx_t location_;
  vector_map<unit_idx_t, stop_signature_t> signature_;
  vector_map<unit_idx_t, std::vector<pair<route_stop, route_transport_idx_t>>>
      transports_;
};

struct classes_t {
  vector_map<route_idx_t, route_classes> routes_;
  std::vector<location_classes> locations_;
};

struct minutes_range {
  void extend(i32_minutes const m) {
    lo_ = std::min(lo_, m);
    hi_ = std::max(hi_, m);
  }

  i32_minutes lo_;
  i32_minutes hi_;
};

i32_minutes get_minutes(transfer_rule const& r) {
  return r.duration_ == footpath::kMaxDuration ? kNever
                                               : i32_minutes{r.duration_};
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

classes_t get_signatures(timetable const& tt) {
  auto const& tr = tt.transfer_rules_;
  auto const& parents = tt.locations_.parents_;
  auto const global_rules = get_global_rules(tt);

  // get_stop_signature, with one pass over the sides of a section's trips
  using sides_t = hash_map<pair<location_idx_t, event_type>, stop_signature_t>;
  auto by_section = hash_map<merged_trips_idx_t, sides_t>{};
  auto const get_sides = [&](merged_trips_idx_t const m) -> sides_t const& {
    return utl::get_or_create(by_section, m, [&]() {
      auto sides = sides_t{};
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

  auto classes = classes_t{};
  classes.routes_.resize(tt.n_routes());
  auto location_idx = hash_map<location_idx_t, std::uint32_t>{};
  auto units = std::vector<hash_map<stop_signature_t, unit_idx_t>>{};
  auto const get_unit = [&](std::uint32_t const l,
                            stop_signature_t const& sig) {
    return utl::get_or_create(units[l], sig, [&]() {
      auto& p = classes.locations_[l];
      p.signature_.push_back(sig);
      return unit_idx_t{p.signature_.size() - 1U};
    });
  };
  auto route_members = hash_map<pair<stop_idx_t, unit_idx_t>, member_idx_t>{};
  auto const get_member = [&](stop_idx_t const i, stop_classes& c,
                              stop_signature_t const& sig) {
    auto const u = get_unit(c.location_, sig);
    return utl::get_or_create(route_members, pair{i, u}, [&]() {
      c.members_.emplace_back().unit_ = u;
      return member_idx_t{c.members_.size() - 1U};
    });
  };

  auto sig = stop_signature_t{};
  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    auto const transports = tt.route_transport_ranges_[r];
    auto const seq = tt.route_location_seq_[r];
    auto& route = classes.routes_[r].stops_;
    classes.routes_[r].part_.resize(transports.size(), part_idx_t{0U});
    route_members.clear();
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
        auto& c = utl::get_or_create(route, i, [&]() {
          auto x = stop_classes{};
          x.location_ = utl::get_or_create(location_idx, base, [&]() {
            classes.locations_.emplace_back().location_ = base;
            units.emplace_back();
            return static_cast<std::uint32_t>(classes.locations_.size() - 1U);
          });
          x.member_.resize(transports.size(), member_idx_t::invalid());
          return x;
        });
        c.member_[route_transport_idx_t{to_idx(t - transports.from_)}] =
            get_member(i, c, sig);
      }
    }

    // transports without sides at a route stop with sides: the empty unit
    for (auto& [i, c] : route) {
      for (auto& m : c.member_) {
        if (m == member_idx_t::invalid()) {
          m = get_member(i, c, {});
        }
      }
      auto& p = classes.locations_[c.location_];
      p.transports_.resize(p.signature_.size());
      for (auto a = route_transport_idx_t{0U}; a != c.member_.size(); ++a) {
        p.transports_[c.members_[c.member_[a]].unit_].push_back({{r, i}, a});
      }
    }
  }
  return classes;
}

// The transport pairs that change under another rule than they should.
void find_conflicts(timetable const& tt,
                    hash_map<transfer_pair, minutes_range> const& rule_times,
                    classes_t const& classes,
                    route_stop const x,
                    route_stop const y,
                    std::vector<change>& conflicts) {
  conflicts.clear();
  auto const& rules = tt.transfer_rules_.rules_;
  auto const& parents = tt.locations_.parents_;
  auto const& rx = classes.routes_[x.route_];
  auto const& ry = classes.routes_[y.route_];
  auto const& cx = rx.stops_.at(x.idx_);
  auto const& cy = ry.stops_.at(y.idx_);
  auto const& px = classes.locations_[cx.location_];
  auto const& py = classes.locations_[cy.location_];
  auto const x_transports = tt.route_transport_ranges_[x.route_];
  auto const y_transports = tt.route_transport_ranges_[y.route_];
  auto const lx = px.location_;
  auto const ly = py.location_;

  auto const walk =
      routing::shortest_transfer(tt, nullptr, kDefaultProfile, lx, ly);
  auto const fallback = walk.has_value() ? i32_minutes{*walk} : kNever;
  auto window = minutes_range{fallback, fallback};
  auto const extend = [&](location_idx_t const from, location_idx_t const to) {
    if (auto const r = utl::lookup(rule_times, transfer_pair{from, to});
        r.has_value()) {
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
  auto const get_traffic_days = [&](interval<transport_idx_t> const transports,
                                    route_transport_idx_t const u) {
    return tt
        .bitfields_[tt.transport_traffic_days_[transports.from_ + to_idx(u)]];
  };
  auto via_cache =
      hash_map<pair<part_idx_t, part_idx_t>, transfer_rule_idx_t>{};
  auto expected_cache =
      hash_map<pair<unit_idx_t, unit_idx_t>, transfer_rule_idx_t>{};
  auto const& departures = cy.departures_;

  for (auto a = route_transport_idx_t{0U}; a != cx.member_.size(); ++a) {
    auto const ux = cx.members_[cx.member_[a]].unit_;
    auto const& a_days = get_traffic_days(x_transports, a);
    auto const arr =
        i32_minutes{tt.event_mam(x.route_, x_transports.from_ + to_idx(a),
                                 x.idx_, event_type::kArr)
                        .as_duration()};
    for (auto start_day_offset = std::chrono::floor<date::days>(
             arr + window.lo_ - departures.back().first);
         start_day_offset <= std::chrono::floor<date::days>(
                                 arr + window.hi_ - departures.front().first);
         ++start_day_offset) {
      auto const arrival = arr - start_day_offset;
      auto const n = start_day_offset.count();
      for (auto it = std::lower_bound(
               begin(departures), end(departures),
               std::pair{arrival + window.lo_, route_transport_idx_t{0U}});
           it != end(departures) && it->first < arrival + window.hi_; ++it) {
        auto const b = it->second;
        if (x.route_ == y.route_ && a == b) {
          continue;
        }
        auto const uy = cy.members_[cy.member_[b]].unit_;
        auto const expected =
            utl::get_or_create(expected_cache, pair{ux, uy}, [&]() {
              return get_change_rule(tt, px.signature_[ux], py.signature_[uy],
                                     lx, ly);
            });
        auto const parts = pair{rx.part_[a], ry.part_[b]};
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
            (days & get_traffic_days(y_transports, b)).any()) {
          conflicts.push_back({a, b, expected, via});
        }
      }
    }
  }
}

bool intersects(stop_signature_t const& a, stop_signature_t const& b) {
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

bool insert(stop_signature_t& sides, transfer_rule_side_idx const side) {
  auto const it = std::ranges::lower_bound(sides, side);
  if (it != end(sides) && *it == side) {
    return false;
  }
  sides.insert(it, side);
  return true;
}

void erase(stop_signature_t& sides, transfer_rule_side_idx const side) {
  if (auto const it = std::ranges::lower_bound(sides, side);
      it != end(sides) && *it == side) {
    sides.erase(it);
  }
}

// Transports with different groups or a fact against each other at a stop
// can't share it.
bool is_apart(member const& a, member const& b) {
  return a.group_ != b.group_ || intersects(a.forbidden_, b.required_) ||
         intersects(b.forbidden_, a.required_);
}

// Each route part at a stop gets the virtual location for the sides its
// transports need. Transports that must not inherit a side another one needs
// at some stop end up in different parts (a route split).
void resolve_conflicts(timetable const& tt, classes_t& classes) {
  auto const& rules = tt.transfer_rules_.rules_;
  auto rule_times = hash_map<transfer_pair, minutes_range>{};
  for (auto const& rule : rules) {
    auto const m = get_minutes(rule);
    rule_times
        .emplace(transfer_pair{rule.from_stop_, rule.to_stop_},
                 minutes_range{m, m})
        .first->second.extend(m);
  }
  auto const get_stop = [&](route_stop const s) -> stop_classes& {
    return classes.routes_[s.route_].stops_.at(s.idx_);
  };
  auto const is_required = [&](stop_signature_t const& required,
                               transfer_rule_side_idx const side) {
    return !rules[side.rule()].is_qualified(side.ev_type()) ||
           std::ranges::binary_search(required, side);
  };

  // Required from the start: sides whose other side is unqualified.
  auto groups = hash_map<stop_signature_t, group_idx_t>{};
  for (auto& route : classes.routes_) {
    for (auto& [_, c] : route.stops_) {
      auto const& p = classes.locations_[c.location_];
      for (auto& m : c.members_) {
        std::ranges::copy_if(
            p.signature_[m.unit_], std::back_inserter(m.required_),
            [&](transfer_rule_side_idx const side) {
              return !rules[side.rule()].is_qualified(flip(side.ev_type()));
            });
        m.group_ =
            m.required_.empty()
                ? group_idx_t{0U}
                : groups.emplace(m.required_, group_idx_t{groups.size() + 1U})
                      .first->second;
      }
    }
  }

  // The facts of each route part at a route stop.
  auto stamp = std::uint64_t{0U};
  auto part_members = std::vector<std::vector<member_idx_t>>{};
  auto const refresh = [&](route_idx_t const r, stop_idx_t const i) {
    auto& route = classes.routes_[r];
    auto& c = route.stops_.at(i);
    part_members.resize(route.n_parts_);
    for (auto& v : part_members) {
      v.clear();
    }
    for (auto a = route_transport_idx_t{0U}; a != route.part_.size(); ++a) {
      part_members[to_idx(route.part_[a])].push_back(c.member_[a]);
    }
    c.required_.resize(route.n_parts_);
    c.forbidden_.resize(route.n_parts_);
    for (auto part = part_idx_t{0U}; part != route.n_parts_; ++part) {
      auto& members = part_members[to_idx(part)];
      auto& required = c.required_[part];
      auto& forbidden = c.forbidden_[part];
      utl::erase_duplicates(members);
      required.clear();
      forbidden.clear();
      for (auto const m : members) {
        auto const& x = c.members_[m];
        required.insert(end(required), begin(x.required_), end(x.required_));
        forbidden.insert(end(forbidden), begin(x.forbidden_),
                         end(x.forbidden_));
      }
      utl::erase_duplicates(required);
      utl::erase_duplicates(forbidden);
    }
    c.sides_ = c.required_;
    c.changed_ = ++stamp;
  };

  // The route parts: colors of the route's transport types (the same members
  // at each stop) such that types apart at some stop differ.
  auto stops = std::vector<stop_idx_t>{};
  auto stop_ptrs = std::vector<stop_classes const*>{};
  auto types = hash_map<std::vector<member_idx_t>, std::uint32_t>{};
  auto type_of = std::vector<std::uint32_t>{};
  auto type_members = std::vector<std::vector<member_idx_t>>{};
  auto member_types = std::vector<std::vector<std::uint32_t>>{};
  auto used = std::vector<member_idx_t>{};
  auto apart = std::vector<std::vector<std::uint32_t>>{};
  auto apart_bits = std::vector<std::uint64_t>{};
  auto side_members =
      std::vector<std::tuple<transfer_rule_side_idx, bool, member_idx_t>>{};
  auto colors = std::vector<std::uint32_t>{};
  auto saturation = std::vector<std::vector<std::uint32_t>>{};
  auto is_colored = std::vector<bool>{};
  auto const recolor = [&](route_idx_t const r) {
    auto& route = classes.routes_[r];
    stops.clear();
    for (auto const& [i, _] : route.stops_) {
      stops.push_back(i);
    }
    utl::sort(stops);
    stop_ptrs.clear();
    for (auto const i : stops) {
      stop_ptrs.push_back(&route.stops_.at(i));
    }

    types.clear();
    type_members.clear();
    type_of.assign(route.part_.size(), 0U);
    auto key = std::vector<member_idx_t>{};
    for (auto a = route_transport_idx_t{0U}; a != route.part_.size(); ++a) {
      key.clear();
      for (auto const c : stop_ptrs) {
        key.push_back(c->member_[a]);
      }
      type_of[to_idx(a)] = utl::get_or_create(types, key, [&]() {
        type_members.push_back(key);
        return static_cast<std::uint32_t>(type_members.size() - 1U);
      });
    }

    // apart: a side one member forbids and another one requires, or groups
    auto const n = static_cast<std::uint32_t>(type_members.size());
    auto const words = (n + 63U) / 64U;
    apart_bits.assign(static_cast<std::size_t>(n) * words, 0U);
    for (auto j = 0U; j != stop_ptrs.size(); ++j) {
      auto const& c = *stop_ptrs[j];
      member_types.resize(c.members_.size());
      used.clear();
      for (auto t = 0U; t != n; ++t) {
        auto const m = type_members[t][j];
        if (member_types[to_idx(m)].empty()) {
          used.push_back(m);
        }
        member_types[to_idx(m)].push_back(t);
      }
      auto const set_apart = [&](member_idx_t const mx, member_idx_t const my) {
        for (auto const tx : member_types[to_idx(mx)]) {
          for (auto const ty : member_types[to_idx(my)]) {
            apart_bits[tx * words + ty / 64U] |= 1ULL << (ty % 64U);
            apart_bits[ty * words + tx / 64U] |= 1ULL << (tx % 64U);
          }
        }
      };
      side_members.clear();
      for (auto const m : used) {
        for (auto const side : c.members_[m].required_) {
          side_members.emplace_back(side, true, m);
        }
        for (auto const side : c.members_[m].forbidden_) {
          side_members.emplace_back(side, false, m);
        }
      }
      utl::sort(side_members);
      utl::equal_ranges_linear(
          side_members,
          [](auto const& a, auto const& b) {
            return std::get<0>(a) == std::get<0>(b);
          },
          [&](auto const from, auto const to) {
            auto const mid = std::find_if(
                from, to, [](auto const& e) { return std::get<1>(e); });
            for (auto f = from; f != mid; ++f) {
              for (auto q = mid; q != to; ++q) {
                set_apart(std::get<2>(*f), std::get<2>(*q));
              }
            }
          });
      for (auto x = 0U; x != used.size(); ++x) {
        for (auto y = 0U; y != x; ++y) {
          if (c.members_[used[x]].group_ != c.members_[used[y]].group_) {
            set_apart(used[x], used[y]);
          }
        }
      }
      for (auto const m : used) {
        member_types[to_idx(m)].clear();
      }
    }
    apart.resize(n);
    for (auto t = 0U; t != n; ++t) {
      apart[t].clear();
      for (auto w = 0U; w != words; ++w) {
        for (auto bits = apart_bits[t * words + w]; bits != 0U;
             bits &= bits - 1U) {
          apart[t].push_back(
              w * 64U + static_cast<std::uint32_t>(std::countr_zero(bits)));
        }
      }
    }

    // DSatur
    colors.assign(n, 0U);
    is_colored.assign(n, false);
    saturation.resize(n);
    for (auto t = 0U; t != n; ++t) {
      saturation[t].clear();
    }
    auto n_colors = 1U;
    for (auto step = 0U; step != n; ++step) {
      auto best = n;
      for (auto t = 0U; t != n; ++t) {
        if (!is_colored[t] &&
            (best == n || saturation[t].size() > saturation[best].size() ||
             (saturation[t].size() == saturation[best].size() &&
              apart[t].size() > apart[best].size()))) {
          best = t;
        }
      }
      auto color = 0U;
      while (utl::find(saturation[best], color) != end(saturation[best])) {
        ++color;
      }
      colors[best] = color;
      is_colored[best] = true;
      n_colors = std::max(n_colors, color + 1U);
      for (auto const o : apart[best]) {
        if (utl::find(saturation[o], color) == end(saturation[o])) {
          saturation[o].push_back(color);
        }
      }
    }

    auto is_changed = false;
    for (auto a = route_transport_idx_t{0U}; a != route.part_.size(); ++a) {
      auto const part = part_idx_t{colors[type_of[to_idx(a)]]};
      is_changed |= route.part_[a] != part;
      route.part_[a] = part;
    }
    route.n_parts_ = std::max(route.n_parts_, n_colors);
    for (auto const i : stops) {
      if (is_changed || route.stops_.at(i).required_.empty()) {
        refresh(r, i);
      }
    }
  };
  for (auto r = route_idx_t{0U}; r != classes.routes_.size(); ++r) {
    if (!classes.routes_[r].stops_.empty()) {
      recolor(r);
    }
  }

  // Whether a route part's transports at a stop conflict among each other.
  auto seen = std::vector<member_idx_t>{};
  auto const is_consistent = [&](route_stop const rs, part_idx_t const part) {
    auto const& c = get_stop(rs);
    auto const& parts = classes.routes_[rs.route_].part_;
    seen.clear();
    for (auto a = route_transport_idx_t{0U}; a != parts.size(); ++a) {
      if (parts[a] == part) {
        seen.push_back(c.member_[a]);
      }
    }
    utl::erase_duplicates(seen);
    for (auto i = 0U; i != seen.size(); ++i) {
      for (auto j = 0U; j != i; ++j) {
        if (is_apart(c.members_[seen[i]], c.members_[seen[j]])) {
          return false;
        }
      }
    }
    return true;
  };

  // Route stops a rule connects (a global rule: at the same station).
  auto const station = [&](route_stop const& s) {
    return tt.locations_.get_root_idx(
        stop{tt.route_location_seq_[s.route_][s.idx_]}.location_idx());
  };
  auto side_stops = hash_map<pair<transfer_rule_side_idx, location_idx_t>,
                             std::vector<route_stop>>{};
  auto stop_sides = stop_signature_t{};
  for (auto r = route_idx_t{0U}; r != classes.routes_.size(); ++r) {
    for (auto& [k, c] : classes.routes_[r].stops_) {
      auto const& p = classes.locations_[c.location_];
      stop_sides.clear();
      for (auto const& m : c.members_) {
        auto const& sig = p.signature_[m.unit_];
        stop_sides.insert(end(stop_sides), begin(sig), end(sig));
      }
      utl::erase_duplicates(stop_sides);
      for (auto const side : stop_sides) {
        side_stops[pair{side, rules[side.rule()].is_global()
                                  ? station({r, k})
                                  : location_idx_t::invalid()}]
            .push_back({r, k});
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
  auto connected = std::vector<pair<route_stop, route_stop>>{};
  for (auto const& [key, from] : side_stops) {
    auto const& [side, at] = key;
    auto const to = side_stops.find(
        pair{transfer_rule_side_idx{side.rule(), event_type::kDep}, at});
    if (side.ev_type() != event_type::kArr || to == end(side_stops)) {
      continue;
    }
    for (auto const& x : from) {
      for (auto const& y : to->second) {
        connected.emplace_back(x, y);
      }
    }
  }
  utl::erase_duplicates(connected);
  auto checked = std::vector<std::uint64_t>(connected.size(), 0U);
  auto const n_initial = static_cast<std::ptrdiff_t>(connected.size());
  auto later = hash_set<pair<route_stop, route_stop>>{};
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
    auto const at = rules[side.rule()].is_global() ? station(rs)
                                                   : location_idx_t::invalid();
    side_stops[pair{side, at}].push_back(rs);
    auto const partners = side_stops.find(
        pair{transfer_rule_side_idx{side.rule(), flip(side.ev_type())}, at});
    if (partners == end(side_stops)) {
      return;
    }
    for (auto const& o : partners->second) {
      auto const p =
          side.ev_type() == event_type::kArr ? pair{rs, o} : pair{o, rs};
      if (!std::binary_search(begin(connected), begin(connected) + n_initial,
                              p) &&
          later.emplace(p).second) {
        connected.push_back(p);
        checked.push_back(0U);
      }
    }
  };

  // A transport fact: transports with the same signature and facts share a
  // member.
  auto touched = std::vector<pair<route_stop, part_idx_t>>{};
  auto const add_fact = [&](route_stop const rs, route_transport_idx_t const t,
                            transfer_rule_side_idx const side,
                            bool const is_required_fact) {
    auto& c = get_stop(rs);
    auto m = c.members_[c.member_[t]];
    if (is_required_fact) {
      if (!insert(m.required_, side)) {
        return false;
      }
      erase(m.forbidden_, side);
    } else if (!insert(m.forbidden_, side)) {
      return false;
    }
    auto const it = utl::find_if(c.members_, [&](member const& o) {
      return o.unit_ == m.unit_ && o.required_ == m.required_ &&
             o.forbidden_ == m.forbidden_;
    });
    if (it == end(c.members_)) {
      c.members_.push_back(std::move(m));
      c.member_[t] = member_idx_t{c.members_.size() - 1U};
    } else {
      c.member_[t] = member_idx_t{
          static_cast<std::size_t>(std::distance(begin(c.members_), it))};
    }
    touched.emplace_back(rs, classes.routes_[rs.route_].part_[t]);
    return true;
  };

  // A required side of a global rule: for all transports at the stop location
  // whose signature has it (the rule applies to them anyway), so that route
  // parts at the location need the same sides and can share a location.
  auto const add_required = [&](route_stop const rs,
                                route_transport_idx_t const t,
                                transfer_rule_side_idx const side) {
    auto is_added = add_fact(rs, t, side, true);
    if (!rules[side.rule()].is_global()) {
      return is_added;
    }
    auto const& p = classes.locations_[get_stop(rs).location_];
    for (auto u = unit_idx_t{0U}; u != p.signature_.size(); ++u) {
      if (std::ranges::binary_search(p.signature_[u], side)) {
        for (auto const& [o_rs, o] : p.transports_[u]) {
          is_added |= add_fact(o_rs, o, side, true);
        }
      }
    }
    return is_added;
  };

  // Whether the transport fits into another (non-empty) part of its route
  // with the side forbidden: then forbidding it needs no new route.
  auto part_transport = std::vector<route_transport_idx_t>{};
  auto const can_move = [&](route_stop const rs, route_transport_idx_t const t,
                            transfer_rule_side_idx const side) {
    auto const& route = classes.routes_[rs.route_];
    part_transport.assign(route.n_parts_, route_transport_idx_t::invalid());
    for (auto a = route_transport_idx_t{0U}; a != route.part_.size(); ++a) {
      part_transport[to_idx(route.part_[a])] = a;
    }
    for (auto part = part_idx_t{0U}; part != route.n_parts_; ++part) {
      auto const o = part_transport[to_idx(part)];
      if (part == route.part_[t] || o == route_transport_idx_t::invalid()) {
        continue;
      }
      auto const fits = utl::all_of(route.stops_, [&](auto const& entry) {
        auto const& [i, c] = entry;
        auto const& m = c.members_[c.member_[t]];
        auto const& required = c.required_[part];
        return m.group_ == c.members_[c.member_[o]].group_ &&
               !intersects(m.forbidden_, required) &&
               !intersects(m.required_, c.forbidden_[part]) &&
               (i != rs.idx_ || !std::ranges::binary_search(required, side));
      });
      if (fits) {
        return true;
      }
    }
    return false;
  };

  // Sharing: a route part at a stop takes the location (class) of others at
  // the stop location that has none of the sides it forbids. A class grows
  // by sides of rules that are not global if none of its route parts forbids
  // them; global rules are the same for all stations: a route part only
  // inherits their sides from a class with all its own. Pairs connected by
  // inherited sides are checked as well.
  struct location_class {
    stop_signature_t sides_, forbidden_;
    std::vector<pair<route_stop, part_idx_t>> members_;
  };
  auto location_classes =
      std::vector<std::vector<location_class>>(classes.locations_.size());
  auto const set_sides = [&](route_stop const rs, part_idx_t const part,
                             stop_signature_t const& sides) {
    auto& c = get_stop(rs);
    if (c.sides_[part] == sides) {
      return;
    }
    c.sides_[part] = sides;
    c.changed_ = ++stamp;
    for (auto const side : sides) {
      if (!std::ranges::binary_search(c.required_[part], side)) {
        connect(rs, side);
      }
    }
  };
  auto grow = stop_signature_t{};
  auto const place = [&](route_stop const rs, part_idx_t const part) {
    auto& c = get_stop(rs);
    auto const& required = c.required_[part];
    auto const& forbidden = c.forbidden_[part];
    auto& ks = location_classes[c.location_];
    for (auto& k : ks) {
      std::erase(k.members_, pair{rs, part});
    }
    auto best = end(ks);
    auto best_cost = std::numeric_limits<std::size_t>::max();
    for (auto it = begin(ks); it != end(ks); ++it) {
      grow.clear();
      std::ranges::set_difference(required, it->sides_,
                                  std::back_inserter(grow));
      if (intersects(it->sides_, forbidden) ||
          intersects(grow, it->forbidden_) ||
          utl::any_of(grow, [&](transfer_rule_side_idx const side) {
            return rules[side.rule()].is_global();
          })) {
        continue;
      }
      auto const cost =
          it->sides_.size() + grow.size() * (it->members_.size() + 1U);
      if (cost < best_cost) {
        best = it;
        best_cost = cost;
      }
    }
    if (best == end(ks)) {
      ks.emplace_back().sides_ = required;
      best = std::prev(end(ks));
    }
    auto& k = *best;
    k.forbidden_.insert(end(k.forbidden_), begin(forbidden), end(forbidden));
    utl::erase_duplicates(k.forbidden_);
    k.sides_.insert(end(k.sides_), begin(required), end(required));
    utl::erase_duplicates(k.sides_);
    k.members_.push_back({rs, part});
    for (auto const& [m_rs, m_part] : k.members_) {
      set_sides(m_rs, m_part, k.sides_);
    }
  };

  // Resolve conflicts until there are none. A pair whose route stops kept
  // their parts and sides since it was checked has no conflict. Sides
  // inherited at both ends wait until no other fact is left (they often
  // resolve it): then they are forbidden where the transport fits into
  // another part of its route (the departure if both or none do). Then the
  // same for sharing: an inherited side that applies is forbidden for its
  // route part at the stop, which moves.
  auto conflicts = std::vector<change>{};
  auto n_pass = 0U;
  auto n_stuck = 0U;
  auto n_added = std::size_t{0U};
  auto n_forbidden = std::size_t{0U};
  for (auto const is_sharing : {false, true}) {
    if (is_sharing) {
      auto parts = std::vector<
          std::tuple<std::uint32_t, std::size_t, route_stop, part_idx_t>>{};
      for (auto r = route_idx_t{0U}; r != classes.routes_.size(); ++r) {
        for (auto const& [k, c] : classes.routes_[r].stops_) {
          for (auto part = part_idx_t{0U}; part != c.required_.size(); ++part) {
            if (!c.required_[part].empty()) {
              parts.emplace_back(c.location_,
                                 std::numeric_limits<std::size_t>::max() -
                                     c.required_[part].size(),
                                 route_stop{r, k}, part);
            }
          }
        }
      }
      utl::sort(parts);
      for (auto const& [_, __, rs, part] : parts) {
        place(rs, part);
      }
    }
    for (auto is_changed = true, allow_both = false; is_changed;) {
      is_changed = false;
      ++n_pass;
      auto n_pending = 0U;
      auto const is_allow_pass = allow_both;
      for (auto i = std::size_t{0U}; i != connected.size(); ++i) {
        auto const [x, y] = connected[i];
        if (get_stop(x).changed_ <= checked[i] &&
            get_stop(y).changed_ <= checked[i]) {
          continue;
        }
        auto is_pending = false;
        for (;;) {
          find_conflicts(tt, rule_times, classes, x, y, conflicts);
          auto is_progress = false;
          for (auto const& c : conflicts) {
            auto const has = [&](route_stop const rs,
                                 route_transport_idx_t const t,
                                 transfer_rule_side_idx const side) {
              auto const& s = get_stop(rs);
              return is_required(s.members_[s.member_[t]].required_, side);
            };

            // sharing: the route part where the side is inherited moves
            if (is_sharing) {
              auto const part = [&](route_stop const rs,
                                    route_transport_idx_t const t) {
                return classes.routes_[rs.route_].part_[t];
              };
              auto const is_inherited = [&](route_stop const rs,
                                            route_transport_idx_t const t,
                                            transfer_rule_side_idx const side) {
                auto const& o = get_stop(rs);
                return std::ranges::binary_search(o.sides_[part(rs, t)],
                                                  side) &&
                       !std::ranges::binary_search(o.required_[part(rs, t)],
                                                   side);
              };
              auto const from =
                  transfer_rule_side_idx{c.via_, event_type::kArr};
              auto const to = transfer_rule_side_idx{c.via_, event_type::kDep};
              auto const is_from = is_inherited(x, c.from_, from);
              auto const rs = is_from ? x : y;
              auto const t = is_from ? c.from_ : c.to_;
              auto const side = is_from ? from : to;
              if ((is_from || is_inherited(y, c.to_, to)) &&
                  insert(get_stop(rs).forbidden_[part(rs, t)], side)) {
                place(rs, part(rs, t));
                is_progress = true;
                ++n_forbidden;
              }
              continue;
            }

            // the expected rule is missing: both transports need its sides
            if (c.expected_ != transfer_rule_idx_t::invalid()) {
              auto const from =
                  transfer_rule_side_idx{c.expected_, event_type::kArr};
              auto const to =
                  transfer_rule_side_idx{c.expected_, event_type::kDep};
              auto const is_from =
                  !has(x, c.from_, from) && add_required(x, c.from_, from);
              auto const is_to =
                  !has(y, c.to_, to) && add_required(y, c.to_, to);
              if (is_from || is_to) {
                n_added += (is_from ? 1U : 0U) + (is_to ? 1U : 0U);
                is_progress = true;
                continue;
              }
            }

            // an inherited side applies (none: this batch's facts supply the
            // expected rule): forbid it where it is inherited
            if (c.via_ == transfer_rule_idx_t::invalid()) {
              continue;
            }
            auto const from = transfer_rule_side_idx{c.via_, event_type::kArr};
            auto const to = transfer_rule_side_idx{c.via_, event_type::kDep};
            auto is_to = !has(y, c.to_, to);
            auto const is_both = is_to && !has(x, c.from_, from);
            if (is_both && !allow_both) {
              is_pending = true;
              continue;
            }
            if (is_both) {
              is_to = can_move(y, c.to_, to) || !can_move(x, c.from_, from);
            }
            if (add_fact(is_to ? y : x, is_to ? c.to_ : c.from_,
                         is_to ? to : from, false)) {
              is_progress = true;
              ++n_forbidden;
            }
          }
          if (!is_progress) {
            n_pending += is_pending ? 1U : 0U;
            n_stuck += !conflicts.empty() && !is_pending ? 1U : 0U;
            break;
          }
          utl::erase_duplicates(touched);
          for (auto j = 0U; j != touched.size(); ++j) {
            if (j == 0U || touched[j].first != touched[j - 1U].first) {
              refresh(touched[j].first.route_, touched[j].first.idx_);
            }
          }
          for (auto const& [rs, part] : touched) {
            if (!is_consistent(rs, part)) {
              recolor(rs.route_);
            }
          }
          touched.clear();
          is_changed = true;
        }
        if (!is_pending) {
          checked[i] = stamp;
        }
      }
      if (is_allow_pass || (!is_changed && n_pending != 0U)) {
        allow_both = !is_allow_pass;
        is_changed = true;
      }
    }
  }

  if (n_stuck != 0U) {
    log(log_lvl::error, "loader.virtual_locations", "{} unresolved conflicts",
        n_stuck);
  }
  log(log_lvl::info, "loader.virtual_locations",
      "{} route stop pairs, {} passes ({} sides added, {} forbidden, {} route "
      "parts)",
      connected.size(), n_pass, n_added, n_forbidden,
      std::ranges::fold_left(classes.routes_, std::size_t{0U},
                             [](std::size_t const n, route_classes const& r) {
                               return n + r.n_parts_ - 1U;
                             }));
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

bool write(timetable& tt, classes_t const& classes, shapes_storage* shapes) {
  auto& tr = tt.transfer_rules_;
  auto virts =
      hash_map<pair<location_idx_t, std::vector<transfer_rule_side_idx>>,
               location_idx_t>{};

  auto origin = vector_map<route_idx_t, route_idx_t>{};
  auto route_transports =
      vector_map<route_idx_t, std::vector<transport_idx_t>>{};
  auto const n_routes = tt.n_routes();
  for (auto r = route_idx_t{0U}; r != n_routes; ++r) {
    auto const transports = tt.route_transport_ranges_[r];
    origin.push_back(r);
    route_transports.emplace_back(begin(transports), end(transports));
  }

  auto seqs = vector_map<route_transport_idx_t, stop_seq_t>{};
  for (auto r = route_idx_t{0U}; r != n_routes; ++r) {
    auto const& route = classes.routes_[r];
    if (route.stops_.empty()) {
      continue;
    }
    auto const transports = tt.route_transport_ranges_[r];
    auto const seq = stop_seq_t{tt.route_location_seq_[r].begin(),
                                tt.route_location_seq_[r].end()};
    seqs.clear();
    seqs.resize(transports.size(), seq);
    for (auto const& [k, c] : route.stops_) {
      auto const base = classes.locations_[c.location_].location_;
      auto const locations = utl::to_vec(c.sides_, [&](auto const& sides) {
        return sides.empty() ? base
                             : get_or_create_virt(tt, virts, base, sides);
      });
      for (auto u = route_transport_idx_t{0U}; u != c.member_.size(); ++u) {
        seqs[u][k] = stop{seqs[u][k]}
                         .with_location(locations[to_idx(route.part_[u])])
                         .value();
      }
    }

    auto parts =
        std::vector<std::pair<stop_seq_t, std::vector<transport_idx_t>>>{};
    for (auto u = route_transport_idx_t{0U}; u != seqs.size(); ++u) {
      auto it = utl::find_if(
          parts, [&](auto const& part) { return part.first == seqs[u]; });
      if (it == end(parts)) {
        it = parts.insert(end(parts), {seqs[u], {}});
      }
      it->second.push_back(transports.from_ + to_idx(u));
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
  auto classes = get_signatures(tt);
  resolve_conflicts(tt, classes);
  return write(tt, classes, shapes);
}

}  // namespace nigiri::loader
