#include <map>

#include "nigiri/loader/gtfs/transfer_rules.h"

#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <optional>
#include <ranges>
#include <vector>

#include "utl/erase.h"
#include "utl/erase_duplicates.h"
#include "utl/erase_if.h"
#include "utl/get_or_create.h"
#include "utl/helpers/algorithm.h"
#include "utl/pairwise.h"
#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"
#include "utl/progress_tracker.h"

#include "nigiri/loader/register.h"
#include "nigiri/loader/transfer_rules.h"
#include "nigiri/logging.h"
#include "nigiri/stop.h"
#include "nigiri/timetable.h"

namespace nigiri::loader::gtfs {

// ranking of specificity acc. to the GTFS reference (least specific first)
enum class specificity : std::uint8_t {
  kStopsOnly,
  kOneRoute,
  kBothRoutes,
  kOneTrip,
  kTripAndRoute,
  kBothTrips
};

// The precedence key handed to the shared rule writer: the GTFS ladder first,
// then whether the rule named the stops exactly rather than their station.
std::uint16_t rank(specificity const s, std::uint8_t const n_exact_stops) {
  return static_cast<std::uint16_t>((static_cast<std::uint16_t>(s) << 2U) |
                                    n_exact_stops);
}

enum class transfer_type : std::uint8_t {
  kRecommended = 0U,
  kTimed = 1U,
  kMinimumChangeTime = 2U,
  kNotPossible = 3U,
  kStaySeated = 4U,
  kNoStaySeated = 5U,
};

struct csv_transfer {
  utl::csv_col<utl::cstr, UTL_NAME("from_stop_id")> from_stop_id_;
  utl::csv_col<utl::cstr, UTL_NAME("to_stop_id")> to_stop_id_;
  utl::csv_col<int, UTL_NAME("transfer_type")> transfer_type_;
  utl::csv_col<std::optional<int>, UTL_NAME("min_transfer_time")>
      min_transfer_time_;
  utl::csv_col<utl::cstr, UTL_NAME("from_route_id")> from_route_id_;
  utl::csv_col<utl::cstr, UTL_NAME("to_route_id")> to_route_id_;
  utl::csv_col<utl::cstr, UTL_NAME("from_trip_id")> from_trip_id_;
  utl::csv_col<utl::cstr, UTL_NAME("to_trip_id")> to_trip_id_;
};

struct rule {
  rule(csv_transfer const& t,
       transfer_type const type,
       stops_map_t const& stops,
       route_map_t const& routes,
       trip_data const& trips)
      : forbidden_{type == transfer_type::kNotPossible},
        time_{(t.min_transfer_time_->value_or(0) + 59) / 60} {
    auto const resolve_stop = [&](utl::cstr const& id) {
      auto const it = stops.find(id.view());
      if (it == end(stops)) {
        log(log_lvl::error, "loader.gtfs.transfers", "stop {:?} not found",
            id.view());
        ok_ = false;
        return location_idx_t::invalid();
      }
      return it->second;
    };
    // route/trip qualifiers are optional: an empty id means "not scoped"
    auto const resolve_opt = [&](utl::cstr const& id, auto const& map,
                                 auto const invalid, auto&& get) {
      if (id.empty()) {
        return invalid;
      }
      auto const it = map.find(id.view());
      if (it == end(map)) {
        ok_ = false;
        return invalid;
      }
      return get(it);
    };

    from_stop_ = resolve_stop(*t.from_stop_id_);
    to_stop_ = resolve_stop(*t.to_stop_id_);
    from_route_ =
        resolve_opt(*t.from_route_id_, routes, route_id_idx_t::invalid(),
                    [](auto const& it) { return it->second->route_id_idx_; });
    to_route_ =
        resolve_opt(*t.to_route_id_, routes, route_id_idx_t::invalid(),
                    [](auto const& it) { return it->second->route_id_idx_; });
    from_trip_ =
        resolve_opt(*t.from_trip_id_, trips.trips_, gtfs_trip_idx_t::invalid(),
                    [](auto const& it) { return it->second; });
    to_trip_ =
        resolve_opt(*t.to_trip_id_, trips.trips_, gtfs_trip_idx_t::invalid(),
                    [](auto const& it) { return it->second; });
  }

  // synthesized unqualified rule, see fold_pair_defaults
  rule(location_idx_t const from, location_idx_t const to, duration_t const d)
      : from_stop_{from}, to_stop_{to}, time_{d} {}

  specificity get_specificity() const {
    auto const from_trip = from_trip_ != gtfs_trip_idx_t::invalid();
    auto const to_trip = to_trip_ != gtfs_trip_idx_t::invalid();
    auto const from_route = from_route_ != route_id_idx_t::invalid();
    auto const to_route = to_route_ != route_id_idx_t::invalid();

    if (from_trip && to_trip) {
      return specificity::kBothTrips;
    } else if ((from_trip && to_route) || (from_route && to_trip)) {
      return specificity::kTripAndRoute;
    } else if (from_trip || to_trip) {
      return specificity::kOneTrip;
    } else if (from_route && to_route) {
      return specificity::kBothRoutes;
    } else if (from_route || to_route) {
      return specificity::kOneRoute;
    } else {
      return specificity::kStopsOnly;
    }
  }

  bool is_qualified() const {
    return get_specificity() != specificity::kStopsOnly;
  }

  duration_t duration() const {
    return forbidden_ ? footpath::kMaxDuration : time_;
  }

  location_idx_t from_stop_{location_idx_t::invalid()};
  location_idx_t to_stop_{location_idx_t::invalid()};
  route_id_idx_t from_route_{route_id_idx_t::invalid()};
  route_id_idx_t to_route_{route_id_idx_t::invalid()};
  gtfs_trip_idx_t from_trip_{gtfs_trip_idx_t::invalid()};
  gtfs_trip_idx_t to_trip_{gtfs_trip_idx_t::invalid()};
  bool forbidden_{false};
  duration_t time_{0};
  bool ok_{true};
};

using rule_vec_t = vector_map<rule_idx_t, rule>;

void apply_rules(timetable& tt, rule_vec_t const& rules, trip_data& trips) {
  auto const base_of = [&](location_idx_t const l) {
    return tt.locations_.types_[l] == location_type::kVirt
               ? tt.locations_.parents_[l]
               : l;
  };

  // Map (route -> trips) for the routes referenced by a rule.
  auto route_trips = hash_map<route_id_idx_t, std::vector<gtfs_trip_idx_t>>{};
  for (auto const& r : rules) {
    for (auto const route : {r.from_route_, r.to_route_}) {
      if (route != route_id_idx_t::invalid()) {
        route_trips.emplace(route, std::vector<gtfs_trip_idx_t>{});
      }
    }
  }
  if (!route_trips.empty()) {
    for (auto i = gtfs_trip_idx_t{0U}; i != trips.data_.size(); ++i) {
      auto const it = route_trips.find(trips.data_[i].route_);
      if (it != end(route_trips)) {
        it->second.emplace_back(i);
      }
    }
  }

  // Map (trip, stop position) -> matched rule sides.
  auto trip_stop_signatures =
      hash_map<pair<gtfs_trip_idx_t, stop_idx_t>, signature_t>{};
  for (auto rule_idx = rule_idx_t{0U}; rule_idx != rules.size(); ++rule_idx) {
    auto const& r = rules[rule_idx];
    auto const match_side = [&](location_idx_t const rule_stop,
                                route_id_idx_t const route,
                                gtfs_trip_idx_t const trip,
                                bool const is_from_side) {
      if (route == route_id_idx_t::invalid() &&
          trip == gtfs_trip_idx_t::invalid()) {
        return;  // unqualified side -> no split
      }

      auto const match_trip = [&](gtfs_trip_idx_t const trp_idx) {
        auto const& t = trips.data_[trp_idx];
        if (!t.flex_stops_.empty()) {
          return;
        }
        // A trip that never runs produces no transports, so splitting its
        // stops off would leave virtual locations nothing ever reaches.
        if (t.service_ != nullptr && t.service_->none()) {
          return;
        }
        auto const n_stops = static_cast<stop_idx_t>(t.stop_seq_.size());
        for (auto pos = stop_idx_t{0U}; pos != n_stops; ++pos) {
          if (covers(tt, rule_stop, stop{t.stop_seq_[pos]}.location_idx())) {
            trip_stop_signatures[{trp_idx, pos}].push_back(
                side_ref(rule_idx, is_from_side));
          }
        }
      };

      if (trip != gtfs_trip_idx_t::invalid()) {
        match_trip(trip);
      } else {
        for (auto const trp_idx : route_trips.at(route)) {
          match_trip(trp_idx);
        }
      }
    };

    match_side(r.from_stop_, r.from_route_, r.from_trip_, true);
    match_side(r.to_stop_, r.to_route_, r.to_trip_, false);
  }

  // Update all trip stops to virtual locations.
  //
  // Virtual locations are staged rather than registered right away: they get
  // the index they would receive on registration, everything below works with
  // that index, and only the survivors are written into the timetable at the
  // end. Nothing has to be renumbered afterwards.
  auto const first_virt = location_idx_t{tt.n_locations()};
  auto staged = std::vector<location>{};
  auto virt_locs =
      hash_map<std::pair<location_idx_t, signature_t>, location_idx_t>{};
  auto side_virts = hash_map<sided_rule_idx_t, std::vector<location_idx_t>>{};
  // which trip stops at which virtual location, and at which position - the
  // time-aware merge below needs the events to decide whether a pair it would
  // create is one the timetable could ever use
  auto virt_stops =
      hash_map<location_idx_t,
               std::vector<std::pair<gtfs_trip_idx_t, std::size_t>>>{};
  for (auto& [trip_stop, sig] : trip_stop_signatures) {
    auto const [trp_idx, pos] = trip_stop;
    auto& t = trips.data_[trp_idx];
    auto const s = stop{t.stop_seq_[pos]};
    auto const base = s.location_idx();
    utl::erase_duplicates(sig);

    auto const virt =
        utl::get_or_create(virt_locs, std::pair{base, sig}, [&]() {
          auto l = location{};
          l.src_ = tt.locations_.src_[base];
          l.pos_ = tt.locations_.coordinates_[base];
          l.type_ = location_type::kVirt;
          l.parent_ = base;
          l.transfer_time_ = tt.locations_.transfer_time_[base];

          // Rules where from == to side (e.g. transfers from route R -> R)
          // end up on the same virt node -> set transfer_time to self.
          auto best = std::optional<candidate>{};
          for (auto i = std::size_t{1U}; i < sig.size(); ++i) {
            auto const from = sig[i - 1U];
            auto const to = sig[i];
            if (rule_of(from) != rule_of(to)) {
              continue;
            }
            auto const rule_idx = rule_of(from);
            auto const& r = rules[rule_idx];
            auto const c =
                candidate{.rank_ = rank(r.get_specificity(),
                                        static_cast<std::uint8_t>(
                                            (r.from_stop_ == base) +
                                            (r.to_stop_ == base))),
                          .rule_idx_ = rule_idx};
            if (!best.has_value() || *best < c) {
              best = c;
            }
          }
          if (best.has_value()) {
            l.transfer_time_ = rules[best->rule_idx_].duration();
          }

          auto const v = location_idx_t{cista::to_idx(first_virt) +
                                       static_cast<std::uint32_t>(
                                           staged.size())};
          staged.push_back(l);
          tt.locations_.children_[base].emplace_back(v);

          for (auto const side : sig) {
            side_virts[side].push_back(v);
          }

          return v;
        });

    virt_stops[virt].emplace_back(trp_idx, pos);
    t.stop_seq_[pos] =
        stop{virt, s.in_allowed(), s.out_allowed(), s.in_allowed_wheelchair(),
             s.out_allowed_wheelchair()}
            .value();
  }

  // Materialize the staged locations. They are appended in staging order, so
  // the indices handed out above are the ones they end up with.
  for (auto const& l : staged) {
    auto const registered = register_location(tt, l);
    utl::verify(cista::to_idx(registered) ==
                    cista::to_idx(first_virt) +
                        static_cast<std::uint32_t>(&l - staged.data()),
                "virtual location {} was not registered at its staged index",
                registered);
  }

  // Let the rules compete for specificity on all location pairs they apply to.
  // An unqualified side applies to the rule stop and everything below it, a
  // qualified one only to the virtual locations that were split off for it.
  auto const for_each =
      [&](rule_idx_t const rule_idx, location_idx_t const rule_stop,
          route_id_idx_t const route, gtfs_trip_idx_t const trip,
          bool const from_side, auto&& fn) {
        if (route == route_id_idx_t::invalid() &&
            trip == gtfs_trip_idx_t::invalid()) {
          fn(rule_stop);
          for (auto const child : tt.locations_.children_[rule_stop]) {
            fn(child);
            for (auto const grand_child : tt.locations_.children_[child]) {
              fn(grand_child);
            }
          }
          return;
        }

        auto const it = side_virts.find(side_ref(rule_idx, from_side));
        if (it != end(side_virts)) {
          for (auto const virt : it->second) {
            fn(virt);
          }
        }
      };

  auto most_specific = hash_map<transfer_pair, candidate>{};
  for (auto rule_idx = rule_idx_t{0U}; rule_idx != rules.size(); ++rule_idx) {
    auto const& r = rules[rule_idx];
    auto const update = [&](location_idx_t const x, location_idx_t const y) {
      if (x == y) {
        return;
      }
      auto const c = candidate{
          .rank_ = rank(r.get_specificity(),
                        static_cast<std::uint8_t>((r.from_stop_ == base_of(x)) +
                                                  (r.to_stop_ == base_of(y)))),
          .rule_idx_ = rule_idx};
      auto const [it, is_new] = most_specific.emplace(transfer_pair{x, y}, c);
      if (!is_new) {
        it->second = std::max(it->second, c);
      }
    };

    for_each(rule_idx, r.from_stop_, r.from_route_, r.from_trip_, true,
             [&](location_idx_t const x) {
               for_each(rule_idx, r.to_stop_, r.to_route_, r.to_trip_, false,
                        [&](location_idx_t const y) { update(x, y); });
             });
  }

  auto durations = vector_map<rule_idx_t, duration_t>{};
  durations.reserve(rules.size());
  for (auto const& r : rules) {
    durations.push_back(r.duration());
  }

  write_transfer_rules(tt, most_specific, durations, first_virt);

  // Two virtual locations of one stop are the same node if nothing the routing
  // can observe tells them apart: their own transfer time, the cells written
  // for them in either direction, and the hubs they sit in. What splits them
  // is the rule sides they matched, and different sides regularly end up
  // stating the same thing - so the trips of one move to the other and the
  // duplicate is retired, before it can split a route. This runs on the
  // written state, after elision: cells that a hub derives are gone by now, so
  // two locations differing only in those are correctly seen as one.
  if (std::getenv("NIGIRI_NO_VIRT_MERGE") == nullptr) {
    auto const n = tt.n_locations();
    constexpr auto const kSelf = 0xFFFFFFFFU;

    auto cols = hash_map<location_idx_t, std::vector<std::pair<std::uint32_t, std::int32_t>>>{};
    auto rows = cols;
    for (auto l = location_idx_t{0U};
         l != location_idx_t{std::min(
             static_cast<std::size_t>(tt.locations_.transfer_rule_fps_.size()),
             static_cast<std::size_t>(cista::to_idx(n)))};
         ++l) {
      for (auto const fp : tt.locations_.transfer_rule_fps_[l]) {
        if (tt.locations_.types_[l] == location_type::kVirt) {
          rows[l].emplace_back(fp.target() == l ? kSelf : to_idx(fp.target()),
                               fp.duration().count());
        }
        if (tt.locations_.types_[fp.target()] == location_type::kVirt) {
          cols[fp.target()].emplace_back(l == fp.target() ? kSelf : to_idx(l),
                                         fp.duration().count());
        }
      }
    }

    auto hub_mem = hash_map<location_idx_t, std::vector<std::uint32_t>>{};
    for (auto h = 0U; h != tt.locations_.hub_in_[kDefaultProfile].size(); ++h) {
      auto const hi = hub_idx_t{h};
      for (auto const m : tt.locations_.hub_in_[kDefaultProfile][hi]) {
        hub_mem[m].push_back(h * 2U);
      }
      for (auto const m : tt.locations_.hub_out_[kDefaultProfile][hi]) {
        hub_mem[m].push_back(h * 2U + 1U);
      }
    }

    // Cells that two candidates write about each other are not an observable
    // that tells them apart: merged, the two become one node and the pair
    // becomes a self-edge, which the route scan already applies as that node's
    // own transfer time. That holds only if the pair says exactly that, so a
    // pair stating anything else keeps them apart.
    auto const cells_without = [&](auto const& m, location_idx_t const v,
                                   location_idx_t const partner) {
      auto out = std::vector<std::pair<std::uint32_t, std::int32_t>>{};
      if (auto const it = m.find(v); it != end(m)) {
        for (auto const& [t, d] : it->second) {
          if (t != to_idx(partner)) {
            out.emplace_back(t, d);
          }
        }
      }
      utl::sort(out);
      return out;
    };
    auto const pair_is_own_time = [&](location_idx_t const a,
                                      location_idx_t const b) {
      auto const own = tt.locations_.transfer_time_[a].count();
      // A pair nobody wrote a rule about is not free: it costs what the stop
      // charges for an unwritten transfer. Merging turns the pair into a
      // self-edge at `own`, so an unstated pair may only be merged when `own`
      // is exactly what it already costs - otherwise a virt carrying a 0min
      // self-rule hands that 0 to every trip merged with it, and the rules
      // never said those trips may transfer at all.
      auto const dflt =
          tt.locations_.transfer_time_[tt.locations_.parents_[a]].count();
      auto const states_own_time = [&](location_idx_t const x,
                                       location_idx_t const y) {
        auto stated = false;
        if (auto const it = rows.find(x); it != end(rows)) {
          for (auto const& [t, d] : it->second) {
            if (t == to_idx(y)) {
              stated = true;
              if (d != own) {
                return false;
              }
            }
          }
        }
        return stated || own == dflt;
      };
      return states_own_time(a, b) && states_own_time(b, a);
    };
    auto const hubs_of = [&](location_idx_t const v) {
      auto out = std::vector<std::uint32_t>{};
      if (auto const it = hub_mem.find(v); it != end(hub_mem)) {
        out = it->second;
      }
      utl::sort(out);
      return out;
    };
    // A merge fuses two nodes, so the rule between one of them and a partner
    // is handed to the other as well. That is only harmless when the pair it
    // creates is one the timetable cannot use anyway: the partner has already
    // left, or the gap is wide enough that the stop's own transfer time
    // already allowed it. Then the rule grants nothing and the fused node
    // answers exactly as the two did.
    //
    // Off by default: it makes the merge depend on the event times, so two
    // timetables over different date ranges can merge differently.
    // The margin is how much delay a merge has to survive: a created pair only
    // starts to matter once the feeder is late by more than gap - default. A
    // constant is the wrong shape for it - the same number is unreachable at a
    // stop served every four minutes and trivial at one served hourly. What it
    // wants to express is "far enough that the next departure of the service is
    // the relevant one", so it is taken from the stop's own headway. Negative
    // means a fixed margin in minutes instead, for A/B.
    static auto const time_merge = []() -> std::optional<int> {
      auto const* v = std::getenv("NIGIRI_VIRT_TIME_MERGE");
      return v == nullptr ? std::nullopt : std::optional{std::atoi(v)};
    }();

    auto stop_headway = hash_map<location_idx_t, duration_t>{};
    if (time_merge.has_value() && *time_merge >= 0) {
      auto deps = hash_map<location_idx_t, std::vector<minutes_after_midnight_t>>{};
      for (auto const& t : trips.data_) {
        for (auto const [i, s] : utl::enumerate(t.stop_seq_)) {
          if (i >= t.event_times_.size()) {
            break;
          }
          auto const d = t.event_times_[i].dep_;
          if (d != kInterpolate) {
            deps[stop{s}.location_idx()].push_back(d);
          }
        }
      }
      for (auto& [l, v] : deps) {
        if (v.size() < 2U) {
          continue;
        }
        utl::sort(v);
        auto gaps = std::vector<duration_t>{};
        for (auto i = std::size_t{1U}; i != v.size(); ++i) {
          if (v[i] > v[i - 1U]) {
            gaps.push_back(v[i] - v[i - 1U]);
          }
        }
        if (gaps.empty()) {
          continue;
        }
        std::nth_element(begin(gaps), begin(gaps) + gaps.size() / 2, end(gaps));
        stop_headway[l] = gaps[gaps.size() / 2U];  // the median headway
      }
    }
    auto const margin_at = [&](location_idx_t const base) {
      if (*time_merge < 0) {
        return duration_t{static_cast<duration_t::rep>(-*time_merge)};
      }
      auto const it = stop_headway.find(base);
      return it == end(stop_headway) ? duration_t{*time_merge} : it->second;
    };

    // What a hub derives, decided the way build_hubs decides it: a cell longer
    // than the stop charges makes its source a restricted one and its target
    // one that the restricted hub drops. Without this the merge has to guess
    // what an unwritten pair costs, and "it costs the default" is wrong for
    // exactly the pairs elision left out.
    auto slow_from_m = hash_set<location_idx_t>{};
    auto slow_to_m = hash_set<location_idx_t>{};
    if (time_merge.has_value()) {
      auto const n_cells = std::min(
          static_cast<std::size_t>(tt.locations_.transfer_rule_fps_.size()),
          static_cast<std::size_t>(cista::to_idx(n)));
      for (auto l = location_idx_t{0U}; l != location_idx_t{n_cells}; ++l) {
        auto const b = tt.locations_.types_[l] == location_type::kVirt
                           ? tt.locations_.parents_[l]
                           : l;
        for (auto const fp : tt.locations_.transfer_rule_fps_[l]) {
          auto const tb = tt.locations_.types_[fp.target()] == location_type::kVirt
                              ? tt.locations_.parents_[fp.target()]
                              : fp.target();
          if (tb == b && fp.duration() > duration_t{tt.locations_.transfer_time_[b]}) {
            slow_from_m.insert(l);
            slow_to_m.insert(fp.target());
          }
        }
      }
    }
    auto const hub_derives = [&](location_idx_t const x, location_idx_t const y,
                                 location_idx_t const base) {
      if (x != base && tt.locations_.transfer_time_[x] >
                           tt.locations_.transfer_time_[base]) {
        return false;  // slow member: in neither in-list
      }
      return !slow_from_m.contains(x) || !slow_to_m.contains(y);
    };

    // what `x` already pays to reach `y`, or nothing if it cannot
    auto const current_cost = [&](location_idx_t const x, location_idx_t const y,
                                  location_idx_t const base,
                                  auto const& cells_of_fn) {
      auto best = std::optional<duration_t>{};
      for (auto const& [t, d] : cells_of_fn(rows, x)) {
        if (t != kSelf && location_idx_t{t} == y) {
          auto const dd = duration_t{static_cast<duration_t::rep>(d)};
          best = best.has_value() ? std::min(*best, dd) : dd;
        }
      }
      if (hub_derives(x, y, base)) {
        auto const dflt = duration_t{tt.locations_.transfer_time_[base]};
        best = best.has_value() ? std::min(*best, dflt) : dflt;
      }
      return best;
    };

    auto const events_at = [&](location_idx_t const v) {
      auto out = std::vector<stop_events>{};
      if (auto const it = virt_stops.find(v); it != end(virt_stops)) {
        for (auto const& [trp, pos] : it->second) {
          out.push_back(trips.data_[trp].event_times_[pos]);
        }
      }
      return out;
    };

    // every trip pair of x -> y is either impossible or already allowed
    auto const grants_nothing = [&](location_idx_t const x,
                                    location_idx_t const y,
                                    duration_t const need_base,
                                    duration_t const margin) {
      auto const need = need_base + margin;
      auto const from = events_at(x), to = events_at(y);
      if (from.empty() || to.empty()) {
        return false;
      }
      for (auto const& f : from) {
        for (auto const& t : to) {
          if (f.arr_ == kInterpolate || t.dep_ == kInterpolate) {
            return false;
          }
          auto const gap = t.dep_ - f.arr_;
          if (gap < duration_t{0} || gap >= need) {
            continue;  // gone by then, or wide enough that nothing changes
          }
          return false;
        }
      }
      return true;
    };

    auto const cells_of = [&](auto const& m, location_idx_t const v) {
      auto out = std::vector<std::pair<std::uint32_t, std::int32_t>>{};
      if (auto const it = m.find(v); it != end(m)) {
        out = it->second;
      }
      return out;
    };

    // may `a` take over every cell of `b` (and the other way round) without
    // changing an answer?
    auto const time_safe = [&](location_idx_t const a, location_idx_t const b) {
      if (!time_merge.has_value() ||
          tt.locations_.transfer_time_[a] != tt.locations_.transfer_time_[b] ||
          tt.locations_.parents_[a] != tt.locations_.parents_[b]) {
        return false;
      }
      if (!pair_is_own_time(a, b)) {
        return false;
      }
      if (hubs_of(a) != hubs_of(b)) {
        // the hubs were emitted against these indices before the merge ran, so
        // fusing two nodes that sit in different ones drops the transfers only
        // one of them was reached by - the merge would lose journeys, not gain
        // them
        return false;
      }
      auto const base = tt.locations_.parents_[a];
      auto const dflt = duration_t{tt.locations_.transfer_time_[base]};
      auto const margin = margin_at(base);
      // A cell the other side already states identically is not granted by the
      // merge - both nodes carry it before and after. Every virtual location of
      // a stop inherits that stop's own rules this way, so this is most of
      // them.
      auto const already_has = [&](auto const& m, location_idx_t const l,
                                   std::uint32_t const partner,
                                   std::int32_t const d) {
        auto const it = m.find(l);
        return it != end(m) &&
               utl::any_of(it->second, [&](auto const& c) {
                 return c.first == partner && c.second == d;
               });
      };
      auto const check = [&](location_idx_t const owner,
                             location_idx_t const other) {
        // Each cell of `owner` is handed to `other` by the merge. It is
        // harmless when `other` already reaches that partner at least as
        // cheaply - then the minimum does not move - and otherwise only when
        // the timetable cannot tell: the partner has gone by, or the gap is
        // wide enough that what `other` already pays was enough anyway. A
        // partner `other` cannot reach at all has no such gap, so a cell to it
        // is never granted.
        auto const granted = [&](location_idx_t const partner,
                                 std::int32_t const d, bool const outgoing) {
          auto const dd = duration_t{static_cast<duration_t::rep>(d)};
          auto const cur = outgoing ? current_cost(other, partner, base, cells_of)
                                    : current_cost(partner, other, base, cells_of);
          if (cur.has_value() && dd >= *cur) {
            return true;  // the minimum does not move
          }
          if (!cur.has_value()) {
            return false;  // unreachable today: the merge would create it
          }
          return outgoing ? grants_nothing(other, partner, *cur, margin)
                          : grants_nothing(partner, other, *cur, margin);
        };
        for (auto const& [t, d] : cells_of(rows, owner)) {
          if (already_has(rows, other, t, d)) {
            continue;  // states the same thing already
          }
          if (t == kSelf || location_idx_t{t} == other) {
            continue;  // becomes the fused node's own time, see above
          }
          auto const target = location_idx_t{t};
          if (tt.locations_.parents_[target] != base && target != base) {
            return false;  // crosses to another stop: not modelled here
          }
          if (!granted(target, d, /*outgoing=*/true)) {
            return false;
          }
        }
        for (auto const& [f, d] : cells_of(cols, owner)) {
          if (already_has(cols, other, f, d)) {
            continue;
          }
          if (f == kSelf || location_idx_t{f} == other) {
            continue;
          }
          auto const src = location_idx_t{f};
          if (tt.locations_.parents_[src] != base && src != base) {
            return false;
          }
          if (!granted(src, d, /*outgoing=*/false)) {
            return false;
          }
        }
        return true;
      };
      return check(a, b) && check(b, a);
    };

    auto const same_node = [&](location_idx_t const a, location_idx_t const b) {
      return tt.locations_.transfer_time_[a] ==
                 tt.locations_.transfer_time_[b] &&
             hubs_of(a) == hubs_of(b) && pair_is_own_time(a, b) &&
             cells_without(rows, a, b) == cells_without(rows, b, a) &&
             cells_without(cols, a, b) == cells_without(cols, b, a);
    };

    // Merging is a minimum clique cover, not an equivalence: two nodes may
    // each be mergeable with a third and not with each other, because the
    // cells a merged pair writes about itself are excluded from the
    // comparison. So this is first-fit greedy, the usual heuristic. Measured
    // on Switzerland: ordering the candidates most-constrained-first and
    // requiring a match against every group member rather than only the
    // representative gives the identical 4,679 virtual locations, so the
    // simple form is not leaving anything on the table. Refining a partition
    // instead - what the HRD loader can do, because its rules never cross
    // stops - is not applicable here and lands at 6,478.
    auto group_members =
        hash_map<location_idx_t, std::vector<location_idx_t>>{};
    auto reps = hash_map<location_idx_t, std::vector<location_idx_t>>{};
    auto remap = hash_map<location_idx_t, location_idx_t>{};
    for (auto v = first_virt; v != n; ++v) {
      auto& base_reps = reps[tt.locations_.parents_[v]];
      auto const it = utl::find_if(base_reps, [&](location_idx_t const r) {
        if (same_node(v, r)) {
          return true;
        }
        if (!time_merge.has_value()) {
          return false;
        }
        // the group answers as one node, so the candidate has to be safe
        // against every member, not only the representative
        if (!time_safe(v, r)) {
          return false;
        }
        auto const& members = group_members[r];
        return utl::all_of(members, [&](location_idx_t const m) {
          return time_safe(v, m);
        });
      });
      if (it == end(base_reps)) {
        base_reps.push_back(v);
      } else {
        remap.emplace(v, *it);
        group_members[*it].push_back(v);
      }
    }

    if (!remap.empty()) {

      for (auto const& [trip_stop, sig] : trip_stop_signatures) {
        auto const [trp_idx, pos] = trip_stop;
        auto& t = trips.data_[trp_idx];
        auto const st = stop{t.stop_seq_[pos]};
        if (auto const it = remap.find(st.location_idx()); it != end(remap)) {
          t.stop_seq_[pos] = stop{it->second,
                                  st.in_allowed(),
                                  st.out_allowed(),
                                  st.in_allowed_wheelchair(),
                                  st.out_allowed_wheelchair()}
                                 .value();
        }
      }

      // A duplicate merged on equality states exactly what its representative
      // states, so dropping its cells and moving them are the same thing. The
      // time-aware merge fuses nodes that state different cells - certified
      // harmless, but only if they actually end up on the fused node - so the
      // cells move, and the ones pointing at a duplicate follow it.
      auto const rep_of = [&](location_idx_t const l) {
        auto const it = remap.find(l);
        return it == end(remap) ? l : it->second;
      };
      auto cells = mutable_fws_multimap<location_idx_t, footpath>{};
      auto seen = hash_map<std::pair<location_idx_t, location_idx_t>,
                           duration_t>{};
      for (auto l = location_idx_t{0U};
           l != location_idx_t{std::min(
               static_cast<std::size_t>(
                   tt.locations_.transfer_rule_fps_.size()),
               static_cast<std::size_t>(cista::to_idx(n)))};
           ++l) {
        auto const from = rep_of(l);
        for (auto const fp : tt.locations_.transfer_rule_fps_[l]) {
          auto const to = rep_of(fp.target());
          if (from == to) {
            continue;  // became the node's own time, which it already carries
          }
          auto const [it, ins] = seen.emplace(std::pair{from, to},
                                              fp.duration());
          if (ins) {
            cells[from].emplace_back(to, fp.duration());
          } else if (fp.duration() < it->second) {
            it->second = fp.duration();
            for (auto& existing : cells[from]) {
              if (existing.target() == to) {
                existing = footpath{to, fp.duration()};
              }
            }
          }
        }
      }
      tt.locations_.transfer_rule_fps_ = std::move(cells);

      auto hub_in = vecvec<hub_idx_t, location_idx_t>{};
      auto hub_out = hub_in;
      auto hub_time = vector_map<hub_idx_t, duration_t>{};
      auto keep = std::vector<location_idx_t>{};
      auto keep_out = std::vector<location_idx_t>{};
      for (auto h = 0U; h != tt.locations_.hub_in_[kDefaultProfile].size(); ++h) {
        auto const hi = hub_idx_t{h};
        keep.clear();
        keep_out.clear();
        for (auto const m : tt.locations_.hub_in_[kDefaultProfile][hi]) {
          keep.push_back(rep_of(m));  // the representative stands in for it
        }
        utl::erase_duplicates(keep);
        for (auto const m : tt.locations_.hub_out_[kDefaultProfile][hi]) {
          keep_out.push_back(rep_of(m));
        }
        utl::erase_duplicates(keep_out);
        if (keep.empty() || keep_out.empty()) {
          continue;
        }
        hub_in.emplace_back(keep);
        hub_out.emplace_back(keep_out);
        hub_time.push_back(tt.locations_.hub_time_[kDefaultProfile][hi]);
      }
      tt.locations_.hub_in_[kDefaultProfile] = std::move(hub_in);
      tt.locations_.hub_out_[kDefaultProfile] = std::move(hub_out);
      tt.locations_.hub_time_[kDefaultProfile] = std::move(hub_time);

      for (auto const& [dup, rep] : remap) {
        auto bucket = tt.locations_.children_[tt.locations_.parents_[dup]];
        utl::erase(bucket, dup);
      }

      // The duplicates are the last locations there are and nothing refers to
      // them any more, so the survivors move down into their slots and the
      // arrays end after them. Only the per-location arrays are shortened -
      // the multimaps keep their empty tail buckets, which cost an offset
      // each.
      auto compact = hash_map<location_idx_t, location_idx_t>{};
      auto next = first_virt;
      for (auto v = first_virt; v != n; ++v) {
        if (remap.contains(v)) {
          continue;
        }
        compact.emplace(v, next);
        if (next != v) {
          auto& loc = tt.locations_;
          loc.names_[next] = loc.names_[v];
          loc.platform_codes_[next] = loc.platform_codes_[v];
          loc.stop_codes_[next] = loc.stop_codes_[v];
          loc.descriptions_[next] = loc.descriptions_[v];
          loc.coordinates_[next] = loc.coordinates_[v];
          loc.src_[next] = loc.src_[v];
          loc.types_[next] = loc.types_[v];
          loc.location_timezones_[next] = loc.location_timezones_[v];
          loc.transfer_time_[next] = loc.transfer_time_[v];
          loc.parents_[next] = loc.parents_[v];
        }
        ++next;
      }

      auto const renumber = [&](location_idx_t const l) {
        auto const it = compact.find(l);
        return it == end(compact) ? l : it->second;
      };

      for (auto const& [trip_stop, sig] : trip_stop_signatures) {
        auto const [trp_idx, pos] = trip_stop;
        auto& t = trips.data_[trp_idx];
        auto const st = stop{t.stop_seq_[pos]};
        auto const moved = renumber(st.location_idx());
        if (moved != st.location_idx()) {
          t.stop_seq_[pos] = stop{moved,
                                  st.in_allowed(),
                                  st.out_allowed(),
                                  st.in_allowed_wheelchair(),
                                  st.out_allowed_wheelchair()}
                                 .value();
        }
      }

      auto renumbered = mutable_fws_multimap<location_idx_t, footpath>{};
      for (auto l = location_idx_t{0U};
           l != location_idx_t{std::min(
               static_cast<std::size_t>(
                   tt.locations_.transfer_rule_fps_.size()),
               static_cast<std::size_t>(cista::to_idx(n)))};
           ++l) {
        for (auto const fp : tt.locations_.transfer_rule_fps_[l]) {
          renumbered[renumber(l)].emplace_back(renumber(fp.target()),
                                               fp.duration());
        }
      }
      tt.locations_.transfer_rule_fps_ = std::move(renumbered);

      auto in2 = vecvec<hub_idx_t, location_idx_t>{};
      auto out2 = in2;
      auto members = std::vector<location_idx_t>{};
      for (auto h = 0U; h != tt.locations_.hub_in_[kDefaultProfile].size(); ++h) {
        auto const hi = hub_idx_t{h};
        members.clear();
        for (auto const m : tt.locations_.hub_in_[kDefaultProfile][hi]) {
          members.push_back(renumber(m));
        }
        in2.emplace_back(members);
        members.clear();
        for (auto const m : tt.locations_.hub_out_[kDefaultProfile][hi]) {
          members.push_back(renumber(m));
        }
        out2.emplace_back(members);
      }
      tt.locations_.hub_in_[kDefaultProfile] = std::move(in2);
      tt.locations_.hub_out_[kDefaultProfile] = std::move(out2);

      for (auto b = location_idx_t{0U}; b != first_virt; ++b) {
        for (auto& c : tt.locations_.children_[b]) {
          c = renumber(c);
        }
      }

      auto const n_left = cista::to_idx(next);
      tt.locations_.names_.resize(n_left);
      tt.locations_.platform_codes_.resize(n_left);
      tt.locations_.stop_codes_.resize(n_left);
      tt.locations_.descriptions_.resize(n_left);
      tt.locations_.coordinates_.resize(n_left);
      tt.locations_.src_.resize(n_left);
      tt.locations_.types_.resize(n_left);
      tt.locations_.location_timezones_.resize(n_left);
      tt.locations_.transfer_time_.resize(n_left);
      tt.locations_.parents_.resize(n_left);

      // The multimaps have no truncate, so the live buckets are copied into a
      // fresh one - otherwise their empty tails would be serialised.
      auto const shrink = [&](auto& m) {
        auto shrunk = std::decay_t<decltype(m)>{};
        for (auto l = location_idx_t{0U}; l != location_idx_t{n_left}; ++l) {
          auto bucket = shrunk[l];
          for (auto const& x : m[l]) {
            bucket.push_back(x);
          }
        }
        m = std::move(shrunk);
      };
      // ids_ is per location too, and build_lb_graph iterates *its* size -
      // leaving it long makes every consumer read past the other arrays
      {
        auto shrunk = vecvec<location_idx_t, char>{};
        for (auto l = location_idx_t{0U}; l != location_idx_t{n_left}; ++l) {
          shrunk.emplace_back(tt.locations_.ids_[l].view());
        }
        tt.locations_.ids_ = std::move(shrunk);
      }
      tt.locations_.ticketing_unavailable_.resize(n_left);

      shrink(tt.locations_.children_);
      shrink(tt.locations_.equivalences_);
      shrink(tt.locations_.preprocessing_footpaths_out_);
      log(log_lvl::info, "loader.gtfs.transfer_rules",
          "{} virtual locations merged into {}", remap.size(),
          static_cast<std::size_t>(cista::to_idx(n) -
                                   cista::to_idx(first_virt)) -
              remap.size());
    }
  }
}

// Detects the most common rule between two stops
// -> removes them and makes their min_transfer_time the new default

void read_transfers(timetable& tt,
                    std::string_view file_content,
                    stops_map_t const& stops,
                    route_map_t const& routes,
                    trip_data& trips) {
  // Capture transfer times before this feed's rules modify them.
  tt.locations_.sync_base_transfer_time();

  if (file_content.empty()) {
    return;
  }

  auto const timer = scoped_timer{"loader.gtfs.transfers"};

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Read Transfers").in_high(file_content.size());

  auto const wire_stay_seated = [&](csv_transfer const& t) {
    if (t.from_trip_id_->empty() || t.to_trip_id_->empty()) {
      log(log_lvl::error, "loader.gtfs.transfers",
          "stay seated transfers require from_trip_id and to_trip_id");
      return;
    }
    auto const from = trips.trips_.find(t.from_trip_id_->view());
    auto const to = trips.trips_.find(t.to_trip_id_->view());
    if (from == end(trips.trips_) || to == end(trips.trips_)) {
      log(log_lvl::error, "loader.gtfs.transfers", "trip {} not found",
          from == end(trips.trips_) ? t.from_trip_id_->view()
                                    : t.to_trip_id_->view());
      return;
    }
    // deduplicate: the CH feed introduces duplicate primary keys through an
    // additional service_id
    auto const push_unique = [](auto& vec, auto const value) {
      if (utl::find(vec, value) == end(vec)) {
        vec.push_back(value);
      }
    };
    push_unique(trips.data_[to->second].seated_in_, from->second);
    push_unique(trips.data_[from->second].seated_out_, to->second);
  };

  auto rules = rule_vec_t{};

  // Profiles that ignore the qualified rules still honor the plain minimum
  // transfer time of a stop: it is interchange time, not a walk, and street
  // routing cannot supply it. Folded like a loader without rule support does
  // it - minimum over every reflexive row, qualified or not, replacing the
  // default rather than capping it, and rounded down to whole minutes.
  constexpr auto const kNoRule = duration_t::max();
  auto reflexive_min = vector_map<location_idx_t, duration_t>{};
  reflexive_min.resize(tt.n_locations());
  std::fill(begin(reflexive_min), end(reflexive_min), kNoRule);

  utl::line_range{
      utl::make_buf_reader(file_content, progress_tracker->update_fn())}  //
      | utl::csv<csv_transfer>()  //
      |
      utl::for_each([&](csv_transfer const& t) {
        if (*t.transfer_type_ == static_cast<int>(transfer_type::kStaySeated)) {
          wire_stay_seated(t);
          return;
        }
        if (*t.transfer_type_ < 0 ||
            *t.transfer_type_ > static_cast<int>(transfer_type::kNotPossible)) {
          return;  // no stay seated (5) / unknown
        }

        auto const type = static_cast<transfer_type>(*t.transfer_type_);
        auto const r = rule{t, type, stops, routes, trips};
        if (!r.ok_) {
          return;
        }

        if (!r.forbidden_ && r.from_stop_ == r.to_stop_) {
          auto const d = duration_t{t.min_transfer_time_->value_or(0) / 60};
          reflexive_min[r.from_stop_] = std::min(reflexive_min[r.from_stop_], d);
        }

        // a recommended transfer without a time is only a preference, it
        // does not constrain anything
        auto const enforceable = type != transfer_type::kRecommended ||
                                 t.min_transfer_time_->has_value();

        if (type == transfer_type::kRecommended ||
            type == transfer_type::kTimed) {
          auto const trip_idx = [&](gtfs_trip_idx_t const trp_idx) {
            return trp_idx == gtfs_trip_idx_t::invalid()
                       ? trip_idx_t::invalid()
                       : trips.data_[trp_idx].trip_idx_;
          };
          tt.locations_.preferred_transfers_[r.from_stop_].emplace_back(
              preferred_transfer{.to_ = r.to_stop_,
                                 .from_trip_ = trip_idx(r.from_trip_),
                                 .to_trip_ = trip_idx(r.to_trip_),
                                 .from_route_ = r.from_route_,
                                 .to_route_ = r.to_route_});
        }

        if (!r.forbidden_ && !r.is_qualified()) {
          if (r.from_stop_ != r.to_stop_) {
            tt.locations_.preprocessing_footpaths_out_[r.from_stop_]
                .emplace_back(r.to_stop_, r.time_);
          } else if (enforceable) {
            tt.locations_.transfer_time_[r.from_stop_] = r.time_;
          }
        }

        if (enforceable) {
          rules.emplace_back(r);
        }
      });

  for (auto l = location_idx_t{0U}; l != location_idx_t{reflexive_min.size()};
       ++l) {
    if (reflexive_min[l] != kNoRule) {
      tt.locations_.base_transfer_time_[l] = u8_minutes{
          static_cast<std::uint8_t>(std::clamp<int>(reflexive_min[l].count(), 0, 255))};
    }
  }

  fold_pair_defaults(tt, rules);
  if (!rules.empty()) {
    apply_rules(tt, rules, trips);
  }
}

}  // namespace nigiri::loader::gtfs
