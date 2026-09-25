#include "nigiri/loader/gtfs/transfer_rules.h"

#include <algorithm>
#include <optional>
#include <vector>

#include "utl/erase_duplicates.h"
#include "utl/get_or_create.h"
#include "utl/helpers/algorithm.h"
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
  return transfer_rules::rank(static_cast<std::uint8_t>(s), n_exact_stops);
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
        timed_{type == transfer_type::kTimed} {
    if (t.min_transfer_time_->has_value()) {
      auto const seconds = **t.min_transfer_time_;
      if (seconds < 0) {
        // invalid input: at most "no time", never "no transfer"
        log(log_lvl::error, "loader.gtfs.transfers",
            "{} -> {}: negative min_transfer_time {}, row ignored",
            t.from_stop_id_->view(), t.to_stop_id_->view(), seconds);
        ok_ = false;
      } else {
        // seconds in the feed, whole minutes in the timetable: always up;
        // kMaxDuration is a ban, so a longer time is capped just below it
        auto const minutes = duration_t{(seconds + 59) / 60};
        auto const longest = footpath::kMaxDuration - duration_t{1};
        if (minutes > longest) {
          log(log_lvl::info, "loader.gtfs.transfers",
              "{} -> {}: min_transfer_time {} s capped to {} min",
              t.from_stop_id_->view(), t.to_stop_id_->view(), seconds,
              longest.count());
        }
        min_transfer_time_ = std::min(minutes, longest);
      }
    }
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
        unknown_qualifier_ = id.view();
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
    if (from_trip_ != gtfs_trip_idx_t::invalid()) {
      from_trip_route_ = trips.data_[from_trip_].route_;
    }
    if (to_trip_ != gtfs_trip_idx_t::invalid()) {
      to_trip_route_ = trips.data_[to_trip_].route_;
    }
  }

  // synthesized unqualified rule, see fold_pair_defaults
  rule(location_idx_t const from, location_idx_t const to, duration_t const d)
      : from_stop_{from}, to_stop_{to}, min_transfer_time_{d} {}

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

  // Whether some trip pair exists that both rules name - by their route and
  // trip qualifiers, the stops are the caller's business. An unqualified side
  // names every trip.
  bool overlaps(rule const& o) const {
    return overlaps(from_trip_, from_trip_route_, from_route_, o.from_trip_,
                    o.from_trip_route_, o.from_route_) &&
           overlaps(to_trip_, to_trip_route_, to_route_, o.to_trip_,
                    o.to_trip_route_, o.to_route_);
  }

  // ... on one side: (trip, the trip's route, route) of each rule
  static bool overlaps(gtfs_trip_idx_t const a_trip,
                       route_id_idx_t const a_trip_route,
                       route_id_idx_t const a_route,
                       gtfs_trip_idx_t const b_trip,
                       route_id_idx_t const b_trip_route,
                       route_id_idx_t const b_route) {
    auto const a_names_trip = a_trip != gtfs_trip_idx_t::invalid();
    auto const b_names_trip = b_trip != gtfs_trip_idx_t::invalid();
    auto const a_names_route = a_route != route_id_idx_t::invalid();
    auto const b_names_route = b_route != route_id_idx_t::invalid();
    if (a_names_trip && b_names_trip) {
      return a_trip == b_trip;
    } else if (a_names_trip && b_names_route) {
      return a_trip_route == b_route;
    } else if (a_names_route && b_names_trip) {
      return b_trip_route == a_route;
    } else if (a_names_route && b_names_route) {
      return a_route == b_route;
    }
    return true;
  }

  // a timed transfer and a ban say nothing about how long a change takes -
  // only a stated time of another type can become a default (see
  // fold_pair_defaults)
  bool states_time() const {
    return !forbidden_ && !timed_ && min_transfer_time_.has_value();
  }

  // A timed transfer (type 1) needs no minimum, whatever time it states: the
  // departing vehicle waits, so the pair costs nothing beyond being there.
  duration_t duration() const {
    return forbidden_ ? footpath::kMaxDuration
           : timed_   ? duration_t{0}
                      : min_transfer_time_.value_or(duration_t{0});
  }

  location_idx_t from_stop_{location_idx_t::invalid()};
  location_idx_t to_stop_{location_idx_t::invalid()};
  route_id_idx_t from_route_{route_id_idx_t::invalid()};
  route_id_idx_t to_route_{route_id_idx_t::invalid()};
  gtfs_trip_idx_t from_trip_{gtfs_trip_idx_t::invalid()};
  gtfs_trip_idx_t to_trip_{gtfs_trip_idx_t::invalid()};
  route_id_idx_t from_trip_route_{route_id_idx_t::invalid()};
  route_id_idx_t to_trip_route_{route_id_idx_t::invalid()};
  bool forbidden_{false};
  bool timed_{false};
  std::optional<duration_t> min_transfer_time_;
  bool ok_{true};
  std::string_view unknown_qualifier_;  // set when ok_ is false for a qualifier
};

using rule_vec_t = vector_map<rule_idx_t, rule>;

// What one matched rule side states, seen from the trip stop that matched it.
// The same key as rt_transfer_rules.cc (side_value) builds: resolve() finds
// the static virtual location of a real-time stop by comparing the two.
struct rule_side {
  CISTA_COMPARABLE()
  bool is_from_;
  location_idx_t rule_stop_;  // the stop the rule names on this side
  location_idx_t other_stop_;
  route_id_idx_t other_route_;
  gtfs_trip_idx_t other_trip_;
  duration_t duration_;
  bool by_trip_;  // this side names a trip, not a route (see virt_for)
};

// Everything the routing can observe about a trip stop's rules: two trip
// stops of one base with the same key are one virtual location.
struct virt_key {
  CISTA_COMPARABLE()
  location_idx_t base_;
  duration_t own_;
  std::vector<rule_side> sides_;
};

void apply_rules(timetable& tt,
                 rule_vec_t const& rules,
                 trip_data& trips,
                 bool const rule_hubs) {
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
  auto const first_virt = location_idx_t{tt.n_locations()};
  // A trip stop is keyed by what its rules state, not by which rules they
  // are: per matched side the stop the rule names on this side, the stop and
  // the qualifier it names on the other side, and the duration - plus the own
  // transfer time the self rules resolve to, and where sides of different
  // durations compete, whether a side names a trip or a route (its rank).
  // Two trip stops with the same key get, for every partner, the same winning
  // value, so they are one node from the start and never need to be told
  // apart again.
  auto virt_locs = hash_map<virt_key, location_idx_t>{};
  auto side_virts = hash_map<sided_rule_idx_t, std::vector<location_idx_t>>{};
  auto const virt_for = [&](location_idx_t const base, signature_t& sig) {
    utl::erase_duplicates(sig);

    // A rule whose from and to side both apply here (e.g. transfers from
    // route R -> R, or from R -> anything at this stop) sets the change time
    // between two trip stops of this virt. A qualified side applies if it is
    // in the (sorted) signature, an unqualified one if its stop covers base.
    auto const applies = [&](rule_idx_t const rule_idx, bool const is_from) {
      auto const& r = rules[rule_idx];
      auto const qualified =
          is_from ? (r.from_route_ != route_id_idx_t::invalid() ||
                     r.from_trip_ != gtfs_trip_idx_t::invalid())
                  : (r.to_route_ != route_id_idx_t::invalid() ||
                     r.to_trip_ != gtfs_trip_idx_t::invalid());
      return qualified ? std::binary_search(begin(sig), end(sig),
                                            side_ref(rule_idx, is_from))
                       : covers(tt, is_from ? r.from_stop_ : r.to_stop_, base);
    };
    auto own = from_transfer_time(tt.locations_.transfer_time_[base]);
    auto best = std::optional<candidate>{};
    for (auto const side : sig) {
      auto const rule_idx = rule_of(side);
      if (!applies(rule_idx, true) || !applies(rule_idx, false)) {
        continue;
      }
      auto const& r = rules[rule_idx];
      auto const c = candidate{
          .rank_ = rank(r.get_specificity(),
                        static_cast<std::uint8_t>((r.from_stop_ == base) +
                                                  (r.to_stop_ == base))),
          .rule_idx_ = rule_idx};
      if (!best.has_value() || *best < c) {
        best = c;
      }
    }
    if (best.has_value()) {
      own = rules[best->rule_idx_].duration();
    }

    auto key = virt_key{.base_ = base, .own_ = own, .sides_ = {}};
    for (auto const side : sig) {
      auto const& r = rules[rule_of(side)];
      auto const is_from = side == side_ref(rule_of(side), true);
      key.sides_.push_back(
          {.is_from_ = is_from,
           .rule_stop_ = is_from ? r.from_stop_ : r.to_stop_,
           .other_stop_ = is_from ? r.to_stop_ : r.from_stop_,
           .other_route_ = is_from ? r.to_route_ : r.from_route_,
           .other_trip_ = is_from ? r.to_trip_ : r.from_trip_,
           .duration_ = r.duration(),
           .by_trip_ = (is_from ? r.from_trip_ : r.to_trip_) !=
                       gtfs_trip_idx_t::invalid()});
    }
    keep_rank_where_it_decides(tt, key.sides_);
    utl::erase_duplicates(key.sides_);

    auto const virt = utl::get_or_create(virt_locs, std::move(key), [&]() {
      auto l = location{};
      l.src_ = tt.locations_.src_[base];
      l.pos_ = tt.locations_.coordinates_[base];
      l.type_ = location_type::kVirt;
      l.parent_ = base;
      l.transfer_time_ = own;

      auto const v = register_location(tt, l);
      tt.locations_.children_[base].emplace_back(v);
      return v;
    });

    // several trip stops with different rule sides can share one virtual
    // location, so its sides are the union of theirs (deduplicated below)
    for (auto const side : sig) {
      side_virts[side].push_back(virt);
    }
    return virt;
  };

  // Where one vehicle continues from trip a as trip b (block_id or a
  // stay-seated transfer), the joined transport stops once: a's last stop is
  // b's first. That stop is where passengers alight from a and where they
  // board b, so it carries the rule sides of both trip stops. Collected
  // before the trip stops are rewritten: the two have to be the same stop as
  // the feed states it, which is what the block join compares.
  struct junction {
    gtfs_trip_idx_t from_, to_;
    location_idx_t base_;
    signature_t sig_;
  };
  auto junctions = std::vector<junction>{};
  auto const add_junction = [&](gtfs_trip_idx_t const a,
                                gtfs_trip_idx_t const b) {
    auto const& ta = trips.data_[a];
    auto const& tb = trips.data_[b];
    if (a == b || ta.stop_seq_.empty() || tb.stop_seq_.empty()) {
      return;
    }
    auto const last = static_cast<stop_idx_t>(ta.stop_seq_.size() - 1U);
    auto const base = stop{ta.stop_seq_[last]}.location_idx();
    if (base != stop{tb.stop_seq_.front()}.location_idx()) {
      return;
    }
    auto sig = signature_t{};
    for (auto const key : {pair{a, last}, pair{b, stop_idx_t{0U}}}) {
      if (auto const it = trip_stop_signatures.find(key);
          it != end(trip_stop_signatures)) {
        sig.insert(end(sig), begin(it->second), end(it->second));
      }
    }
    if (!sig.empty()) {
      junctions.push_back({a, b, base, std::move(sig)});
    }
  };
  for (auto const& [_, blk] : trips.blocks_) {
    // every pair block::rule_services can join: b departs no earlier, starts
    // where a ends and runs on one of a's days (a superset of the joins made)
    for (auto const a : blk->trips_) {
      for (auto const b : blk->trips_) {
        auto const& ta = trips.data_[a];
        auto const& tb = trips.data_[b];
        if (!ta.event_times_.empty() && !tb.event_times_.empty() &&
            ta.service_ != nullptr && tb.service_ != nullptr &&
            ta.event_times_.front().dep_ <= tb.event_times_.front().dep_ &&
            (*ta.service_ & *tb.service_).any()) {
          add_junction(a, b);
        }
      }
    }
  }
  for (auto a = gtfs_trip_idx_t{0U}; a != trips.data_.size(); ++a) {
    for (auto const b : trips.data_[a].seated_out_) {
      add_junction(a, b);
    }
  }

  for (auto& [trip_stop, sig] : trip_stop_signatures) {
    auto const [trp_idx, pos] = trip_stop;
    auto& t = trips.data_[trp_idx];
    auto const s = stop{t.stop_seq_[pos]};
    auto const virt = virt_for(s.location_idx(), sig);
    t.stop_seq_[pos] = s.with_location(virt).value();
  }
  for (auto& j : junctions) {
    trips.junctions_.emplace(pair{j.from_, j.to_}, virt_for(j.base_, j.sig_));
  }
  for (auto& [side, virts] : side_virts) {
    utl::erase_duplicates(virts);
  }

  // Keep the rules for real-time stop changes (rt/rt_transfer_rules.h): a
  // trip that moves to another platform has to find the rules that apply
  // there. Rule indices are local to this feed, so they are shifted.
  {
    auto& tr = tt.transfer_rules_;
    auto const offset = static_cast<std::uint32_t>(tr.rules_.size());
    auto const global = [&](sided_rule_idx_t const s) {
      return transfer_rules::side(offset + to_idx(rule_of(s)),
                                  s == side_ref(rule_of(s), true));
    };
    auto const trip_idx = [&](gtfs_trip_idx_t const t) {
      return t == gtfs_trip_idx_t::invalid() ? trip_idx_t::invalid()
                                             : trips.data_[t].trip_idx_;
    };
    auto const src = tt.locations_.src_[rules[rule_idx_t{0U}].from_stop_];
    for (auto rule_idx = rule_idx_t{0U}; rule_idx != rules.size(); ++rule_idx) {
      auto const& r = rules[rule_idx];
      tr.rules_.emplace_back(stop_transfer_rule{
          .from_stop_ = r.from_stop_,
          .to_stop_ = r.to_stop_,
          .from_route_ = r.from_route_,
          .to_route_ = r.to_route_,
          .from_trip_ = trip_idx(r.from_trip_),
          .to_trip_ = trip_idx(r.to_trip_),
          .src_ = src,
          .duration_ = r.duration(),
          .specificity_ = static_cast<std::uint8_t>(r.get_specificity())});
      auto const add_side = [&](location_idx_t const stop,
                                route_id_idx_t const route,
                                gtfs_trip_idx_t const trip,
                                bool const is_from) {
        auto const s = transfer_rules::side(offset + to_idx(rule_idx), is_from);
        if (trip != gtfs_trip_idx_t::invalid()) {
          tr.trip_sides_.push_back({to_idx(trip_idx(trip)), s});
        } else if (route != route_id_idx_t::invalid()) {
          tr.route_sides_.push_back({to_idx(route), s});
        } else {
          tr.stop_sides_.push_back({to_idx(stop), s});
        }
      };
      add_side(r.from_stop_, r.from_route_, r.from_trip_, true);
      add_side(r.to_stop_, r.to_route_, r.to_trip_, false);
    }
    for (auto const& [side, virts] : side_virts) {
      for (auto const v : virts) {
        tr.side_virts_.push_back({global(side), to_idx(v)});
      }
    }
    auto const n_virt_sides = tr.virt_sides_.size();
    for (auto const& [trip_stop, sig] : trip_stop_signatures) {
      auto const v =
          stop{trips.data_[trip_stop.first].stop_seq_[trip_stop.second]}
              .location_idx();
      for (auto const side : sig) {
        tr.virt_sides_.push_back({to_idx(v), global(side)});
      }
    }
    for (auto const& j : junctions) {
      auto const v = trips.junctions_.at(pair{j.from_, j.to_});
      for (auto const side : j.sig_) {
        tr.virt_sides_.push_back({to_idx(v), global(side)});
      }
    }
    // many trip stops share a virtual location and state the same sides:
    // drop the repetitions right away, they would dominate the size
    auto const feed_begin =
        begin(tr.virt_sides_) + static_cast<std::ptrdiff_t>(n_virt_sides);
    std::sort(feed_begin, end(tr.virt_sides_));
    tr.virt_sides_.erase(std::unique(feed_begin, end(tr.virt_sides_)),
                         end(tr.virt_sides_));
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
    for_each(
        rule_idx, r.from_stop_, r.from_route_, r.from_trip_, true,
        [&](location_idx_t const x) {
          for_each(
              rule_idx, r.to_stop_, r.to_route_, r.to_trip_, false,
              [&](location_idx_t const y) {
                if (x == y) {
                  return;
                }
                auto& best = most_specific[transfer_pair{x, y}];
                best = std::max(
                    best,
                    candidate{
                        .rank_ = rank(
                            r.get_specificity(),
                            static_cast<std::uint8_t>(
                                (r.from_stop_ ==
                                 tt.locations_.get_base_idx(x)) +
                                (r.to_stop_ == tt.locations_.get_base_idx(y)))),
                        .rule_idx_ = rule_idx});
              });
        });
  }

  auto durations = vector_map<rule_idx_t, duration_t>{};
  durations.reserve(rules.size());
  for (auto const& r : rules) {
    durations.push_back(r.duration());
  }

  write_transfer_rules(tt, most_specific, durations, first_virt, rule_hubs);

  log(log_lvl::info, "loader.gtfs.transfer_rules", "{} virtual locations",
      tt.n_locations() - first_virt.v_);
}

void read_transfers(timetable& tt,
                    std::string_view file_content,
                    stops_map_t const& stops,
                    route_map_t const& routes,
                    trip_data& trips,
                    bool const rule_hubs) {
  // Capture transfer times before this feed's rules modify them.
  tt.locations_.sync_base_transfer_time();

  if (file_content.empty()) {
    return;
  }

  auto const timer = scoped_timer{"loader.gtfs.transfers"};

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Read Transfers").in_high(file_content.size());

  // the trips an in-seat row (type 4 / 5) links: both are required
  auto const linked_trips = [&](csv_transfer const& t)
      -> std::optional<pair<gtfs_trip_idx_t, gtfs_trip_idx_t>> {
    if (t.from_trip_id_->empty() || t.to_trip_id_->empty()) {
      log(log_lvl::error, "loader.gtfs.transfers",
          "in-seat transfers (type 4, 5) require from_trip_id and to_trip_id");
      return std::nullopt;
    }
    auto const from = trips.trips_.find(t.from_trip_id_->view());
    auto const to = trips.trips_.find(t.to_trip_id_->view());
    if (from == end(trips.trips_) || to == end(trips.trips_)) {
      log(log_lvl::error, "loader.gtfs.transfers", "trip {} not found",
          from == end(trips.trips_) ? t.from_trip_id_->view()
                                    : t.to_trip_id_->view());
      return std::nullopt;
    }
    return pair{from->second, to->second};
  };

  auto const wire_stay_seated = [&](csv_transfer const& t) {
    auto const linked = linked_trips(t);
    if (!linked.has_value()) {
      return;
    }
    auto const [from, to] = *linked;
    // deduplicate: the CH feed introduces duplicate primary keys through an
    // additional service_id
    auto const push_unique = [](auto& vec, auto const value) {
      if (utl::find(vec, value) == end(vec)) {
        vec.push_back(value);
      }
    };
    push_unique(trips.data_[to].seated_in_, from);
    push_unique(trips.data_[from].seated_out_, to);
  };

  auto rules = rule_vec_t{};

  // The profiles other than the default one are purely routed: nothing in
  // transfers.txt applies to them, not even a stop's own change time. They
  // read base_transfer_time_, which keeps the value from before these rules
  // (sync_base_transfer_time above).

  auto n_unknown_qualifiers = 0U;
  auto first_unknown_qualifier = std::string_view{};
  auto n_same_trip = 0U;

  utl::line_range{
      utl::make_buf_reader(file_content, progress_tracker->update_fn())}  //
      | utl::csv<csv_transfer>()  //
      |
      utl::for_each([&](csv_transfer const& t) {
        if (*t.transfer_type_ == static_cast<int>(transfer_type::kStaySeated)) {
          wire_stay_seated(t);
          return;
        }
        if (*t.transfer_type_ ==
            static_cast<int>(transfer_type::kNoStaySeated)) {
          if (auto const linked = linked_trips(t); linked.has_value()) {
            trips.no_stay_seated_.emplace(*linked);
          }
          return;
        }
        if (*t.transfer_type_ < 0 ||
            *t.transfer_type_ > static_cast<int>(transfer_type::kNotPossible)) {
          return;  // unknown
        }

        auto const type = static_cast<transfer_type>(*t.transfer_type_);
        auto const r = rule{t, type, stops, routes, trips};
        if (!r.ok_) {
          if (!r.unknown_qualifier_.empty()) {
            if (n_unknown_qualifiers++ == 0U) {
              first_unknown_qualifier = r.unknown_qualifier_;
            }
          }
          return;
        }
        // a trip does not transfer to itself: such a row states nothing
        if (r.from_trip_ != gtfs_trip_idx_t::invalid() &&
            r.from_trip_ == r.to_trip_) {
          ++n_same_trip;
          return;
        }

        // A timed transfer holds the departing vehicle and a forbidden one
        // needs no time; every other row constrains only if it states a
        // time. A recommended transfer without one is a preference, and a
        // minimum change time without one is a defective row - neither says
        // how long a transfer takes, so neither becomes a rule.
        auto const enforceable = type == transfer_type::kTimed ||
                                 type == transfer_type::kNotPossible ||
                                 r.min_transfer_time_.has_value();

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

        // An unqualified same-stop row is the stop's own transfer time - a
        // ban included, it becomes kNoTransfer. A cross-stop one is a rule
        // cell like every other rule (see apply_rules), which is what keeps
        // it authoritative once street routing recomputes the walks.
        if (enforceable && !r.is_qualified() && r.from_stop_ == r.to_stop_) {
          tt.locations_.transfer_time_[r.from_stop_] =
              to_transfer_time(r.duration());
        }

        if (enforceable) {
          rules.emplace_back(r);
        }
      });

  if (n_unknown_qualifiers != 0U) {
    log(log_lvl::error, "loader.gtfs.transfers",
        "{} rules dropped: unknown route/trip id (e.g. {:?})",
        n_unknown_qualifiers, first_unknown_qualifier);
  }
  if (n_same_trip != 0U) {
    log(log_lvl::info, "loader.gtfs.transfers",
        "{} rows from a trip to itself ignored", n_same_trip);
  }

  fold_pair_defaults(tt, rules);
  if (!rules.empty()) {
    apply_rules(tt, rules, trips, rule_hubs);
  }
}

}  // namespace nigiri::loader::gtfs
