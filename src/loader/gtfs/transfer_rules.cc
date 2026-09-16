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
      : forbidden_{type == transfer_type::kNotPossible} {
    if (t.min_transfer_time_->has_value()) {
      // seconds in the feed, whole minutes in the timetable: always up
      min_transfer_time_ = duration_t{(**t.min_transfer_time_ + 59) / 60};
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

  // a timed transfer without a time and a ban say nothing about how long a
  // change takes - only a stated time can become a default (see
  // fold_pair_defaults)
  bool states_time() const {
    return !forbidden_ && min_transfer_time_.has_value();
  }

  // A timed transfer (type 1) needs no minimum: the departing vehicle waits,
  // so the pair costs nothing beyond being there.
  duration_t duration() const {
    return forbidden_ ? footpath::kMaxDuration
                      : min_transfer_time_.value_or(duration_t{0});
  }

  location_idx_t from_stop_{location_idx_t::invalid()};
  location_idx_t to_stop_{location_idx_t::invalid()};
  route_id_idx_t from_route_{route_id_idx_t::invalid()};
  route_id_idx_t to_route_{route_id_idx_t::invalid()};
  gtfs_trip_idx_t from_trip_{gtfs_trip_idx_t::invalid()};
  gtfs_trip_idx_t to_trip_{gtfs_trip_idx_t::invalid()};
  bool forbidden_{false};
  std::optional<duration_t> min_transfer_time_;
  bool ok_{true};
  std::string_view unknown_qualifier_;  // set when ok_ is false for a qualifier
};

using rule_vec_t = vector_map<rule_idx_t, rule>;

// What one matched rule side states, seen from the trip stop that matched it.
struct rule_side {
  CISTA_COMPARABLE()
  bool is_from_;
  location_idx_t rule_stop_;  // the stop the rule names on this side
  location_idx_t other_stop_;
  route_id_idx_t other_route_;
  gtfs_trip_idx_t other_trip_;
  duration_t duration_;
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
  // transfer time the self rules resolve to. Two trip stops with the same key
  // have, for every partner, rules of the same value and specificity, so they
  // are one node from the start and never need to be told apart again.
  auto virt_locs = hash_map<virt_key, location_idx_t>{};
  auto side_virts = hash_map<sided_rule_idx_t, std::vector<location_idx_t>>{};
  for (auto& [trip_stop, sig] : trip_stop_signatures) {
    auto const [trp_idx, pos] = trip_stop;
    auto& t = trips.data_[trp_idx];
    auto const s = stop{t.stop_seq_[pos]};
    auto const base = s.location_idx();
    utl::erase_duplicates(sig);

    // Rules where from == to side (e.g. transfers from route R -> R) end up
    // on the same virt node -> set transfer_time to self. The signature is
    // sorted and a rule's from side directly precedes its to side (see
    // side_ref), so such a rule shows up as two adjacent entries.
    auto own = from_transfer_time(tt.locations_.transfer_time_[base]);
    auto best = std::optional<candidate>{};
    for (auto i = std::size_t{1U}; i < sig.size(); ++i) {
      auto const from = sig[i - 1U];
      auto const to = sig[i];
      if (rule_of(from) != rule_of(to)) {
        continue;
      }
      auto const rule_idx = rule_of(from);
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
           .duration_ = r.duration()});
    }
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

    t.stop_seq_[pos] =
        stop{virt, s.in_allowed(), s.out_allowed(), s.in_allowed_wheelchair(),
             s.out_allowed_wheelchair()}
            .value();
  }
  for (auto& [side, virts] : side_virts) {
    utl::erase_duplicates(virts);
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
    for_each(rule_idx, r.from_stop_, r.from_route_, r.from_trip_, true,
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
                         candidate{.rank_ = rank(
                                       r.get_specificity(),
                                       static_cast<std::uint8_t>(
                                           (r.from_stop_ == base_of(tt, x)) +
                                           (r.to_stop_ == base_of(tt, y)))),
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
  // it - minimum over every reflexive row that states a time, qualified or
  // not, replacing the default rather than capping it.
  constexpr auto const kNoRule = duration_t::max();
  auto reflexive_min = vector_map<location_idx_t, duration_t>{};
  reflexive_min.resize(tt.n_locations());
  std::fill(begin(reflexive_min), end(reflexive_min), kNoRule);

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
        if (*t.transfer_type_ < 0 ||
            *t.transfer_type_ > static_cast<int>(transfer_type::kNotPossible)) {
          return;  // no stay seated (5) / unknown
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

        if (enforceable && !r.forbidden_ && r.from_stop_ == r.to_stop_ &&
            r.min_transfer_time_.has_value()) {
          reflexive_min[r.from_stop_] =
              std::min(reflexive_min[r.from_stop_], *r.min_transfer_time_);
        }

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
        // it authoritative once street routing recomputes the walks - and it
        // also seeds the footpath layer, because a cell is only overlaid at
        // the end while a footpath takes part in the layer's closure, its
        // hubs and the lower bounds.
        if (enforceable && !r.is_qualified()) {
          if (r.from_stop_ == r.to_stop_) {
            tt.locations_.transfer_time_[r.from_stop_] =
                to_transfer_time(r.duration());
          } else if (!r.forbidden_) {
            tt.locations_.preprocessing_footpaths_out_[r.from_stop_]
                .emplace_back(r.to_stop_, r.duration());
          }
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

  for (auto l = location_idx_t{0U}; l != location_idx_t{reflexive_min.size()};
       ++l) {
    if (reflexive_min[l] != kNoRule) {
      tt.locations_.base_transfer_time_[l] = to_transfer_time(reflexive_min[l]);
    }
  }

  fold_pair_defaults(tt, rules);
  if (!rules.empty()) {
    apply_rules(tt, rules, trips, rule_hubs);
  }
}

}  // namespace nigiri::loader::gtfs
