#include "nigiri/loader/gtfs/transfer_rules.h"

#include <cstdlib>
#include <algorithm>
#include <ranges>
#include <vector>

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
#include "nigiri/logging.h"
#include "nigiri/stop.h"
#include "nigiri/timetable.h"

namespace nigiri::loader::gtfs {

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

// ranking of specificity acc. to the GTFS reference (least specific first)
enum class specificity : std::uint8_t {
  kStopsOnly,
  kOneRoute,
  kBothRoutes,
  kOneTrip,
  kTripAndRoute,
  kBothTrips
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

using rule_idx_t = cista::strong<std::uint32_t, struct rule_idx_>;
using rule_vec_t = vector_map<rule_idx_t, rule>;

// One side (from/to) of one rule, packed as rule << 1 | side. A rule has no
// other sides, so in a sorted, deduplicated signature its from side sits
// directly before its to side - which is what lets the scan below find the
// rules that match with both of their sides by looking at adjacent entries.
using sided_rule_idx_t = cista::strong<std::uint32_t, struct sided_rule_idx_>;

sided_rule_idx_t side_ref(rule_idx_t const rule_idx, bool const is_from) {
  return sided_rule_idx_t{(to_idx(rule_idx) << 1U) | (is_from ? 0U : 1U)};
}

rule_idx_t rule_of(sided_rule_idx_t const s) {
  return rule_idx_t{to_idx(s) >> 1U};
}

// all rule sides that apply to a trip stop
using signature_t = std::vector<sided_rule_idx_t>;

// An ordered pair of locations: a transfer leads from_ -> to_.
struct transfer_pair {
  CISTA_COMPARABLE()
  location_idx_t from_{location_idx_t::invalid()};
  location_idx_t to_{location_idx_t::invalid()};
};

// A rule stop covers an event location if they are the same stop or if the
// event location is a child of the rule stop (station level cascade).
bool covers(timetable const& tt,
            location_idx_t const rule_stop,
            location_idx_t const l) {
  return rule_stop == l || tt.locations_.parents_[l] == rule_stop;
}

struct candidate {
  auto operator<=>(candidate const&) const = default;

  // keep member order for lexicographical comparison
  specificity specificity_{specificity::kStopsOnly};
  std::uint8_t n_exact_stops_{0U};  // from+to match the stop, not the station
  std::uint32_t rule_idx_{0U};
};

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
  auto virt_locs =
      hash_map<std::pair<location_idx_t, signature_t>, location_idx_t>{};
  auto side_virts = hash_map<sided_rule_idx_t, std::vector<location_idx_t>>{};
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
          auto reflexive_self_rules =
              utl::pairwise(sig)  //
              | std::views::filter([](auto const& sides) {
                  auto const& [from, to] = sides;
                  return rule_of(from) == rule_of(to);
                })  //
              | std::views::transform([&](auto const& sides) {
                  auto const rule_idx = rule_of(std::get<0>(sides));
                  auto const& r = rules[rule_idx];
                  return candidate{
                      .specificity_ = r.get_specificity(),
                      .n_exact_stops_ = static_cast<std::uint8_t>(
                          (r.from_stop_ == base) + (r.to_stop_ == base)),
                      .rule_idx_ = to_idx(rule_idx)};
                });
          if (auto const it = std::ranges::max_element(reflexive_self_rules);
              it != std::ranges::end(reflexive_self_rules)) {
            l.transfer_time_ = rules[rule_idx_t{(*it).rule_idx_}].duration();
          }

          auto const v = register_location(tt, l);
          tt.locations_.children_[base].emplace_back(v);

          for (auto const side : sig) {
            side_virts[side].push_back(v);
          }

          return v;
        });

    t.stop_seq_[pos] =
        stop{virt, s.in_allowed(), s.out_allowed(), s.in_allowed_wheelchair(),
             s.out_allowed_wheelchair()}
            .value();
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
          .specificity_ = r.get_specificity(),
          .n_exact_stops_ = static_cast<std::uint8_t>(
              (r.from_stop_ == base_of(x)) + (r.to_stop_ == base_of(y))),
          .rule_idx_ = to_idx(rule_idx)};
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

  // A transfer x -> y can be left out only if ALL of these hold:
  //   - x and y have the same base, because the derivation works per base
  //   - the transfer itself takes exactly the base's transfer time, because
  //     that is the only value the derivation produces
  //   - x's own transfer time, the one for changing at x, is not longer than
  //     the base's, because the derivation also yields x -> x at the base's
  //     time and would undercut it
  //   - no slower transfer starts at x, OR none leads to y (either one is
  //     enough), because the derivation reaches either every member from x,
  //     or only the members that no slower transfer leads to
  auto slow_from = hash_set<location_idx_t>{};
  auto slow_to = hash_set<location_idx_t>{};
  for (auto const& [xy, c] : most_specific) {
    if (base_of(xy.from_) == base_of(xy.to_) &&
        rules[rule_idx_t{c.rule_idx_}].duration() >
            tt.locations_.transfer_time_[base_of(xy.from_)]) {
      slow_from.insert(xy.from_);
      slow_to.insert(xy.to_);
    }
  }

  // NIGIRI_MATERIALIZE_TRANSFERS=1 writes every cell and builds no hubs, so
  // the result can be compared against the hub-derived one (see build_hubs).
  auto const materialize =
      std::getenv("NIGIRI_MATERIALIZE_TRANSFERS") != nullptr;
  auto const derivable = [&](location_idx_t const x, location_idx_t const y,
                             location_idx_t const base) {
    if (materialize) {
      return false;
    }
    auto const is_slow = x != base && tt.locations_.transfer_time_[x] >
                                          tt.locations_.transfer_time_[base];
    return !is_slow && (!slow_from.contains(x) || !slow_to.contains(y));
  };

  // A rule can also state one value for a whole cross product of locations -
  // typically a stop pair whose sides both carry virtual locations. That costs
  // |X| * |Y| transfers, where one hub covers it in |X| + |Y| edges. It is
  // only safe when every pair of the cross product really belongs to this
  // rule: otherwise the hub would deliver its value for a pair that a more
  // specific rule gave a slower one.
  struct cross_rule {
    std::size_t n_cells_{0U};
    hash_set<location_idx_t> x_, y_;
  };
  auto cross = hash_map<std::uint32_t, cross_rule>{};
  for (auto const& [xy, c] : most_specific) {
    if (base_of(xy.from_) == base_of(xy.to_)) {
      continue;  // covered by the base's own hubs
    }
    auto& g = cross[c.rule_idx_];
    ++g.n_cells_;
    g.x_.insert(xy.from_);
    g.y_.insert(xy.to_);
  }

  auto hub_rules = hash_set<std::uint32_t>{};
  auto hub_in = std::vector<location_idx_t>{};
  auto hub_out = std::vector<location_idx_t>{};
  for (auto const& [rule_idx, g] : cross) {
    auto const cells = g.x_.size() * g.y_.size();
    if (materialize || g.n_cells_ != cells ||
        cells <= g.x_.size() + g.y_.size()) {
      continue;
    }
    hub_rules.insert(rule_idx);

    hub_in.assign(begin(g.x_), end(g.x_));
    hub_out.assign(begin(g.y_), end(g.y_));
    std::sort(begin(hub_in), end(hub_in));
    std::sort(begin(hub_out), end(hub_out));
    tt.locations_.hub_in_.emplace_back(hub_in);
    tt.locations_.hub_out_.emplace_back(hub_out);
    tt.locations_.hub_time_.push_back(rules[rule_idx_t{rule_idx}].duration());
  }

  // Write the most specific transfer per pair.
  for (auto const& [xy, c] : most_specific) {
    auto const d = rules[rule_idx_t{c.rule_idx_}].duration();
    auto const base = base_of(xy.from_);
    if (base == base_of(xy.to_)) {
      if (d == tt.locations_.transfer_time_[base] &&
          derivable(xy.from_, xy.to_, base)) {
        continue;
      }
    } else if (hub_rules.contains(c.rule_idx_)) {
      continue;  // derived by this rule's hub
    }
    tt.locations_.transfer_rule_fps_[xy.from_].emplace_back(xy.to_, d);
  }

  // Apply the default between all pairs without a rule.
  for (auto virt = first_virt; virt != tt.n_locations(); ++virt) {
    auto const base = tt.locations_.parents_[virt];
    auto const d = duration_t{tt.locations_.transfer_time_[base]};
    auto const add_default_rule = [&](location_idx_t const x,
                                      location_idx_t const y) {
      if (!most_specific.contains({x, y}) && !derivable(x, y, base)) {
        tt.locations_.transfer_rule_fps_[x].emplace_back(y, d);
      }
    };

    add_default_rule(virt, base);
    add_default_rule(base, virt);
    for (auto const sibling : tt.locations_.children_[base]) {
      if (sibling != virt &&
          tt.locations_.types_[sibling] == location_type::kVirt) {
        add_default_rule(virt, sibling);  // sibling - virt added from sibling
      }
    }
  }
}

// Detects the most common rule between two stops
// -> removes them and makes their min_transfer_time the new default
void fold_pair_defaults(timetable& tt, rule_vec_t& rules) {
  struct counted_duration {
    duration_t d_;
    unsigned n_{0U};
  };

  auto qualified = hash_map<transfer_pair, std::vector<counted_duration>>{};
  auto pair_default = hash_map<transfer_pair, duration_t>{};
  for (auto const& r : rules) {
    auto const p = transfer_pair{r.from_stop_, r.to_stop_};
    if (!r.is_qualified()) {
      pair_default[p] = r.duration();  // duplicate rows: last one wins
    } else {
      auto& durations = qualified[p];
      auto const it = utl::find_if(durations, [&](counted_duration const& c) {
        return c.d_ == r.duration();
      });
      if (it == end(durations)) {
        durations.push_back({r.duration(), 1U});
      } else {
        ++it->n_;
      }
    }
  }

  auto synthetic = std::vector<rule>{};
  for (auto const& [p, durations] : qualified) {
    if (pair_default.contains(p)) {
      continue;  // explicit unqualified row
    }

    auto const majority = std::max_element(
        begin(durations), end(durations),
        [](auto const& a, auto const& b) { return a.n_ < b.n_; });
    if (majority->d_ == footpath::kMaxDuration) {
      continue;
    }
    pair_default.emplace(p, majority->d_);

    if (p.from_ == p.to_) {
      tt.locations_.transfer_time_[p.from_] = majority->d_;
    } else {
      // Add an unqualified rule
      // -> applies to all trips
      // -> won't be overwritten by street routing
      synthetic.emplace_back(p.from_, p.to_, majority->d_);
    }
  }

  // Remove all rules that re-state the default derived from the majority.
  utl::erase_if(rules, [&](rule const& r) {
    if (!r.is_qualified()) {
      return false;
    }
    auto const it = pair_default.find(transfer_pair{r.from_stop_, r.to_stop_});
    return it != end(pair_default) && r.duration() == it->second;
  });

  // Add the new default rules derived from the majority.
  for (auto const& r : synthetic) {
    rules.emplace_back(r);
  }
}

void read_transfers(timetable& tt,
                    std::string_view file_content,
                    stops_map_t const& stops,
                    route_map_t const& routes,
                    trip_data& trips) {
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

  fold_pair_defaults(tt, rules);
  if (!rules.empty()) {
    apply_rules(tt, rules, trips);
  }
}

}  // namespace nigiri::loader::gtfs
