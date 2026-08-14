#include "nigiri/loader/gtfs/transfer_rules.h"

#include <algorithm>
#include <span>
#include <vector>

#include "utl/erase_duplicates.h"
#include "utl/helpers/algorithm.h"
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

// one side (from/to) of one rule
using sided_rule_idx_t = cista::strong<std::uint32_t, struct sided_rule_idx_>;

// all rule sides that apply to a trip stop
using signature_t = std::vector<sided_rule_idx_t>;

sided_rule_idx_t side_ref(rule_idx_t const rule_idx, bool const is_from) {
  return sided_rule_idx_t{(to_idx(rule_idx) << 1U) | (is_from ? 0U : 1U)};
}

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

  // (route -> trips) for the routes referenced by a rule
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

  // (trip, stop position) -> matched rule sides
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

  // Generate one virtual location per (stop, signature) and move the matched
  // trip stops there.
  auto const first_virt = location_idx_t{tt.n_locations()};
  auto virt_locs =
      hash_map<std::pair<location_idx_t, signature_t>, location_idx_t>{};
  auto side_groups = hash_map<sided_rule_idx_t, std::vector<location_idx_t>>{};
  for (auto& [trip_stop, sig] : trip_stop_signatures) {
    auto& t = trips.data_[trip_stop.first];
    auto const s = stop{t.stop_seq_[trip_stop.second]};
    auto const base = s.location_idx();
    utl::erase_duplicates(sig);

    auto const [it, is_new] =
        virt_locs.emplace(std::pair{base, sig}, first_virt);
    if (is_new) {
      auto l = location{};
      l.src_ = tt.locations_.src_[base];
      l.pos_ = tt.locations_.coordinates_[base];
      l.type_ = location_type::kVirt;
      l.parent_ = base;
      l.transfer_time_ = tt.locations_.transfer_time_[base];

      // A rule matching this group with BOTH sides (e.g. the same-route rule
      // R5 -> R5: every R5 trip matches from and to) applies to all transfers
      // within the group. The pair emission below skips same-location pairs,
      // so the value has to become the group's transfer time. The from/to
      // side refs of one rule are adjacent in the sorted signature.
      auto best_self = std::optional<candidate>{};
      for (auto j = 0U; j + 1U < sig.size(); ++j) {
        auto const side = to_idx(sig[j]);
        if ((side & 1U) == 0U && to_idx(sig[j + 1U]) == side + 1U) {
          auto const rule_idx = rule_idx_t{side >> 1U};
          auto const& r = rules[rule_idx];
          auto const c =
              candidate{.specificity_ = r.get_specificity(),
                        .n_exact_stops_ = static_cast<std::uint8_t>(
                            (r.from_stop_ == base) + (r.to_stop_ == base)),
                        .rule_idx_ = to_idx(rule_idx)};
          best_self = std::max(best_self.value_or(c), c);
        }
      }
      if (best_self.has_value()) {
        l.transfer_time_ = rules[rule_idx_t{best_self->rule_idx_}].duration();
      }

      it->second = register_location(tt, l);
      tt.locations_.children_[base].emplace_back(it->second);
      for (auto const side : sig) {
        side_groups[side].push_back(it->second);
      }
    }

    t.stop_seq_[trip_stop.second] =
        stop{it->second, s.in_allowed(), s.out_allowed(),
             s.in_allowed_wheelchair(), s.out_allowed_wheelchair()}
            .value();
  }

  // Let the rules compete for specificity on all location pairs they apply to.
  auto subtree = std::vector<location_idx_t>{};
  auto const get_subtree = [&](location_idx_t const s) {
    subtree.assign({s});
    for (auto const child : tt.locations_.children_[s]) {
      subtree.emplace_back(child);
      for (auto const grand_child : tt.locations_.children_[child]) {
        subtree.emplace_back(grand_child);
      }
    }
    return std::span{subtree};
  };

  auto most_specific =
      hash_map<pair<location_idx_t, location_idx_t>, candidate>{};
  auto from_nodes = std::vector<location_idx_t>{};
  for (auto rule_idx = rule_idx_t{0U}; rule_idx != rules.size(); ++rule_idx) {
    auto const& r = rules[rule_idx];
    auto const get_side_nodes =
        [&](location_idx_t const rule_stop, route_id_idx_t const route,
            gtfs_trip_idx_t const trip,
            bool const is_from_side) -> std::span<location_idx_t const> {
      if (route == route_id_idx_t::invalid() &&
          trip == gtfs_trip_idx_t::invalid()) {
        return get_subtree(rule_stop);
      }
      auto const it = side_groups.find(side_ref(rule_idx, is_from_side));
      return it == end(side_groups) ? std::span<location_idx_t const>{}
                                    : std::span{it->second};
    };

    // get_subtree reuses one buffer -> the from side needs its own copy
    auto const from_span =
        get_side_nodes(r.from_stop_, r.from_route_, r.from_trip_, true);
    from_nodes.assign(begin(from_span), end(from_span));
    auto const to_nodes =
        get_side_nodes(r.to_stop_, r.to_route_, r.to_trip_, false);

    for (auto const x : from_nodes) {
      for (auto const y : to_nodes) {
        if (x == y) {
          continue;
        }
        auto const c = candidate{
            .specificity_ = r.get_specificity(),
            .n_exact_stops_ = static_cast<std::uint8_t>(
                (r.from_stop_ == base_of(x)) + (r.to_stop_ == base_of(y))),
            .rule_idx_ = to_idx(rule_idx)};
        auto const [it, is_new] = most_specific.emplace(pair{x, y}, c);
        if (!is_new) {
          it->second = std::max(it->second, c);
        }
      }
    }
  }

  // Default-valued cells between the members of a transfer group (a stop plus
  // its virtual locations) are derived at query time by the group's hubs
  // (loader::build_hubs re-derives the same classification from the surviving
  // cells) and therefore not emitted: the broadcast hub carries (x -> *) for
  // quiet, row-clean x, the collect hub (x -> y) for quiet, col-clean x and
  // col-clean y. A member is unclean if a materialized member cell above the
  // group default sits in its row/column, loud (never gathers) if its own
  // transfer time exceeds the default. Deviating cells always stay
  // materialized and win the min against the hub-derived default.
  auto row_unclean = hash_set<location_idx_t>{};
  auto col_unclean = hash_set<location_idx_t>{};
  for (auto const& [xy, c] : most_specific) {
    if (base_of(xy.first) == base_of(xy.second) &&
        rules[rule_idx_t{c.rule_idx_}].duration() >
            tt.locations_.transfer_time_[base_of(xy.first)]) {
      row_unclean.insert(xy.first);
      col_unclean.insert(xy.second);
    }
  }
  auto const derivable = [&](location_idx_t const x, location_idx_t const y,
                             location_idx_t const group) {
    auto const quiet = x == group || tt.locations_.transfer_time_[x] <=
                                         tt.locations_.transfer_time_[group];
    return quiet && (!row_unclean.contains(x) ||
                     (!col_unclean.contains(x) && !col_unclean.contains(y)));
  };

  // Write the most specific transfer per pair.
  for (auto const& [xy, c] : most_specific) {
    auto const d = rules[rule_idx_t{c.rule_idx_}].duration();
    auto const group = base_of(xy.first);
    if (group == base_of(xy.second) &&
        d == tt.locations_.transfer_time_[group] &&
        derivable(xy.first, xy.second, group)) {
      continue;
    }
    tt.locations_.transfer_rule_fps_[xy.first].emplace_back(xy.second, d);
  }

  // Apply the group default between all members without a rule.
  for (auto virt = first_virt; virt != tt.n_locations(); ++virt) {
    auto const base = tt.locations_.parents_[virt];
    auto const d = duration_t{tt.locations_.transfer_time_[base]};
    auto const add = [&](location_idx_t const x, location_idx_t const y) {
      if (!most_specific.contains({x, y}) && !derivable(x, y, base)) {
        tt.locations_.transfer_rule_fps_[x].emplace_back(y, d);
      }
    };

    add(virt, base);
    add(base, virt);
    for (auto const sibling : tt.locations_.children_[base]) {
      if (sibling != virt &&
          tt.locations_.types_[sibling] == location_type::kVirt) {
        add(virt, sibling);  // the other direction is added for the sibling
      }
    }
  }
}

// A qualified rule whose value equals the default of its stop pair does not
// require any trip splitting: the plain stop pair value realizes it for
// everyone. The default is the value given by an unqualified row for the pair
// or, if there is none, the most frequent value among the pair's qualified
// rules (ties: first listed), which is then materialized as the pair's
// transfer time / footpath. Only rules deviating from their pair's default are
// kept, so virtual locations are only created for the exceptions. (Feeds like
// us-ny/MetroNorth state one uniform timed transfer per trip pair: 13k rows
// fold to zero splits.)
rule_vec_t fold_pair_defaults(timetable& tt, rule_vec_t const& rules) {
  using stop_pair = pair<location_idx_t, location_idx_t>;

  struct counted_duration {
    duration_t d_;
    unsigned n_{0U};
  };

  auto qualified = hash_map<stop_pair, std::vector<counted_duration>>{};
  auto exemplar = hash_map<stop_pair, rule_idx_t>{};
  auto pair_default = hash_map<stop_pair, duration_t>{};
  for (auto i = rule_idx_t{0U}; i != rules.size(); ++i) {
    auto const& r = rules[i];
    auto const p = stop_pair{r.from_stop_, r.to_stop_};
    if (!r.is_qualified()) {
      pair_default[p] = r.duration();  // duplicate rows: last one wins
    } else {
      exemplar.emplace(p, i);
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
      continue;  // banning everyone would restrict unnamed trips, too
    }
    pair_default.emplace(p, majority->d_);

    if (p.first == p.second) {
      tt.locations_.transfer_time_[p.first] = majority->d_;
    } else {
      tt.locations_.preprocessing_footpaths_out_[p.first].emplace_back(
          p.second, majority->d_);
      // authoritative unqualified rule -> the edge survives street routing
      auto r = rules[exemplar.at(p)];
      r.from_route_ = r.to_route_ = route_id_idx_t::invalid();
      r.from_trip_ = r.to_trip_ = gtfs_trip_idx_t::invalid();
      r.forbidden_ = false;
      r.time_ = majority->d_;
      synthetic.push_back(r);
    }
  }

  auto kept = rule_vec_t{};
  for (auto const& r : rules) {
    auto const it = pair_default.find(stop_pair{r.from_stop_, r.to_stop_});
    if (!r.is_qualified() || it == end(pair_default) ||
        r.duration() != it->second) {
      kept.emplace_back(r);
    }
  }
  for (auto const& r : synthetic) {
    kept.emplace_back(r);
  }
  return kept;
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

  if (!rules.empty()) {
    rules = fold_pair_defaults(tt, rules);
  }
  if (!rules.empty()) {
    apply_rules(tt, rules, trips);
  }
}

}  // namespace nigiri::loader::gtfs
