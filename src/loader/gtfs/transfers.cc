#include "nigiri/loader/gtfs/transfers.h"

#include <algorithm>
#include <optional>
#include <span>
#include <vector>

#include "geo/latlng.h"

#include "utl/concat.h"
#include "utl/enumerate.h"
#include "utl/erase_duplicates.h"
#include "utl/helpers/algorithm.h"
#include "utl/lookup.h"
#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"
#include "utl/progress_tracker.h"

#include "nigiri/loader/transfer_rules.h"
#include "nigiri/constants.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"

namespace nigiri::loader::gtfs {

enum class specificity : std::uint8_t {
  kStopsOnly,
  kOneRoute,
  kBothRoutes,
  kOneTrip,
  kTripAndRoute,
  kBothTrips
};

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

transfer_rule_specificity_t get_specificity(
    timetable const& tt,
    location_idx_t const from_stop,
    location_idx_t const to_stop,
    route_id_idx_t const from_route = route_id_idx_t::invalid(),
    route_id_idx_t const to_route = route_id_idx_t::invalid(),
    trip_idx_t const from_trip = trip_idx_t::invalid(),
    trip_idx_t const to_trip = trip_idx_t::invalid()) {
  auto const has_from_trip = from_trip != trip_idx_t::invalid();
  auto const has_to_trip = to_trip != trip_idx_t::invalid();
  auto const has_from_route = from_route != route_id_idx_t::invalid();
  auto const has_to_route = to_route != route_id_idx_t::invalid();
  auto const level =
      has_from_trip && has_to_trip ? specificity::kBothTrips
      : (has_from_trip && has_to_route) || (has_from_route && has_to_trip)
          ? specificity::kTripAndRoute
      : has_from_trip || has_to_trip   ? specificity::kOneTrip
      : has_from_route && has_to_route ? specificity::kBothRoutes
      : has_from_route || has_to_route ? specificity::kOneRoute
                                       : specificity::kStopsOnly;
  auto const has_parent = [&](location_idx_t const stop) {
    return tt.locations_.parents_[stop] != location_idx_t::invalid() ? 1U : 0U;
  };
  return static_cast<transfer_rule_specificity_t>(
      (static_cast<unsigned>(level) << 2U) |
      (has_parent(from_stop) + has_parent(to_stop)));
}

bool is_qualified(transfer_rule const& r) {
  return r.is_qualified(event_type::kArr) || r.is_qualified(event_type::kDep);
}

bool is_qualifier_overlapping(timetable const& tt,
                              transfer_rule const& a,
                              transfer_rule const& b) {
  auto const is_side_overlapping = [&](event_type const ev) {
    auto const a_trip = a.trip(ev);
    auto const b_trip = b.trip(ev);
    auto const a_route = a.route(ev);
    auto const b_route = b.route(ev);
    auto const has_a_trip = a_trip != trip_idx_t::invalid();
    auto const has_b_trip = b_trip != trip_idx_t::invalid();
    auto const has_a_route = a_route != route_id_idx_t::invalid();
    auto const has_b_route = b_route != route_id_idx_t::invalid();
    if (has_a_trip && has_b_trip) {
      return a_trip == b_trip;
    } else if (has_a_trip && has_b_route) {
      return tt.trip_route_id_[a_trip] == b_route;
    } else if (has_a_route && has_b_trip) {
      return tt.trip_route_id_[b_trip] == a_route;
    } else if (has_a_route && has_b_route) {
      return a_route == b_route;
    }
    return true;
  };
  return is_side_overlapping(event_type::kArr) &&
         is_side_overlapping(event_type::kDep);
}

void fold_pair_defaults(timetable& tt,
                        source_idx_t const src,
                        std::span<transfer_rule const> rules,
                        std::vector<bool> const& votes) {
  auto const& parents = tt.locations_.parents_;

  auto qualified =
      hash_map<transfer_pair, std::vector<pair<duration_t, unsigned>>>{};
  // Unqualified durations per stop pair with their order: the most specific
  // one applies, the later one on a tie (as get_more_specific decides).
  struct default_duration_t {
    duration_t duration_;
    std::size_t order_;
  };
  auto default_duration = hash_map<transfer_pair, default_duration_t>{};
  // Prepare for lookup by stop pair.
  auto by_pair = hash_map<transfer_pair, std::vector<transfer_rule const*>>{};
  for (auto const [i, r] : utl::enumerate(rules)) {
    auto const p = transfer_pair{r.from_stop_, r.to_stop_};
    if (!is_qualified(r)) {
      // Store explicit unqualified transfer durations.
      default_duration[p] = {r.duration_, i};
      continue;
    }
    by_pair[p].push_back(&rules[i]);
    if (votes[i]) {
      // Count durations of qualified transfer rules for each stop pair.
      // Forbidden and recommended (=0min) transfers are not counted.
      auto& durations = qualified[p];
      auto const it = utl::find_if(
          durations, [&](auto const& c) { return c.first == r.duration_; });
      if (it == end(durations)) {
        durations.push_back({r.duration_, 1U});
      } else {
        ++it->second;
      }
    }
  }

  // Resolves the duration of the stop pair taking into account stop hierarchy.
  auto const get_default_duration =
      [&](transfer_pair const p) -> std::optional<duration_t> {
    auto best = std::optional<duration_t>{};
    auto best_key = std::pair<transfer_rule_specificity_t, std::size_t>{};
    for (auto const from : {p.from_, parents[p.from_]}) {
      for (auto const to : {p.to_, parents[p.to_]}) {
        auto const d = utl::lookup(default_duration, transfer_pair{from, to});
        if (!d.has_value()) {
          continue;
        }
        auto const key = std::pair{get_specificity(tt, from, to), d->order_};
        if (!best.has_value() || best_key < key) {
          best = d->duration_;
          best_key = key;
        }
      }
    }
    return best;
  };

  // Add majority transfer durations to explicit unqualified rules durations.
  auto majority_rules = std::vector<transfer_rule>{};
  for (auto const& [p, durations] : qualified) {
    // For P1 -> P2 both (S1, P2) and (P1, S2) apply with the same specificity.
    // Prevent a majority rule overriding an explicit unqualified rule.
    if ((parents[p.to_] != location_idx_t::invalid() &&
         !tt.locations_.children_[p.from_].empty()) ||
        (parents[p.from_] != location_idx_t::invalid() &&
         !tt.locations_.children_[p.to_].empty())) {
      continue;
    }

    // If there's an explicit default duration -> skip.
    if (get_default_duration(p).has_value()) {
      continue;
    }

    // Convert the majority of qualified rules
    // for this stop pair into an unqualified rule.
    auto const majority = std::max_element(
        begin(durations), end(durations),
        [](auto const& a, auto const& b) { return a.second < b.second; });
    majority_rules.push_back(
        transfer_rule{.from_stop_ = p.from_,
                      .to_stop_ = p.to_,
                      .src_ = src,
                      .duration_ = majority->first,
                      .specificity_ = get_specificity(tt, p.from_, p.to_)});
  }
  for (auto const [i, r] : utl::enumerate(majority_rules)) {
    default_duration.emplace(transfer_pair{r.from_stop_, r.to_stop_},
                             default_duration_t{r.duration_, rules.size() + i});
  }

  auto const is_kept = [&](transfer_rule const& r) {
    // Keep all unqualified rules.
    if (!is_qualified(r)) {
      return true;
    }

    // Keep all rules that don't just restate the default duration.
    auto const from_to = transfer_pair{r.from_stop_, r.to_stop_};
    auto const d = get_default_duration(from_to);
    if (d != r.duration_) {
      return true;
    }

    // Check if a majority rule conflict with an existing rule?
    auto const any_with_children = [&](location_idx_t const l, auto&& fn) {
      return fn(l) || utl::any_of(tt.locations_.children_[l], fn);
    };
    auto const any_with_children_and_parent = [&](location_idx_t const l,
                                                  auto&& fn) {
      return any_with_children(l, fn) ||
             (parents[l] != location_idx_t::invalid() && fn(parents[l]));
    };
    auto const any_pair = [&](auto const& any_of, auto&& fn) {
      return any_of(r.from_stop_, [&](location_idx_t const from) {
        return any_of(r.to_stop_, [&](location_idx_t const to) {
          return fn(transfer_pair{from, to});
        });
      });
    };

    return
        // Does this rule override an explicit/majority default
        // with a different duration? => keep this rule
        any_pair(any_with_children,  // this/children stop pairs
                 [&](transfer_pair const p) {
                   return utl::lookup(default_duration, p)
                       .transform([&](default_duration_t const& x) {
                         return x.duration_ != r.duration_;
                       })
                       .value_or(false);
                 })

        // Would removing this rule let a less or equally specific overlapping
        // rule with a different duration take effect? => keep this rule
        || any_pair(
               any_with_children_and_parent,  // this/parent/children stop pairs
               [&](transfer_pair const p) {
                 auto const it = by_pair.find(p);
                 return it != end(by_pair) &&
                        utl::any_of(it->second, [&](transfer_rule const* o) {
                          return o->duration_ != r.duration_ &&
                                 o->specificity_ <= r.specificity_ &&
                                 is_qualifier_overlapping(tt, *o, r);
                        });
               });
  };

  // Add rules to the timetable.
  for (auto const& r : rules) {
    if (is_kept(r)) {
      tt.transfer_rules_.rules_.push_back(r);
    }
  }
  utl::concat(tt.transfer_rules_.rules_, majority_rules);

  // A change at a stop takes its default: its own, between it and its station,
  // or the station's.
  auto stops = std::vector<location_idx_t>{};
  for (auto const& [p, _] : default_duration) {
    if (p.from_ == p.to_) {
      stops.push_back(p.from_);
      utl::concat(stops, tt.locations_.children_[p.from_]);
    } else if (parents[p.from_] == p.to_) {
      stops.push_back(p.from_);
    } else if (parents[p.to_] == p.from_) {
      stops.push_back(p.to_);
    }
  }
  utl::erase_duplicates(stops);
  for (auto const l : stops) {
    tt.locations_.transfer_time_[l] =
        to_transfer_time(*get_default_duration({l, l}));
  }
}

std::optional<duration_t> adjust_to_walk_speed(timetable const& tt,
                                               location_idx_t const a,
                                               location_idx_t const b,
                                               duration_t const duration) {
  constexpr auto const kMaxWalkDistance =
      std::numeric_limits<u8_minutes::rep>::max() * 60.0 * kWalkSpeed;

  auto const distance = geo::distance(tt.locations_.coordinates_[a],
                                      tt.locations_.coordinates_[b]);
  if (distance > kMaxWalkDistance) {
    log(log_lvl::error, "loader.gtfs.transfers",
        "{} -> {}: {:.1f} km apart, not walkable, row ignored",
        tt.locations_.ids_[a].view(), tt.locations_.ids_[b].view(),
        distance / 1000.0);
    return std::nullopt;
  }

  return std::max(
      duration,
      duration_t{static_cast<duration_t::rep>(distance / kWalkSpeed / 60)});
}

void read_transfers(source_idx_t const src,
                    timetable& tt,
                    std::string_view file_content,
                    stops_map_t const& stops,
                    trip_data& trips,
                    bool const adjust_footpaths) {
  if (file_content.empty()) {
    return;
  }

  auto const timer = scoped_timer{"loader.gtfs.transfers"};

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Read Transfers").in_high(file_content.size());

  auto const& route_ids = tt.route_ids_[src].ids_;
  auto const find_trip = [&](utl::cstr const id) {
    return id.empty() ? std::optional{gtfs_trip_idx_t::invalid()}
                      : utl::lookup(trips.trips_, id.view());
  };
  auto const find_route = [&](utl::cstr const id) {
    return id.empty() ? std::optional{route_id_idx_t::invalid()}
                      : route_ids.find(id.view());
  };
  auto const push_unique = [](auto& vec, auto const value) {
    if (utl::find(vec, value) == end(vec)) {
      vec.push_back(value);
    }
  };

  auto const first_rule = transfer_rule_idx_t{tt.transfer_rules_.rules_.size()};
  auto rules = std::vector<transfer_rule>{};
  auto votes = std::vector<bool>{};

  utl::line_range{
      utl::make_buf_reader(file_content, progress_tracker->update_fn())}  //
      | utl::csv<csv_transfer>()  //
      |
      utl::for_each([&](csv_transfer const& t) {
        // Skip invalid/unkonwn transfer types.
        if (*t.transfer_type_ < 0 ||
            *t.transfer_type_ >
                static_cast<int>(transfer_type::kNoStaySeated)) {
          return;
        }

        auto const type = static_cast<transfer_type>(*t.transfer_type_);
        auto const from_gtfs_trip = find_trip(*t.from_trip_id_);
        auto const to_gtfs_trip = find_trip(*t.to_trip_id_);

        // Store stay-seated transfers + not stay seated transfers.
        if (type == transfer_type::kStaySeated ||
            type == transfer_type::kNoStaySeated) {
          if (t.from_trip_id_->empty() || t.to_trip_id_->empty()) {
            log(log_lvl::error, "loader.gtfs.transfers",
                "in-seat transfers (type 4, 5) require from_trip_id and "
                "to_trip_id");
            return;
          }
          if (!from_gtfs_trip.has_value() || !to_gtfs_trip.has_value()) {
            log(log_lvl::error, "loader.gtfs.transfers", "trip {} not found",
                from_gtfs_trip.has_value() ? t.to_trip_id_->view()
                                           : t.from_trip_id_->view());
            return;
          }

          auto const from = *from_gtfs_trip;
          auto const to = *to_gtfs_trip;
          if (from == to) {
            log(log_lvl::error, "loader.gtfs.transfers",
                "trip {} stays seated in itself", t.from_trip_id_->view());
            return;
          }
          if (type == transfer_type::kStaySeated) {
            // Stay seated transfer: annotate seated_in / seated out.
            push_unique(trips.data_[to].seated_in_, from);
            push_unique(trips.data_[from].seated_out_, to);
          } else {
            // Store transfer_type=5 - no in-seat transfer allowed
            // -> prevents stay-seated transfers for block_id trips.
            trips.no_stay_seated_.emplace(from, to);
          }
          return;
        }

        // Parse min_transfer_time if available (types 0 and 2 only).
        auto min_transfer_time = std::optional<duration_t>{};
        if (t.min_transfer_time_->has_value() &&
            (type == transfer_type::kRecommended ||
             type == transfer_type::kMinimumChangeTime)) {
          auto const seconds = std::int64_t{**t.min_transfer_time_};
          if (seconds < 0) {
            log(log_lvl::error, "loader.gtfs.transfers",
                "{} -> {}: negative min_transfer_time {}, row ignored",
                t.from_stop_id_->view(), t.to_stop_id_->view(), seconds);
            return;
          }

          auto const longest = footpath::kMaxDuration - duration_t{1};
          if ((seconds + 59) / 60 > longest.count()) {
            log(log_lvl::info, "loader.gtfs.transfers",
                "{} -> {}: min_transfer_time {} s capped to {} min",
                t.from_stop_id_->view(), t.to_stop_id_->view(), seconds,
                longest.count());
          }
          min_transfer_time = duration_t{static_cast<duration_t::rep>(
              std::min((seconds + 59) / 60, std::int64_t{longest.count()}))};
        }

        // Resolve from_stop_id and to_stop_id.
        auto const from_stop = utl::lookup(stops, t.from_stop_id_->view());
        auto const to_stop = utl::lookup(stops, t.to_stop_id_->view());
        if (!from_stop.has_value() || !to_stop.has_value()) {
          return;
        }

        // Drop transfer rule if adjusted footpath exceeds kMaxDuration.
        if (adjust_footpaths) {
          auto const adjusted =
              adjust_to_walk_speed(tt, *from_stop, *to_stop,
                                   min_transfer_time.value_or(duration_t{0}));
          if (!adjusted.has_value()) {
            return;
          }
          if (min_transfer_time.has_value()) {
            min_transfer_time = *adjusted;
          }
        }

        // Resolve from_route_id and to_route_id.
        auto const from_route = find_route(*t.from_route_id_);
        auto const to_route = find_route(*t.to_route_id_);
        if (!from_route.has_value() || !to_route.has_value()) {
          return;
        }

        // Resolve from_trip_id and to_trip_id.
        if (!from_gtfs_trip.has_value() || !to_gtfs_trip.has_value()) {
          return;
        }
        auto const trip_idx = [&](gtfs_trip_idx_t const g) {
          return g == gtfs_trip_idx_t::invalid() ? trip_idx_t::invalid()
                                                 : trips.data_[g].trip_idx_;
        };
        auto const from_trip = trip_idx(*from_gtfs_trip);
        auto const to_trip = trip_idx(*to_gtfs_trip);

        // Skip trip self-transfer rules:
        // Transferring into the same trip can't yield a better journey.
        if (from_trip != trip_idx_t::invalid() && from_trip == to_trip) {
          return;
        }

        // Store recommended / timed transfers.
        if (type == transfer_type::kRecommended ||
            type == transfer_type::kTimed) {
          tt.locations_.preferred_transfers_[*from_stop].push_back(
              preferred_transfer{.to_ = *to_stop,
                                 .from_trip_ = from_trip,
                                 .to_trip_ = to_trip,
                                 .from_route_ = *from_route,
                                 .to_route_ = *to_route});
        }

        auto const is_forbidden = type == transfer_type::kNotPossible;
        auto const is_timed = type == transfer_type::kTimed;
        auto const duration = is_forbidden
                                  ? std::optional{footpath::kMaxDuration}
                              : is_timed ? std::optional{duration_t{0}}
                                         : min_transfer_time;
        if (!duration.has_value()) {
          // Don't store non-binding rules.
          return;
        }

        auto const r = transfer_rule{.from_stop_ = *from_stop,
                                     .to_stop_ = *to_stop,
                                     .from_route_ = *from_route,
                                     .to_route_ = *to_route,
                                     .from_trip_ = from_trip,
                                     .to_trip_ = to_trip,
                                     .src_ = src,
                                     .duration_ = *duration,
                                     .specificity_ = get_specificity(
                                         tt, *from_stop, *to_stop, *from_route,
                                         *to_route, from_trip, to_trip)};

        rules.push_back(r);

        // Forbidden and timed transfers should not be
        // included in the majority fold.
        votes.push_back(!is_forbidden && !is_timed);
      });

  // Majority fold.
  fold_pair_defaults(tt, src, rules, votes);

  // Store rules (merge sort).
  store_rule_lookups(
      tt, interval{first_rule,
                   transfer_rule_idx_t{tt.transfer_rules_.rules_.size()}});
}

}  // namespace nigiri::loader::gtfs
