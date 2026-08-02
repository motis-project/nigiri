#include "nigiri/loader/gtfs/transfer_rules.h"

#include <algorithm>
#include <span>

#include "fmt/format.h"

#include "utl/erase_duplicates.h"
#include "utl/get_or_create.h"
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
        log(log_lvl::error, "loader.gtfs.transfers", "stop \"{}\" not found",
            id.view());
        ok_ = false;
        return location_idx_t::invalid();
      }
      return it->second;
    };
    auto const resolve_route = [&](utl::cstr const& id) {
      if (id.empty()) {
        return route_id_idx_t::invalid();
      }
      auto const it = routes.find(id.view());
      if (it == end(routes)) {
        ok_ = false;
        return route_id_idx_t::invalid();
      }
      return it->second->route_id_idx_;
    };
    auto const resolve_trip = [&](utl::cstr const& id) {
      if (id.empty()) {
        return gtfs_trip_idx_t::invalid();
      }
      auto const it = trips.trips_.find(id.view());
      if (it == end(trips.trips_)) {
        ok_ = false;
        return gtfs_trip_idx_t::invalid();
      }
      return it->second;
    };
    from_stop_ = resolve_stop(*t.from_stop_id_);
    to_stop_ = resolve_stop(*t.to_stop_id_);
    from_route_ = resolve_route(*t.from_route_id_);
    to_route_ = resolve_route(*t.to_route_id_);
    from_trip_ = resolve_trip(*t.from_trip_id_);
    to_trip_ = resolve_trip(*t.to_trip_id_);
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

using virt_location_idx_t = location_idx_t;

// one side (from/to) of one rule
using sided_rule_idx_t = cista::strong<std::uint32_t, struct sided_rule_idx_>;

// all rules that apply to a trip stop
using sig_t = std::vector<sided_rule_idx_t>;

sided_rule_idx_t side_ref(rule_idx_t const rule_idx, bool const is_from) {
  return sided_rule_idx_t{static_cast<std::uint32_t>(to_idx(rule_idx) << 1U) |
                          (is_from ? 0U : 1U)};
}

std::optional<bool /* stop got matched exactly */> rule_applies(
    timetable const& tt,
    location_idx_t const rule_stop,
    location_idx_t const l) {
  if (rule_stop == l) {
    return true;
  } else if (tt.locations_.parents_[l] == rule_stop) {
    return false;
  } else {
    return std::nullopt;  // -> stops are not related
  }
}

struct candidate {
  auto operator<=>(candidate const&) const = default;

  // keep member order for lexicographical comparison
  specificity specificity_{specificity::kStopsOnly};
  std::uint8_t n_exact_stops_{0U};  // from+to matches station or stop
  std::uint32_t rule_idx_{0U};
};

void apply_rules(timetable& tt,
                 vector_map<rule_idx_t, rule> const& rules,
                 trip_data& trips,
                 std::size_t const n_routes) {
  // Build (route -> trip) map.
  auto route_trips = vector_map<route_id_idx_t, std::vector<gtfs_trip_idx_t>>{};
  route_trips.resize(static_cast<unsigned>(n_routes));
  for (auto i = gtfs_trip_idx_t{0U}; i != trips.data_.size(); ++i) {
    auto const& t = trips.data_[i];
    if (!t.stop_seq_.empty() && t.flex_stops_.empty()) {
      route_trips[t.route_].emplace_back(i);
    }
  }

  // Build ((trip, stop position) -> matched (sided) rules) map.
  auto trip_stop_sided_rules =
      hash_map<pair<gtfs_trip_idx_t, stop_idx_t>, sig_t>{};
  for (auto rule_idx = rule_idx_t{0U}; rule_idx != rules.size(); ++rule_idx) {
    auto const& r = rules[rule_idx];
    auto const match_side = [&](location_idx_t const rule_stop,  //
                                route_id_idx_t const route,
                                gtfs_trip_idx_t const trip,
                                bool const is_from_side) {
      if (route == route_id_idx_t::invalid() &&
          trip == gtfs_trip_idx_t::invalid()) {
        return;  // no split
      }

      auto const check_trip = [&](gtfs_trip_idx_t const trp_idx) {
        auto const& t = trips.data_[trp_idx];
        if (t.stop_seq_.empty() || !t.flex_stops_.empty()) {
          return;
        }

        auto const n_stops = static_cast<stop_idx_t>(t.stop_seq_.size());
        for (auto pos = stop_idx_t{0U}; pos != n_stops; ++pos) {
          auto const l = stop{t.stop_seq_[pos]}.location_idx();
          if (rule_applies(tt, rule_stop, l)) {
            trip_stop_sided_rules[{trp_idx, pos}].push_back(
                side_ref(rule_idx, is_from_side));
          }
        }
      };

      if (trip != gtfs_trip_idx_t::invalid()) {
        check_trip(trip);
      } else {
        for (auto const trp_idx : route_trips[route]) {
          check_trip(trp_idx);
        }
      }
    };

    match_side(r.from_stop_, r.from_route_, r.from_trip_, true);
    match_side(r.to_stop_, r.to_route_, r.to_trip_, false);
  }

  // Generate virtual locations.
  auto const first_virt = virt_location_idx_t{tt.n_locations()};
  auto virt_locs =
      hash_map<std::pair<location_idx_t, sig_t>, virt_location_idx_t>{};
  auto side_groups =
      hash_map<sided_rule_idx_t, std::vector<virt_location_idx_t>>{};
  for (auto& [trip_stop, sig] : trip_stop_sided_rules) {
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

          auto const v = virt_location_idx_t{register_location(tt, l)};
          tt.locations_.children_[base].emplace_back(v);

          for (auto const side : sig) {
            side_groups[side].push_back(v);
          }

          return v;
        });

    t.stop_seq_[pos] =
        stop{virt, s.in_allowed(), s.out_allowed(), s.in_allowed_wheelchair(),
             s.out_allowed_wheelchair()}
            .value();
  }

  // Let rules compete for specificity on all location pairs they apply to.
  auto const get_subtree_locations = [&](location_idx_t const s,
                                         std::vector<location_idx_t>& nodes) {
    nodes.clear();
    nodes.emplace_back(s);
    for (auto const platform : tt.locations_.children_[s]) {
      nodes.emplace_back(platform);
      for (virt_location_idx_t const virt : tt.locations_.children_[platform]) {
        nodes.emplace_back(virt);
      }
    }
  };

  auto const base_of = [&](location_idx_t const l) {
    return tt.locations_.types_[l] == location_type::kVirt
               ? tt.locations_.parents_[l]
               : l;
  };

  auto most_specific =
      hash_map<pair<location_idx_t, location_idx_t>, candidate>{};
  auto from_buf = std::vector<location_idx_t>{};
  auto to_buf = std::vector<location_idx_t>{};
  for (auto rule_idx = rule_idx_t{0U}; rule_idx != rules.size(); ++rule_idx) {
    auto const& r = rules[rule_idx];
    auto const get_side_nodes = [&](location_idx_t const rule_stop,  //
                                    route_id_idx_t const route,  //
                                    gtfs_trip_idx_t const trip,
                                    bool const is_from_side,
                                    std::vector<location_idx_t>& buf)
        -> std::span<location_idx_t const> {
      if (route == route_id_idx_t::invalid() &&
          trip == gtfs_trip_idx_t::invalid()) {
        get_subtree_locations(rule_stop, buf);
        return buf;
      }
      auto const it = side_groups.find(side_ref(rule_idx, is_from_side));
      return it == end(side_groups) ? std::span<location_idx_t const>{}
                                    : std::span{it->second};
    };

    auto const from_nodes = get_side_nodes(r.from_stop_, r.from_route_,
                                           r.from_trip_, true, from_buf);
    auto const to_nodes =
        get_side_nodes(r.to_stop_, r.to_route_, r.to_trip_, false, to_buf);

    auto const rule_specificity = r.get_specificity();
    for (auto const x : from_nodes) {
      for (auto const y : to_nodes) {
        if (x == y) {
          continue;
        }

        auto const c = candidate{
            .specificity_ = rule_specificity,
            .n_exact_stops_ = static_cast<std::uint8_t>(
                rule_applies(tt, r.from_stop_, base_of(x)).value_or(false) +
                rule_applies(tt, r.to_stop_, base_of(y)).value_or(false)),
            .rule_idx_ = to_idx(rule_idx)};
        auto& m =
            utl::get_or_create(most_specific, pair{x, y}, [&]() { return c; });
        m = std::max(m, c);
      }
    }
  }

  // Write most specific transfers.
  for (auto const& [pair, c] : most_specific) {
    auto const& r = rules[rule_idx_t{c.rule_idx_}];
    tt.locations_.transfer_rule_fps_[pair.first].emplace_back(
        footpath{pair.second, r.forbidden_ ? footpath::kMaxDuration : r.time_});
  }

  // Apply base transfer time between all virtual locations.
  for (auto virt = first_virt; virt != tt.n_locations(); ++virt) {
    auto const base = tt.locations_.parents_[virt];
    auto const add = [&](location_idx_t const x, location_idx_t const y) {
      if (!most_specific.contains({x, y})) {
        tt.locations_.transfer_rule_fps_[x].emplace_back(
            footpath{y, tt.locations_.transfer_time_[base]});
      }
    };

    add(virt, base);
    add(base, virt);
    for (virt_location_idx_t const sibling : tt.locations_.children_[base]) {
      if (sibling != virt &&
          tt.locations_.types_[sibling] == location_type::kVirt) {
        add(virt, sibling);  // the other direction is added for the sibling
      }
    }
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

  auto progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Read Transfers").in_high(file_content.size());

  auto rules = vector_map<rule_idx_t, rule>{};

  utl::line_range{
      utl::make_buf_reader(file_content, progress_tracker->update_fn())}  //
      | utl::csv<csv_transfer>()  //
      | utl::for_each([&](csv_transfer const& t) {
          auto const type = static_cast<transfer_type>(*t.transfer_type_);

          switch (type) {
            case transfer_type::kRecommended:
            case transfer_type::kTimed:
            case transfer_type::kMinimumChangeTime: [[fallthrough]];
            case transfer_type::kNotPossible: {
              auto const r = rule{t, type, stops, routes, trips};
              if (!r.ok_) {
                return;
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

              auto const enforceable = type != transfer_type::kRecommended ||
                                       t.min_transfer_time_->has_value();

              if (!r.forbidden_ &&
                  r.get_specificity() == specificity::kStopsOnly) {
                auto const transfer_time =
                    duration_t{t.min_transfer_time_->value_or(0) / 60};
                if (r.from_stop_ == r.to_stop_) {
                  if (enforceable) {
                    tt.locations_.transfer_time_[r.from_stop_] = transfer_time;
                  }
                } else {
                  tt.locations_.preprocessing_footpaths_out_[r.from_stop_]
                      .emplace_back(r.to_stop_, transfer_time);
                  tt.locations_.preprocessing_footpaths_in_[r.to_stop_]
                      .emplace_back(r.from_stop_, transfer_time);
                }
              }

              if (enforceable) {
                rules.emplace_back(r);
              }

              return;
            }

            case transfer_type::kStaySeated: {
              if (t.from_trip_id_->empty() || t.to_trip_id_->empty()) {
                log(log_lvl::error, "loader.gtfs.transfers",
                    "stay seated transfers require from_trip_id and "
                    "to_trip_id");
                return;
              }

              auto const from_it = trips.trips_.find(t.from_trip_id_->view());
              if (from_it == end(trips.trips_)) {
                log(log_lvl::error, "nigiri.loader.gtfs.seated",
                    "trip {} not found", t.from_trip_id_->view());
                return;
              }

              auto const to_it = trips.trips_.find(t.to_trip_id_->view());
              if (to_it == end(trips.trips_)) {
                log(log_lvl::error, "nigiri.loader.gtfs.seated",
                    "trip {} not found", t.to_trip_id_->view());
                return;
              }

              trips.data_[to_it->second].seated_in_.push_back(from_it->second);
              trips.data_[from_it->second].seated_out_.push_back(to_it->second);
              return;
            }

            case transfer_type::kNoStaySeated: [[fallthrough]];
            default: return;
          }
        });

  if (!rules.empty()) {
    apply_rules(tt, rules, trips, routes.size());
  }
}

}  // namespace nigiri::loader::gtfs
