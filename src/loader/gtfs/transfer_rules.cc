#include "nigiri/loader/gtfs/transfer_rules.h"

#include <algorithm>
#include <map>

#include "fmt/format.h"

#include "utl/helpers/algorithm.h"

#include "nigiri/loader/register.h"
#include "nigiri/logging.h"
#include "nigiri/stop.h"
#include "nigiri/timetable.h"

namespace nigiri::loader::gtfs {

namespace {

constexpr auto const kNotPossible = std::uint8_t{3U};

struct rule {
  bool qualified() const {
    return from_route_ != route_id_idx_t::invalid() ||
           to_route_ != route_id_idx_t::invalid() ||
           from_trip_ != gtfs_trip_idx_t::invalid() ||
           to_trip_ != gtfs_trip_idx_t::invalid();
  }

  location_idx_t from_stop_{location_idx_t::invalid()};
  location_idx_t to_stop_{location_idx_t::invalid()};
  route_id_idx_t from_route_{route_id_idx_t::invalid()};
  route_id_idx_t to_route_{route_id_idx_t::invalid()};
  gtfs_trip_idx_t from_trip_{gtfs_trip_idx_t::invalid()};
  gtfs_trip_idx_t to_trip_{gtfs_trip_idx_t::invalid()};
  bool forbidden_{false};
  duration_t time_{0};
};

// side reference: rule index * 2, bit 0 = to-side
constexpr std::uint32_t side_ref(std::size_t const rule_idx, bool const from) {
  return static_cast<std::uint32_t>(rule_idx << 1U) | (from ? 0U : 1U);
}

// specificity of the stop relation:
// 2 = exact, 1 = rule stop is ancestor of l (station-level cascade),
// 0 = rule stop is descendant of l (data error, applied defensively),
// -1 = unrelated
int exactness(timetable const& tt,
              location_idx_t const rule_stop,
              location_idx_t const l) {
  if (rule_stop == l) {
    return 2;
  }
  auto const is_ancestor = [&](location_idx_t const anc,
                               location_idx_t const x) {
    auto p = tt.locations_.parents_[x];
    for (auto i = 0U; i != 8U && p != location_idx_t::invalid(); ++i) {
      if (p == anc) {
        return true;
      }
      p = tt.locations_.parents_[p];
    }
    return false;
  };
  if (is_ancestor(rule_stop, l)) {
    return 1;
  }
  if (is_ancestor(l, rule_stop)) {
    return 0;
  }
  return -1;
}

using sig_t = std::vector<std::uint32_t>;

}  // namespace

void build_transfer_rules(timetable& tt,
                          std::vector<raw_transfer_rule> const& raw,
                          stops_map_t const& stops,
                          route_map_t const& routes,
                          trip_data& trips) {
  if (raw.empty()) {
    return;
  }

  auto const timer = scoped_timer{"loader.gtfs.transfer_rules"};

  // --- resolve ids ---------------------------------------------------------
  auto rules = std::vector<rule>{};
  rules.reserve(raw.size());
  auto n_unresolved = 0U;
  for (auto const& t : raw) {
    auto r = rule{};
    auto ok = true;
    auto const resolve_stop = [&](std::string const& id) {
      auto const it = stops.find(id);
      if (it == end(stops)) {
        ok = false;
        return location_idx_t::invalid();
      }
      return it->second;
    };
    auto const resolve_route = [&](std::string const& id) {
      if (id.empty()) {
        return route_id_idx_t::invalid();
      }
      auto const it = routes.find(id);
      if (it == end(routes)) {
        ok = false;
        return route_id_idx_t::invalid();
      }
      return it->second->route_id_idx_;
    };
    auto const resolve_trip = [&](std::string const& id) {
      if (id.empty()) {
        return gtfs_trip_idx_t::invalid();
      }
      auto const it = trips.trips_.find(id);
      if (it == end(trips.trips_)) {
        ok = false;
        return gtfs_trip_idx_t::invalid();
      }
      return it->second;
    };
    r.from_stop_ = resolve_stop(t.from_stop_id_);
    r.to_stop_ = resolve_stop(t.to_stop_id_);
    r.from_route_ = resolve_route(t.from_route_id_);
    r.to_route_ = resolve_route(t.to_route_id_);
    r.from_trip_ = resolve_trip(t.from_trip_id_);
    r.to_trip_ = resolve_trip(t.to_trip_id_);
    r.forbidden_ = t.type_ == kNotPossible;
    r.time_ = duration_t{(t.min_transfer_time_ + 59) / 60};
    if (!ok) {
      ++n_unresolved;
      continue;
    }
    rules.emplace_back(r);
  }
  if (n_unresolved != 0U) {
    log(log_lvl::error, "loader.gtfs.transfer_rules",
        "{} transfer rules dropped (unresolved stop/route/trip id)",
        n_unresolved);
  }
  if (rules.empty()) {
    return;
  }

  // --- match qualified rule sides to (trip, stop position) -----------------
  auto route_trips = hash_map<route_id_idx_t, std::vector<gtfs_trip_idx_t>>{};
  for (auto i = gtfs_trip_idx_t{0U}; i != trips.data_.size(); ++i) {
    auto const& t = trips.data_[i];
    if (!t.stop_seq_.empty() && t.flex_stops_.empty()) {
      route_trips[t.route_].emplace_back(i);
    }
  }

  // (trip, stop position) -> matched side references
  auto memberships =
      hash_map<pair<gtfs_trip_idx_t, std::uint32_t>, sig_t>{};
  for (auto rule_idx = std::size_t{0U}; rule_idx != rules.size(); ++rule_idx) {
    auto const& r = rules[rule_idx];
    auto const match_side = [&](location_idx_t const rule_stop,
                                route_id_idx_t const route,
                                gtfs_trip_idx_t const trip,
                                bool const from) {
      if (route == route_id_idx_t::invalid() &&
          trip == gtfs_trip_idx_t::invalid()) {
        return;  // unqualified side: no split necessary
      }
      auto const check_trip = [&](gtfs_trip_idx_t const t_idx) {
        auto const& t = trips.data_[t_idx];
        if (t.stop_seq_.empty() || !t.flex_stops_.empty()) {
          return;
        }
        for (auto pos = 0U; pos != t.stop_seq_.size(); ++pos) {
          auto const l = stop{t.stop_seq_[pos]}.location_idx();
          if (exactness(tt, rule_stop, l) >= 0) {
            memberships[{t_idx, pos}].emplace_back(side_ref(rule_idx, from));
          }
        }
      };
      if (trip != gtfs_trip_idx_t::invalid()) {
        check_trip(trip);
      } else if (auto const it = route_trips.find(route);
                 it != end(route_trips)) {
        for (auto const t_idx : it->second) {
          check_trip(t_idx);
        }
      }
    };
    match_side(r.from_stop_, r.from_route_, r.from_trip_, true);
    match_side(r.to_stop_, r.to_route_, r.to_trip_, false);
  }

  // --- partition into groups, create virtual locations, rewrite stop seqs --
  auto group_locations = std::map<std::pair<location_idx_t, sig_t>,
                                  location_idx_t>{};  // (base, sig) -> child
  auto base_groups = hash_map<location_idx_t, std::vector<location_idx_t>>{};
  auto side_groups = hash_map<std::uint32_t, std::vector<location_idx_t>>{};
  auto n_created = hash_map<location_idx_t, unsigned>{};
  for (auto& [key, sig] : memberships) {
    auto const [t_idx, pos] = key;
    auto& t = trips.data_[t_idx];
    auto const s = stop{t.stop_seq_[pos]};
    auto const base = s.location_idx();
    utl::sort(sig);
    sig.erase(std::unique(begin(sig), end(sig)), end(sig));

    auto const it = group_locations.find({base, sig});
    auto child = location_idx_t::invalid();
    if (it != end(group_locations)) {
      child = it->second;
    } else {
      auto l = location{tt, base};
      auto const id = fmt::format("TG:{}:{}", l.id_, n_created[base]++);
      l.id_ = id;
      l.type_ = location_type::kGeneratedTransfer;
      l.parent_ = base;
      child = register_location(tt, l);
      tt.locations_.children_[base].emplace_back(child);
      group_locations.emplace(std::pair{base, sig}, child);
      base_groups[base].emplace_back(child);
      for (auto const side : sig) {
        side_groups[side].emplace_back(child);
      }
    }

    t.stop_seq_[pos] = stop{child, s.in_allowed(), s.out_allowed(),
                            s.in_allowed_wheelchair(),
                            s.out_allowed_wheelchair()}
                           .value();
  }

  // --- emit rule edges ------------------------------------------------------
  // locations that carry trips after the rewrite
  auto trip_carrying = hash_set<location_idx_t>{};
  for (auto const& t : trips.data_) {
    for (auto const s : t.stop_seq_) {
      trip_carrying.emplace(stop{s}.location_idx());
    }
  }

  // all nodes an unqualified side with stop s can refer to:
  // trip-carrying locations in the subtree of s and among the ancestors of s
  auto const unqualified_nodes = [&](location_idx_t const s) {
    auto nodes = std::vector<location_idx_t>{};
    auto stack = std::vector<location_idx_t>{s};
    while (!stack.empty()) {
      auto const l = stack.back();
      stack.pop_back();
      if (trip_carrying.contains(l)) {
        nodes.emplace_back(l);
      }
      for (auto const c : tt.locations_.children_[l]) {
        stack.emplace_back(c);
      }
    }
    auto p = tt.locations_.parents_[s];
    for (auto i = 0U; i != 8U && p != location_idx_t::invalid(); ++i) {
      if (trip_carrying.contains(p)) {
        nodes.emplace_back(p);
      }
      p = tt.locations_.parents_[p];
    }
    return nodes;
  };

  // base location of a node (= itself if it is not a transfer group)
  auto const base_of = [&](location_idx_t const l) {
    return tt.locations_.types_[l] == location_type::kGeneratedTransfer
               ? tt.locations_.parents_[l]
               : l;
  };

  struct candidate {
    int score_;
    std::size_t rule_;
  };
  auto best = hash_map<pair<location_idx_t, location_idx_t>, candidate>{};
  auto from_nodes = std::vector<location_idx_t>{};
  auto to_nodes = std::vector<location_idx_t>{};
  for (auto rule_idx = std::size_t{0U}; rule_idx != rules.size(); ++rule_idx) {
    auto const& r = rules[rule_idx];
    auto const side_nodes = [&](location_idx_t const rule_stop,
                                route_id_idx_t const route,
                                gtfs_trip_idx_t const trip, bool const from,
                                std::vector<location_idx_t>& nodes) {
      nodes.clear();
      if (route == route_id_idx_t::invalid() &&
          trip == gtfs_trip_idx_t::invalid()) {
        nodes = unqualified_nodes(rule_stop);
      } else if (auto const it = side_groups.find(side_ref(rule_idx, from));
                 it != end(side_groups)) {
        nodes = it->second;
      }
    };
    side_nodes(r.from_stop_, r.from_route_, r.from_trip_, true, from_nodes);
    side_nodes(r.to_stop_, r.to_route_, r.to_trip_, false, to_nodes);

    auto const side_score = [&](location_idx_t const rule_stop,
                                route_id_idx_t const route,
                                gtfs_trip_idx_t const trip,
                                location_idx_t const node) {
      auto const qualifier = trip != gtfs_trip_idx_t::invalid() ? 8
                             : route != route_id_idx_t::invalid() ? 4
                                                                  : 0;
      return qualifier + exactness(tt, rule_stop, base_of(node));
    };

    for (auto const x : from_nodes) {
      for (auto const y : to_nodes) {
        if (x == y) {
          continue;  // same-location transfers use transfer_time_
        }
        auto const score =
            side_score(r.from_stop_, r.from_route_, r.from_trip_, x) +
            side_score(r.to_stop_, r.to_route_, r.to_trip_, y);
        auto const it = best.find({x, y});
        if (it == end(best) || score > it->second.score_) {
          best[{x, y}] = candidate{score, rule_idx};
        }
      }
    }
  }

  auto n_edges = 0U;
  auto n_forbidden = 0U;
  for (auto const& [pair, c] : best) {
    auto const& r = rules[c.rule_];
    auto const duration = r.forbidden_ ? footpath::kMaxDuration : r.time_;
    n_forbidden += r.forbidden_ ? 1U : 0U;
    tt.locations_.transfer_rule_fps_[pair.first].emplace_back(
        footpath{pair.second, duration});
    ++n_edges;
  }

  // same-base safety clique: pairs among {base} + groups without a rule get
  // the base transfer time (prevents 0-minute street-routed transfers
  // between co-located virtual locations)
  for (auto const& [base, groups] : base_groups) {
    auto nodes = std::vector<location_idx_t>{base};
    nodes.insert(end(nodes), begin(groups), end(groups));
    for (auto const x : nodes) {
      for (auto const y : nodes) {
        if (x == y || best.contains({x, y})) {
          continue;
        }
        tt.locations_.transfer_rule_fps_[x].emplace_back(
            footpath{y, tt.locations_.transfer_time_[base]});
        ++n_edges;
      }
    }
  }

  log(log_lvl::info, "loader.gtfs.transfer_rules",
      "{} transfer rules: {} transfer group locations at {} stops, {} edges "
      "({} forbidden)",
      rules.size(), group_locations.size(), base_groups.size(), n_edges,
      n_forbidden);
}

}  // namespace nigiri::loader::gtfs
