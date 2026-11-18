#include "nigiri/loader/station_graph.h"

#include <cmath>
#include <algorithm>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <queue>
#include <string>
#include <tuple>
#include <utility>

#include "utl/enumerate.h"
#include "utl/erase_if.h"
#include "utl/get_or_create.h"
#include "utl/helpers/algorithm.h"
#include "utl/parallel_for.h"

#include "nigiri/logging.h"
#include "nigiri/station_paths.h"
#include "nigiri/timetable.h"

namespace nigiri::loader {

constexpr auto const kNoLabel = std::numeric_limits<std::uint32_t>::max();

struct station_label {
  std::uint32_t cost() const { return walk_seconds_ + fixed_seconds_; }

  bool dominates(station_label const& o) const {
    return walk_seconds_ <= o.walk_seconds_ &&
           fixed_seconds_ <= o.fixed_seconds_ &&
           is_less_restricted(restriction_, o.restriction_);
  }

  std::uint32_t walk_seconds_;
  std::uint32_t fixed_seconds_;
  footpath_restriction restriction_;
  station_graph::node_idx_t node_;
  std::uint32_t pred_;  // label index
  station_graph::edge_idx_t edge_;  // edge from the predecessor
  bool is_dead_;
};

struct station_search {
  using queue_entry = std::pair<std::uint32_t, std::uint32_t>;  // cost, label

  std::vector<station_label> labels_;
  std::vector<std::vector<std::uint32_t>> bags_;
  std::vector<station_graph::node_idx_t> touched_;
  std::priority_queue<queue_entry,
                      std::vector<queue_entry>,
                      std::greater<queue_entry>>
      pq_;
};

void search(station_graph const& g,
            std::vector<std::vector<station_graph::edge_idx_t>> const& out,
            station_graph::node_idx_t const source,
            station_search& s) {
  s.bags_.resize(g.nodes_.size());
  s.labels_.push_back(station_label{.walk_seconds_ = 0U,
                                    .fixed_seconds_ = 0U,
                                    .restriction_ = {},
                                    .node_ = source,
                                    .pred_ = kNoLabel,
                                    .edge_ = 0U,
                                    .is_dead_ = false});
  s.bags_[source].push_back(0U);
  s.touched_.push_back(source);
  s.pq_.emplace(0U, 0U);

  while (!s.pq_.empty()) {
    auto const [cost, label_idx] = s.pq_.top();
    s.pq_.pop();
    if (s.labels_[label_idx].is_dead_) {
      continue;
    }

    auto const l = s.labels_[label_idx];
    for (auto const e : out[l.node_]) {
      auto const& edge = g.edges_[e];
      if (edge.to_ == source) {
        continue;
      }

      auto const next = station_label{
          .walk_seconds_ = l.walk_seconds_ + edge.walk_seconds_,
          .fixed_seconds_ = l.fixed_seconds_ + edge.fixed_seconds_,
          .restriction_ = l.restriction_ + edge.restriction_,
          .node_ = edge.to_,
          .pred_ = label_idx,
          .edge_ = e,
          .is_dead_ = false};

      auto& bag = s.bags_[edge.to_];
      if (utl::any_of(bag, [&](std::uint32_t const i) {
            return s.labels_[i].dominates(next);
          })) {
        continue;
      }
      utl::erase_if(bag, [&](std::uint32_t const i) {
        if (next.dominates(s.labels_[i])) {
          s.labels_[i].is_dead_ = true;
          return true;
        }
        return false;
      });
      if (bag.empty()) {
        s.touched_.push_back(edge.to_);
      }

      auto const next_idx = static_cast<std::uint32_t>(s.labels_.size());
      s.labels_.push_back(next);
      bag.push_back(next_idx);
      s.pq_.emplace(next.cost(), next_idx);
    }
  }
}

void resolve_refs(timetable const& tt, station_graph& g) {
  auto n_resolved = 0U;
  auto n_refs = 0U;
  for (auto& n : g.nodes_) {
    if (n.ref_.empty() || n.l_ != location_idx_t::invalid()) {
      continue;
    }
    ++n_refs;
    for (auto src = source_idx_t{0U}; src != tt.n_sources(); ++src) {
      auto const it = tt.locations_.location_id_to_idx_.find(
          location_id{.id_ = n.ref_, .src_ = src});
      if (it != end(tt.locations_.location_id_to_idx_)) {
        n.l_ = it->second;
        ++n_resolved;
        break;
      }
    }
  }
  if (n_refs != 0U) {
    log(log_lvl::info, "loader.station_graph", "resolved {}/{} references",
        n_resolved, n_refs);
  }
}

std::vector<station_route> get_station_routes(station_graph const& g) {
  if (g.empty()) {
    return {};
  }

  auto const timer = scoped_timer{"loader.station_graph.routes"};

  auto out =
      std::vector<std::vector<station_graph::edge_idx_t>>(g.nodes_.size());
  for (auto e = station_graph::edge_idx_t{0U}; e != g.edges_.size(); ++e) {
    out[g.edges_[e].from_].push_back(e);
  }

  auto const is_stop = [&](station_graph::node_idx_t const n) {
    return g.nodes_[n].l_ != location_idx_t::invalid();
  };
  auto const is_entrance = [&](station_graph::node_idx_t const n) {
    return !is_stop(n) &&
           g.nodes_[n].type_ == station_graph::node_type::kEntrance;
  };

  auto sources = std::vector<station_graph::node_idx_t>{};
  for (auto n = station_graph::node_idx_t{0U}; n != g.nodes_.size(); ++n) {
    if ((is_stop(n) || is_entrance(n)) && !out[n].empty()) {
      sources.push_back(n);
    }
  }

  auto routes = std::vector<std::vector<station_route>>(sources.size());
  utl::parallel_for_run_threadlocal<station_search>(
      sources.size(), [&](station_search& s, std::size_t const i) {
        auto const source = sources[i];
        auto const from = g.nodes_[source].l_;
        search(g, out, source, s);

        for (auto const n : s.touched_) {
          auto const to = g.nodes_[n].l_;
          auto const is_target =
              is_stop(source) ? (is_stop(n) && to != from) || is_entrance(n)
                              : is_stop(n);
          if (is_target) {
            for (auto const label_idx : s.bags_[n]) {
              auto const& l = s.labels_[label_idx];
              auto r = station_route{.from_ = from,
                                     .to_ = to,
                                     .from_node_ = source,
                                     .to_node_ = n,
                                     .walk_seconds_ = l.walk_seconds_,
                                     .fixed_seconds_ = l.fixed_seconds_,
                                     .restriction_ = l.restriction_,
                                     .edges_ = {}};
              r.restriction_.fixed_ =
                  footpath_restriction::to_fixed(l.fixed_seconds_);
              for (auto x = label_idx; s.labels_[x].pred_ != kNoLabel;
                   x = s.labels_[x].pred_) {
                r.edges_.push_back(s.labels_[x].edge_);
              }
              std::reverse(begin(r.edges_), end(r.edges_));
              routes[i].push_back(std::move(r));
            }
          }
          s.bags_[n].clear();
        }
        s.touched_.clear();
        s.labels_.clear();
      });

  auto all = std::vector<station_route>{};
  for (auto& r : routes) {
    std::move(begin(r), end(r), std::back_inserter(all));
  }
  log(log_lvl::info, "loader.station_graph",
      "{} nodes, {} edges, {} sources -> {} routes", g.nodes_.size(),
      g.edges_.size(), sources.size(), all.size());
  return all;
}

station_piece_mode to_piece_mode(station_graph::edge_mode const m) {
  using em = station_graph::edge_mode;
  switch (m) {
    case em::kWalkway: return station_piece_mode::kWalkway;
    case em::kStairs: return station_piece_mode::kStairs;
    case em::kTravelator: return station_piece_mode::kTravelator;
    case em::kEscalator: return station_piece_mode::kEscalator;
    case em::kElevator: return station_piece_mode::kElevator;
    case em::kFareGate: return station_piece_mode::kFareGate;
    case em::kExitGate: return station_piece_mode::kExitGate;
  }
  return station_piece_mode::kWalkway;
}

void write_station_paths(station_graph const& g,
                         std::vector<station_route> const& routes,
                         station_paths& out) {
  auto const timer = scoped_timer{"loader.station_graph.write_paths"};

  auto strings = hash_map<std::string, station_string_idx_t>{};
  auto const get_string = [&](std::string const& s) {
    if (s.empty()) {
      return station_string_idx_t::invalid();
    }
    return utl::get_or_create(strings, s, [&]() {
      auto const idx = station_string_idx_t{out.strings_.size()};
      out.strings_.emplace_back(s);
      return idx;
    });
  };

  constexpr auto const kNaN = std::numeric_limits<double>::quiet_NaN();
  auto entrance_of = std::vector<station_entrance_idx_t>(
      g.nodes_.size(), station_entrance_idx_t::invalid());
  for (auto const [i, n] : utl::enumerate(g.nodes_)) {
    out.nodes_.push_back(station_node{
        .pos_ = n.pos_.value_or(geo::latlng{kNaN, kNaN}),
        .level_ = n.level_.value_or(std::numeric_limits<float>::quiet_NaN())});
    if (n.type_ == station_graph::node_type::kEntrance) {
      entrance_of[i] = station_entrance_idx_t{out.entrances_.size()};
      out.entrances_.push_back(
          station_entrance{.node_ = station_node_idx_t{out.nodes_.size() - 1U},
                           .name_ = get_string(n.name_)});
    }
  }

  for (auto const& e : g.edges_) {
    out.pieces_.push_back(
        station_piece{.from_ = station_node_idx_t{e.from_},
                      .to_ = station_node_idx_t{e.to_},
                      .signpost_ = get_string(e.signpost_),
                      .seconds_ = e.walk_seconds_ + e.fixed_seconds_,
                      .restriction_ = e.restriction_,
                      .mode_ = to_piece_mode(e.mode_)});
  }

  auto pieces = std::vector<station_piece_idx_t>{};
  auto const get_pieces = [&](station_route const& r) {
    pieces.clear();
    for (auto const e : r.edges_) {
      pieces.push_back(station_piece_idx_t{e});
    }
    return pieces;
  };

  // Entrance paths: only entrances with a position can be reached from the
  // street.
  struct entrance_route {
    station_entrance_idx_t entrance_;
    bool is_exit_;
    location_idx_t l_;
    std::uint32_t route_;
  };
  auto entrance_routes = std::vector<entrance_route>{};
  for (auto const [i, r] : utl::enumerate(routes)) {
    auto const is_exit = r.from_ != location_idx_t::invalid();
    auto const entrance_node = is_exit ? r.to_node_ : r.from_node_;
    if (!r.is_entrance_route() || !g.nodes_[entrance_node].pos_.has_value()) {
      continue;
    }
    entrance_routes.push_back(
        entrance_route{.entrance_ = entrance_of[entrance_node],
                       .is_exit_ = is_exit,
                       .l_ = is_exit ? r.from_ : r.to_,
                       .route_ = static_cast<std::uint32_t>(i)});
  }
  utl::sort(entrance_routes,
            [](entrance_route const& a, entrance_route const& b) {
              return std::tie(a.entrance_, a.is_exit_, a.l_, a.route_) <
                     std::tie(b.entrance_, b.is_exit_, b.l_, b.route_);
            });
  auto has_access = hash_set<location_idx_t>{};
  auto has_egress = hash_set<location_idx_t>{};
  for (auto const& x : entrance_routes) {
    auto const& r = routes[x.route_];
    out.entrance_paths_.push_back(
        station_entrance_path{.entrance_ = x.entrance_,
                              .is_exit_ = x.is_exit_,
                              .l_ = x.l_,
                              .restriction_ = r.restriction_,
                              .walk_seconds_ = r.walk_seconds_,
                              .fixed_seconds_ = r.fixed_seconds_});
    out.entrance_path_pieces_.emplace_back(get_pieces(r));
    (x.is_exit_ ? has_egress : has_access).insert(x.l_);
  }

  auto is_street_accessible = hash_set<location_idx_t>{};
  for (auto const& n : g.nodes_) {
    if (n.is_street_accessible_ && n.l_ != location_idx_t::invalid()) {
      is_street_accessible.insert(n.l_);
    }
  }
  auto entrance_only = std::vector<location_idx_t>{};
  for (auto const l : has_access) {
    if (has_egress.contains(l) && !is_street_accessible.contains(l)) {
      entrance_only.push_back(l);
    }
  }
  utl::sort(entrance_only);
  for (auto const l : entrance_only) {
    out.entrance_only_.push_back(l);
  }

  // Stop -> stop paths.
  auto order = std::vector<std::uint32_t>{};
  for (auto const [i, r] : utl::enumerate(routes)) {
    if (!r.is_entrance_route()) {
      order.push_back(static_cast<std::uint32_t>(i));
    }
  }
  std::sort(begin(order), end(order),
            [&](std::uint32_t const a, std::uint32_t const b) {
              return std::tie(routes[a].from_, routes[a].to_, a) <
                     std::tie(routes[b].from_, routes[b].to_, b);
            });
  for (auto const i : order) {
    auto const& r = routes[i];
    out.paths_.push_back(station_path{.from_ = r.from_,
                                      .to_ = r.to_,
                                      .restriction_ = r.restriction_,
                                      .walk_seconds_ = r.walk_seconds_,
                                      .fixed_seconds_ = r.fixed_seconds_});
    out.path_pieces_.emplace_back(get_pieces(r));
  }

  log(log_lvl::info, "loader.station_graph",
      "station paths: {} nodes, {} pieces, {} paths, {} entrances, {} "
      "entrance paths, {} stops only via entrances",
      out.nodes_.size(), out.pieces_.size(), out.paths_.size(),
      out.entrances_.size(), out.entrance_paths_.size(),
      out.entrance_only_.size());
}

}  // namespace nigiri::loader
