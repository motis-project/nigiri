#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "geo/latlng.h"

#include "nigiri/footpath_restriction.h"
#include "nigiri/types.h"

namespace nigiri {
struct station_paths;
struct timetable;
}  // namespace nigiri

namespace nigiri::loader {

// Pedestrian network inside stations (GTFS pathways, ISA Umsteigebauwerke).
// Filled by the loaders, turned into restricted footpaths by build_footpaths.
struct station_graph {
  using node_idx_t = std::uint32_t;
  using edge_idx_t = std::uint32_t;

  enum class node_type : std::uint8_t {
    kStop,
    kEntrance,
    kGeneric,
    kBoardingArea
  };

  enum class edge_mode : std::uint8_t {
    kWalkway,
    kStairs,
    kTravelator,
    kEscalator,
    kElevator,
    kFareGate,
    kExitGate
  };

  struct node {
    // Footpath endpoint (stop where transports halt), invalid otherwise.
    location_idx_t l_{location_idx_t::invalid()};
    std::optional<geo::latlng> pos_;
    std::optional<float> level_;
    node_type type_{node_type::kGeneric};
    std::string name_{};
    // Location id (any source) that sets l_ at finalize, e.g. an ISA area's
    // global stop id (DHID).
    std::string ref_{};
    // GTFS stop_access=1: reachable from the street without an entrance.
    bool is_street_accessible_{false};
  };

  struct edge {
    node_idx_t from_;
    node_idx_t to_;
    std::uint32_t walk_seconds_{0U};  // scaled by the walking speed factor
    std::uint32_t fixed_seconds_{0U};  // rides and waiting: not scaled
    footpath_restriction restriction_{};  // fixed_ unused
    edge_mode mode_{edge_mode::kWalkway};
    std::string signpost_{};
  };

  node_idx_t add_node(node n) {
    nodes_.push_back(std::move(n));
    return static_cast<node_idx_t>(nodes_.size() - 1U);
  }

  void add_edge(edge e) { edges_.push_back(std::move(e)); }

  bool empty() const { return edges_.empty(); }

  std::vector<node> nodes_;
  std::vector<edge> edges_;
};

// A route through a station graph between two footpath endpoints (stops or
// entrances) that is the best one for some user: Pareto-optimal regarding
// walking time, fixed time and restrictions.
struct station_route {
  bool is_entrance_route() const {
    return from_ == location_idx_t::invalid() ||
           to_ == location_idx_t::invalid();
  }

  location_idx_t from_;  // invalid: from_node_ is an entrance
  location_idx_t to_;  // invalid: to_node_ is an entrance
  station_graph::node_idx_t from_node_;
  station_graph::node_idx_t to_node_;
  std::uint32_t walk_seconds_;
  std::uint32_t fixed_seconds_;
  footpath_restriction restriction_;  // fixed_ set from fixed_seconds_
  std::vector<station_graph::edge_idx_t> edges_;
};

// Sets l_ of nodes with ref_ to the location with this id (first source).
void resolve_refs(timetable const&, station_graph&);

// Routes stop -> stop, entrance -> stop and stop -> entrance.
std::vector<station_route> get_station_routes(station_graph const&);

// Writes nodes, edges (as pieces), entrances, the stop -> stop routes (as
// paths) and the entrance routes (as entrance paths).
void write_station_paths(station_graph const&,
                         std::vector<station_route> const&,
                         station_paths&);

}  // namespace nigiri::loader
