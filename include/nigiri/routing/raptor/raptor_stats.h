#pragma once

#include <cstdint>
#include <map>
#include <string>

namespace nigiri::routing {

struct raptor_stats {
  std::map<std::string, std::uint64_t> to_map() const {
    return {
        {"n_routing_time", n_routing_time_},
        {"n_footpaths_visited", n_footpaths_visited_},
        {"n_routes_visited", n_routes_visited_},
        {"n_earliest_trip_calls", n_earliest_trip_calls_},
        {"n_earliest_arrival_updated_by_route",
         n_earliest_arrival_updated_by_route_},
        {"n_earliest_arrival_updated_by_footpath",
         n_earliest_arrival_updated_by_footpath_},
        {"fp_update_prevented_by_lower_bound",
         fp_update_prevented_by_lower_bound_},
        {"route_update_prevented_by_lower_bound",
         route_update_prevented_by_lower_bound_},
        {"n_pruned_by_ping_bounds", n_pruned_by_ping_bounds_},
        {"gpu_lb_compute_us", gpu_lb_compute_us_},
        {"gpu_prelude_us", gpu_prelude_us_},
        {"gpu_main_us", gpu_main_us_},
        {"n_prelude_runs", n_prelude_runs_},
        {"n_prelude_with_connection", n_prelude_with_connection_},
        {"n_prelude_seed_entries", n_prelude_seed_entries_},
    };
  }

  raptor_stats operator+(raptor_stats const& o) const {
    auto copy = *this;
    copy.n_routing_time_ += o.n_routing_time_;
    copy.n_footpaths_visited_ += o.n_footpaths_visited_;
    copy.n_routes_visited_ += o.n_routes_visited_;
    copy.n_earliest_trip_calls_ += o.n_earliest_trip_calls_;
    copy.n_earliest_arrival_updated_by_route_ +=
        o.n_earliest_arrival_updated_by_route_;
    copy.n_earliest_arrival_updated_by_footpath_ +=
        o.n_earliest_arrival_updated_by_footpath_;
    copy.fp_update_prevented_by_lower_bound_ +=
        o.fp_update_prevented_by_lower_bound_;
    copy.route_update_prevented_by_lower_bound_ +=
        o.route_update_prevented_by_lower_bound_;
    copy.n_pruned_by_ping_bounds_ += o.n_pruned_by_ping_bounds_;
    copy.gpu_lb_compute_us_ += o.gpu_lb_compute_us_;
    copy.gpu_prelude_us_ += o.gpu_prelude_us_;
    copy.gpu_main_us_ += o.gpu_main_us_;
    copy.n_prelude_runs_ += o.n_prelude_runs_;
    copy.n_prelude_with_connection_ += o.n_prelude_with_connection_;
    copy.n_prelude_seed_entries_ += o.n_prelude_seed_entries_;
    return copy;
  }

  std::uint64_t n_routing_time_{0ULL};
  std::uint64_t n_footpaths_visited_{0ULL};
  std::uint64_t n_routes_visited_{0ULL};
  std::uint64_t n_earliest_trip_calls_{0ULL};
  std::uint64_t n_earliest_arrival_updated_by_route_{0ULL};
  std::uint64_t n_earliest_arrival_updated_by_footpath_{0ULL};
  std::uint64_t fp_update_prevented_by_lower_bound_{0ULL};
  std::uint64_t route_update_prevented_by_lower_bound_{0ULL};
  std::uint64_t n_pruned_by_ping_bounds_{0ULL};

  // GPU phase split (see gpu/raptor.cu execute/compute_component_bounds)
  std::uint64_t gpu_lb_compute_us_{0ULL};
  std::uint64_t gpu_prelude_us_{0ULL};
  std::uint64_t gpu_main_us_{0ULL};

  // ad-hoc transfer pattern prelude effectiveness: how many restricted
  // preludes ran / found >= 1 destination arrival / total seeded rounds
  std::uint64_t n_prelude_runs_{0ULL};
  std::uint64_t n_prelude_with_connection_{0ULL};
  std::uint64_t n_prelude_seed_entries_{0ULL};
};

}  // namespace nigiri::routing
