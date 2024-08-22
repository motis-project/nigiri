#pragma once

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/journey.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing::meat {

struct decision_graph {
  struct node {
    location_idx_t stop_id_;
    std::vector<int> out_;
    std::vector<int> in_;
  };

  struct arc {
    int dep_node_;
    int arr_node_;
    unixtime_t dep_time_;
    unixtime_t arr_time_;
    unixtime_t meat_;
    std::variant<journey::run_enter_exit, footpath> uses_;
    double use_prob_;
  };

  int node_count() const { return nodes_.size(); }

  int arc_count() const { return arcs_.size(); }

  void compute_use_probabilities(timetable const& tt, delta_t max_delay);

  std::vector<node> nodes_;
  std::vector<arc> arcs_;
  int source_node_;
  int target_node_;
  int first_arc_;
};

}  // namespace nigiri::routing::meat
