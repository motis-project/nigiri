#pragma once

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/journey.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing::meat {

struct decision_graph {
  struct node {
    location_idx_t stop_id_;
    vector_map<dg_arc_2idx_t, dg_arc_idx_t> out_;
    vector_map<dg_arc_2idx_t, dg_arc_idx_t> in_;
  };

  struct arc {
    dg_node_idx_t dep_node_;
    dg_node_idx_t arr_node_;
    unixtime_t dep_time_;
    unixtime_t arr_time_;
    unixtime_t meat_;
    std::variant<journey::run_enter_exit, footpath> uses_;
    double use_prob_;
  };

  auto node_count() const { return nodes_.size(); }

  auto arc_count() const { return arcs_.size(); }

  void compute_use_probabilities(timetable const& tt,
                                 delta_t max_delay,
                                 bool g_is_sorted_by_dep_time = false);

private:
  void compute_use_probabilities_on_sorted_g(timetable const& tt,
                                             delta_t max_delay);
  void compute_use_probabilities_on_unsorted_g(timetable const& tt,
                                               delta_t max_delay);

public:
  vector_map<dg_node_idx_t, node> nodes_;
  vector_map<dg_arc_idx_t, arc> arcs_;
  dg_node_idx_t source_node_;
  dg_node_idx_t target_node_;
  dg_arc_idx_t first_arc_;
};

}  // namespace nigiri::routing::meat
