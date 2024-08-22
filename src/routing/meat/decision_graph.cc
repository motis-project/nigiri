#include "nigiri/routing/meat/decision_graph.h"

#include <variant>

#include "nigiri/routing/meat/delay.h"

namespace nigiri::routing::meat {

void decision_graph::compute_use_probabilities(timetable const& tt,
                                                delta_t max_delay) {
  if (arc_count() == 0) return;

  std::vector<int> ordered_arcs(arc_count());
  for (int i = 0; i < arc_count(); ++i) ordered_arcs[i] = i;

  std::sort(ordered_arcs.begin(), ordered_arcs.end(), [&](int l, int r) {
    return arcs_[l].dep_time_ < arcs_[r].dep_time_;
  });

  assert(ordered_arcs.front() == first_arc_);

  arcs_[first_arc_].use_prob_ = 1.0;
  for (auto in_id : ordered_arcs) {
    auto& in = arcs_[in_id];
    double assigned_prob = 0.0;

    for (auto out_id : nodes_[in.arr_node_].out_) {
      auto& out = arcs_[out_id];

      double change_prob;

      if (std::holds_alternative<footpath>(in.uses_)) {
        change_prob = in.arr_time_ < out.dep_time_ ? 0 : 1;
      } else {
        change_prob = delay_prob(
            (out.dep_time_ - in.arr_time_).count(),
            tt.locations_.transfer_time_[nodes_[in.arr_node_].stop_id_]
                .count(),
            max_delay);  // ??? Zweiter Parameter ist im Original
                         // falsch! es wird keine umstigeszeit
                         // sondern eine id übergeben TODO: Kommentar entfernen
      }

      arcs_[out_id].use_prob_ += arcs_[in_id].use_prob_ * (change_prob - assigned_prob);
      assigned_prob = change_prob;
    }
  }
  return;
}

}  // namespace nigiri::routing::meat
