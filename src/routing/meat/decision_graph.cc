#include "nigiri/routing/meat/decision_graph.h"

#include <variant>

#include "nigiri/routing/meat/delay.h"

namespace nigiri::routing::meat {

std::vector<double> compute_reach_probabilities(timetable const& tt,
                                                decision_graph const& g,
                                                delta_t max_delay) {
  std::vector<double> p(g.arc_count(), 0.0);

  if (g.arc_count() == 0) return p;

  std::vector<int> ordered_arcs(g.arc_count());
  for (int i = 0; i < g.arc_count(); ++i) ordered_arcs[i] = i;

  std::sort(ordered_arcs.begin(), ordered_arcs.end(), [&](int l, int r) {
    return g.arcs_[l].dep_time_ < g.arcs_[r].dep_time_;
  });

  assert(ordered_arcs.front() == g.first_arc_);

  p[g.first_arc_] = 1.0;
  for (auto in_id : ordered_arcs) {
    auto& in = g.arcs_[in_id];
    double assigned_prob = 0.0;

    for (auto out_id : g.nodes_[in.arr_node_].out_) {
      auto& out = g.arcs_[out_id];

      double change_prob;

      if (std::holds_alternative<footpath>(in.uses_)) {
        change_prob = in.arr_time_ < out.dep_time_ ? 0 : 1;
      } else {
        change_prob = delay_prob(
            (out.dep_time_ - in.arr_time_).count(),
            tt.locations_.transfer_time_[g.nodes_[in.arr_node_].stop_id_]
                .count(),
            max_delay);  // ??? Zweiter Parameter ist im Original
                         // falsch! es wird keine umstigeszeit
                         // sondern eine id übergeben TODO: Kommentar entfernen
      }

      p[out_id] += p[in_id] * (change_prob - assigned_prob);
      assigned_prob = change_prob;
    }
  }
  return p;
}

}  // namespace nigiri::routing::meat
