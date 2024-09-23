#include "nigiri/routing/meat/decision_graph.h"

#include <variant>

#include "nigiri/routing/meat/delay.h"

namespace nigiri::routing::meat {

void decision_graph::compute_use_probabilities_on_sorted_g(timetable const& tt,
                                                           delta_t max_delay) {
  assert(dg_arc_idx_t{0} == first_arc_);

  arcs_[first_arc_].use_prob_ = 1.0;
  for (auto const& in : arcs_) {
    double assigned_prob = 0.0;

    // find first out with dep_time_ >= in.arr_time_
    auto it_out = std::lower_bound(
        nodes_[in.arr_node_].out_.begin(), nodes_[in.arr_node_].out_.end(),
        in.arr_time_, [&](dg_arc_idx_t const& out, unixtime_t const& value) {
          return arcs_[out].dep_time_ < value;
        });
    for (; it_out != nodes_[in.arr_node_].out_.end(); ++it_out) {
      auto& out = arcs_[*it_out];

      double change_prob;

      auto const in_is_fp = std::holds_alternative<footpath>(in.uses_);
      auto const out_is_final_fp =
          std::holds_alternative<footpath>(out.uses_) &&
          out.arr_node_ == target_node_;
      if (in_is_fp || out_is_final_fp) {
        change_prob = 1;
      } else {
        change_prob = delay_prob(
            static_cast<delta_t>((out.dep_time_ - in.arr_time_).count()),
            tt.locations_.transfer_time_[nodes_[in.arr_node_].stop_id_].count(),
            max_delay);
      }

      arcs_[*it_out].use_prob_ += in.use_prob_ * (change_prob - assigned_prob);
      assigned_prob = change_prob;
      if (assigned_prob >= 1) {
        break;
      }
    }
    assert(assigned_prob == 1 ||
           (in.arr_node_ == target_node_ && assigned_prob == 0));
  }
}

void decision_graph::compute_use_probabilities_on_unsorted_g(
    timetable const& tt, delta_t max_delay) {
  auto ordered_arcs = std::vector<dg_arc_idx_t>(arc_count());
  for (auto i = 0U; i < arc_count(); ++i) {
    ordered_arcs[i] = dg_arc_idx_t{i};
  }

  auto ordered_nodes_out =
      std::vector<std::vector<dg_arc_idx_t>>(nodes_.size());
  for (auto i = dg_node_idx_t{0}; i < nodes_.size(); ++i) {
    ordered_nodes_out[to_idx(i)] =
        std::vector<dg_arc_idx_t>(nodes_[i].out_.size());
    for (auto j = dg_arc_2idx_t{0}; j < nodes_[i].out_.size(); ++j) {
      ordered_nodes_out[to_idx(i)][to_idx(j)] = nodes_[i].out_[j];
    }
  }

  std::sort(ordered_arcs.begin(), ordered_arcs.end(),
            [&](dg_arc_idx_t const l, dg_arc_idx_t const r) {
              return arcs_[l].dep_time_ < arcs_[r].dep_time_;
            });

  for (auto& out : ordered_nodes_out) {
    std::sort(out.begin(), out.end(),
              [&](dg_arc_idx_t const l, dg_arc_idx_t const r) {
                return arcs_[l].dep_time_ < arcs_[r].dep_time_;
              });
  }

  assert(ordered_arcs.front() == first_arc_);

  arcs_[first_arc_].use_prob_ = 1.0;
  for (auto const in_id : ordered_arcs) {
    auto const& in = arcs_[in_id];
    double assigned_prob = 0.0;

    // find first out with dep_time_ >= in.arr_time_
    auto it_out = std::lower_bound(
        ordered_nodes_out[to_idx(in.arr_node_)].begin(),
        ordered_nodes_out[to_idx(in.arr_node_)].end(), in.arr_time_,
        [&](dg_arc_idx_t const& out, unixtime_t const& value) {
          return arcs_[out].dep_time_ < value;
        });
    for (; it_out != ordered_nodes_out[to_idx(in.arr_node_)].end(); ++it_out) {
      auto& out = arcs_[*it_out];

      double change_prob;

      auto const in_is_fp = std::holds_alternative<footpath>(in.uses_);
      auto const out_is_final_fp =
          std::holds_alternative<footpath>(out.uses_) &&
          out.arr_node_ == target_node_;
      if (in_is_fp || out_is_final_fp) {
        change_prob = 1;
      } else {
        change_prob = delay_prob(
            static_cast<delta_t>((out.dep_time_ - in.arr_time_).count()),
            tt.locations_.transfer_time_[nodes_[in.arr_node_].stop_id_].count(),
            max_delay);
      }

      arcs_[*it_out].use_prob_ += in.use_prob_ * (change_prob - assigned_prob);
      assigned_prob = change_prob;
      if (assigned_prob >= 1) {
        break;
      }
    }
    assert(assigned_prob == 1 ||
           (in.arr_node_ == target_node_ && assigned_prob == 0));
  }
}

void decision_graph::compute_use_probabilities(timetable const& tt,
                                               delta_t max_delay,
                                               bool g_is_sorted_by_dep_time) {
  if (arc_count() == 0) return;
  if (g_is_sorted_by_dep_time) {
    compute_use_probabilities_on_sorted_g(tt, max_delay);
  } else {
    compute_use_probabilities_on_unsorted_g(tt, max_delay);
  }
}

}  // namespace nigiri::routing::meat
