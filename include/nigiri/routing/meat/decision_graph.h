#pragma once

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/journey.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

namespace nigiri::routing::meat {

struct decision_graph {
  struct node {
    friend bool semantically_equal(node const& a, node const& b) {
      auto const a_out = a.out_.size();
      auto const a_in = a.in_.size();
      auto const b_out = b.out_.size();
      auto const b_in = b.in_.size();
      return std::tie(a.stop_id_, a_out, a_in) ==
             std::tie(b.stop_id_, b_out, b_in);
    }

    location_idx_t stop_id_;
    vector_map<dg_arc_2idx_t, dg_arc_idx_t> out_;
    vector_map<dg_arc_2idx_t, dg_arc_idx_t> in_;
  };

  struct arc {
    friend bool semantically_equal(arc const& a, arc const& b) {
      return std::tie(a.dep_time_, a.arr_time_, a.meat_, a.uses_,
                      a.use_prob_) ==
             std::tie(b.dep_time_, b.arr_time_, b.meat_, b.uses_, b.use_prob_);
    }

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
  friend bool semantically_equal(decision_graph const& a,
                                 decision_graph const& b);

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

// assumes that there are no arcs a,b with a == b in node.in_/node.out_
inline bool semantically_equal(decision_graph const& a,
                               decision_graph const& b) {
  auto possible_same =
      a.nodes_.size() == b.nodes_.size() && a.arcs_.size() == b.arcs_.size() &&
      semantically_equal(a.nodes_[a.source_node_], b.nodes_[b.source_node_]) &&
      semantically_equal(a.nodes_[a.target_node_], b.nodes_[b.target_node_]);
  if (!possible_same) {
    return false;
  }
  auto a_n2b_n = vector_map<dg_node_idx_t, dg_node_idx_t>(
      a.nodes_.size(), dg_node_idx_t::invalid());
  a_n2b_n[a.source_node_] = b.source_node_;
  auto added_to_stack = bitvec{a.nodes_.size()};
  auto node_to_it = std::stack<dg_node_idx_t>{};
  node_to_it.push(a.source_node_);
  added_to_stack.set(a.source_node_.v_);
  while (!node_to_it.empty()) {
    auto const& node_a = a.nodes_[node_to_it.top()];
    auto const& node_b = b.nodes_[a_n2b_n[node_to_it.top()]];
    node_to_it.pop();
    if (!semantically_equal(node_a, node_b)) {
      return false;
    }

    for (auto const& arc_in_idx : node_a.in_) {
      auto const& arc_in_a = a.arcs_[arc_in_idx];
      auto const it =
          std::find_if(node_b.in_.begin(), node_b.in_.end(), [&](auto idx) {
            return semantically_equal(arc_in_a, b.arcs_[idx]) &&
                   semantically_equal(a.nodes_[arc_in_a.dep_node_],
                                      b.nodes_[b.arcs_[idx].dep_node_]) &&
                   semantically_equal(a.nodes_[arc_in_a.arr_node_],
                                      b.nodes_[b.arcs_[idx].arr_node_]);
          });
      if (it == node_b.in_.end()) {
        return false;
      }
    }

    for (auto const& arc_out_idx : node_a.out_) {
      auto const& arc_out_a = a.arcs_[arc_out_idx];
      auto const it =
          std::find_if(node_b.out_.begin(), node_b.out_.end(), [&](auto idx) {
            return semantically_equal(arc_out_a, b.arcs_[idx]) &&
                   semantically_equal(a.nodes_[arc_out_a.dep_node_],
                                      b.nodes_[b.arcs_[idx].dep_node_]) &&
                   semantically_equal(a.nodes_[arc_out_a.arr_node_],
                                      b.nodes_[b.arcs_[idx].arr_node_]);
          });
      if (it == node_b.in_.end()) {
        return false;
      }

      if (!added_to_stack[arc_out_a.arr_node_.v_]) {
        node_to_it.push(arc_out_a.arr_node_);
        added_to_stack.set(arc_out_a.arr_node_.v_);
        a_n2b_n[arc_out_a.arr_node_] = b.arcs_[*it].arr_node_;
      }
    }
  }
  return true;
}

inline std::ostream& operator<<(std::ostream& out,
                                decision_graph::node const& n) {
  out << "stop_id_: " << n.stop_id_ << "  in: ";
  for (auto const& a_idx : n.in_) {
    out << a_idx << " ";
  }
  out << "out: ";
  for (auto const& a_idx : n.out_) {
    out << a_idx << " ";
  }
  return out;
}
inline std::ostream& operator<<(std::ostream& out,
                                decision_graph::arc const& a) {
  out << "dep_node_: " << a.dep_node_ << " arr_node_: " << a.arr_node_
      << " dep_time_: " << a.dep_time_ << " arr_time_: " << a.arr_time_
      << " meat: " << a.meat_ << " use_prob_: " << a.use_prob_;
  std::visit(utl::overloaded{[&](footpath const& fp) {
                               out << " uses_fp: target_: " << fp.target_
                                   << " duration_:" << fp.duration_;
                             },
                             [&](journey::run_enter_exit const& r) {
                               out << " uses_run: transport:" << r.r_.t_
                                   << " range:" << r.stop_range_.from_ << " - "
                                   << r.stop_range_.to_;
                             }},
             a.uses_);
  return out;
}
inline std::ostream& operator<<(std::ostream& out, decision_graph const& dg) {
  out << "source_node_: " << dg.source_node_ << std::endl
      << "target_node_: " << dg.target_node_ << std::endl
      << "first_arc_:   " << dg.first_arc_ << std::endl
      << "nodes_:" << std::endl;
  auto idx = 0U;
  for (auto const& n : dg.nodes_) {
    out << idx++ << ": " << n << std::endl;
  }
  out << std::endl << "arcs_:" << std::endl;
  idx = 0U;
  for (auto const& a : dg.arcs_) {
    out << idx++ << ": " << a << std::endl;
  }
  out << std::endl << std::endl;
  return out;
}

}  // namespace nigiri::routing::meat
