#pragma once

#include <ostream>
#include <vector>

#include "utl/overloaded.h"

#include "nigiri/location.h"
#include "nigiri/routing/journey.h"
#include "nigiri/routing/meat/csa/binary_search.h"
#include "nigiri/routing/meat/decision_graph.h"
#include "nigiri/rt/frun.h"

namespace nigiri::routing::meat {

struct expanded_representation {

  struct slot {
    size_t node_id_;
    unixtime_t when_;
  };

  struct extra_arc {
    size_t dep_slot_;
    size_t arr_slot_;
  };

  union time_slot {
    unixtime_t when_;
    size_t slot_;
  };

  expanded_representation() {}
  explicit expanded_representation(decision_graph const& g) {
    slots_of_node_.resize(g.node_count());

    for (auto& x : g.arcs_) {
      slots_of_node_[to_idx(x.dep_node_)].push_back(time_slot{x.dep_time_});
      slots_of_node_[to_idx(x.arr_node_)].push_back(time_slot{x.arr_time_});
    }

    for (auto& x : slots_of_node_) {
      std::sort(x.begin(), x.end(),
                [](auto l, auto r) { return l.when_ < r.when_; });
      x.erase(std::unique(x.begin(), x.end(),
                          [](auto l, auto r) { return l.when_ == r.when_; }),
              x.end());
    }

    for (auto i = 0U; i < g.node_count(); ++i) {
      for (auto& y : slots_of_node_[i]) {
        auto id = slots_.size();
        slots_.push_back({i, y.when_});
        y.slot_ = id;
      }
    }

    arcs_.resize(g.arc_count());

    auto find_slot = [&](dg_node_idx_t where, unixtime_t when) {
      return csa::binary_find_first_true(
                 slots_of_node_[to_idx(where)].begin(),
                 slots_of_node_[to_idx(where)].end(),
                 [&](auto const& s) { return when <= slots_[s.slot_].when_; })
          ->slot_;
    };

    for (auto i = dg_arc_idx_t{0}; i < g.arc_count(); ++i) {
      arcs_[to_idx(i)].dep_slot_ =
          find_slot(g.arcs_[i].dep_node_, g.arcs_[i].dep_time_);
      arcs_[to_idx(i)].arr_slot_ =
          find_slot(g.arcs_[i].arr_node_, g.arcs_[i].arr_time_);
    }
  }

  auto node_count() const { return slots_of_node_.size(); }

  auto slot_count() const { return slots_.size(); }

  auto arc_count() const { return arcs_.size(); }

  std::vector<slot> slots_;
  std::vector<std::vector<time_slot>> slots_of_node_;
  std::vector<extra_arc> arcs_;
};

inline void write_dot(std::ostream& out,
                      timetable const& tt,
                      decision_graph const& g,
                      expanded_representation const& r) {
  out << "digraph decision_graph{\n"
         "\tsplines=polyline;rankdir=LR;\n";

  for (auto i = dg_node_idx_t{0}; i < g.node_count(); ++i) {
    out << "\tnode" << i
        << "[shape=record,"
        // << "URL=\"javascript:onstop(" << i << "," << g.nodes_[i].stop_id_
        // << ")\","
        << "tooltip=\""
        << location(tt, g.nodes_[i].stop_id_)
        // << "\\nlocation_idx_t value="
        // << g.nodes_[i].stop_id_
        << "\\ntransfer time="
        << location(tt, g.nodes_[i].stop_id_).transfer_time_ << "\","
        << "label=\"" << location(tt, g.nodes_[i].stop_id_).name_;

    for (auto s : r.slots_of_node_[to_idx(i)]) {
      out << "|<slot" << s.slot_ << ">" << r.slots_[s.slot_].when_;
    }
    out << "\"];\n";
  }
  for (auto i = 0U; i < r.arc_count(); ++i) {
    out << "\tnode" << r.slots_[r.arcs_[i].dep_slot_].node_id_ << ":slot"
        << r.arcs_[i].dep_slot_ << " -> node"
        << r.slots_[r.arcs_[i].arr_slot_].node_id_ << ":slot"
        << r.arcs_[i].arr_slot_ << " [";
    std::visit(utl::overloaded{
                   [&](footpath const& fp) {
                     out << "label=\"walk\",tooltip=\"probability of use="
                         << g.arcs_[dg_arc_idx_t{i}].use_prob_
                         << "\\nMEAT=" << g.arcs_[dg_arc_idx_t{i}].meat_
                         << "\\nFOOTPATH (duration=" << fp.duration().count()
                         << ")\"";
                   },
                   [&](journey::run_enter_exit const& run) {
                     out << "label=\"" << tt.transport_name(run.r_.t_.t_idx_)
                         << "\",tooltip=\"probability of use="
                         << g.arcs_[dg_arc_idx_t{i}].use_prob_
                         << "\\nMEAT=" << g.arcs_[dg_arc_idx_t{i}].meat_
                         << "\\n";
                     auto const fr = rt::frun{tt, nullptr, run.r_};
                     for (auto j = run.stop_range_.from_;
                          j != run.stop_range_.to_; ++j) {
                       if (!fr[j].is_canceled()) {
                         fr[j].print(out, j == run.stop_range_.from_,
                                     j == run.stop_range_.to_ - 1U);
                         out << "\\n";
                       }
                     }
                     out << "\"";
                   }},
               g.arcs_[dg_arc_idx_t{i}].uses_);
    out << "];\n";
  }

  out << '}' << std::endl;
}
}  // namespace nigiri::routing::meat