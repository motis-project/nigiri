/*
 * This code is based on the original version by Ben Strasser
 */

#pragma once

#include <ostream>
#include <vector>

#include "utl/enumerate.h"
#include "utl/overloaded.h"

#include "nigiri/location.h"
#include "nigiri/routing/meat/decision_graph.h"
#include "nigiri/rt/frun.h"
#include "nigiri/types.h"

namespace nigiri::routing::meat {

struct compact_representation {

  struct out_arrow {
    dg_node_idx_t arr_node_;
    unixtime_t min_time_;
    unixtime_t max_time_;
    std::vector<dg_arc_idx_t> dg_arcs_;
  };

  size_t node_count() const { return out_.size(); }

  int arrow_count() const {
    int n = 0;
    for (auto& x : out_) n += x.size();
    return n;
  }
  compact_representation() {}
  explicit compact_representation(decision_graph const& g) {
    out_.resize(g.node_count());

    for (auto const [i, x] : utl::enumerate(g.arcs_)) {
      out_[to_idx(x.dep_node_)].emplace_back(
          x.arr_node_, x.dep_time_, x.dep_time_,
          std::vector<dg_arc_idx_t>(1, dg_arc_idx_t{i}));
    }

    for (auto& x : out_) {
      if (x.empty()) {
        continue;
      }
      auto begin = x.begin(), end = x.end();
      std::sort(begin, end, [](out_arrow l, out_arrow r) {
        return l.min_time_ < r.min_time_;
      });

      auto in = begin, out = begin;

      *out = *in;
      ++in;
      while (in != end) {
        if (in->arr_node_ == out->arr_node_) {
          out->max_time_ = in->min_time_;
          for (auto const i : in->dg_arcs_) {
            out->dg_arcs_.emplace_back(i);
          }
        } else {
          ++out;
          *out = *in;
        }
        ++in;
      }
      ++out;
      x.erase(out, end);
    }
  }

  std::vector<std::vector<out_arrow>> out_;
};

inline void write_dot(std::ostream& out,
                      timetable const& tt,
                      decision_graph const& g,
                      compact_representation const& r) {
  out << "digraph decision_graph{\n"
         "\tsplines=polyline;rankdir=LR;\n";

  for (auto i = dg_node_idx_t{0}; i < g.node_count(); ++i) {
    out << "\tnode" << i
        << "[shape=record,"
        // << "URL=\"javascript:onstop(" << i << "," << g.nodes_[i].stop_id_
        // << ")\","
        << "tooltip=\"" << location(tt, g.nodes_[i].stop_id_)
        << "\\ntransfer time="
        << location(tt, g.nodes_[i].stop_id_).transfer_time_ << "\","
        << "label=\"" << location(tt, g.nodes_[i].stop_id_).name_;

    // if (r.out_[i].size() > 1) {
    for (auto j = 0U; j < r.out_[to_idx(i)].size(); ++j) {
      out << "|<slot" << j << ">" << r.out_[to_idx(i)][j].min_time_;
      if (r.out_[to_idx(i)][j].min_time_ != r.out_[to_idx(i)][j].max_time_) {
        out << " - " << r.out_[to_idx(i)][j].max_time_;
      }
    }
    //}
    out << "\"];\n";
  }

  for (auto i = 0U; i < g.node_count(); ++i) {
    // if (r.out_[i].size() > 1) {
    for (auto j = 0U; j < r.out_[i].size(); ++j) {
      out << "\tnode" << i << ":slot" << j << " -> node"
          << r.out_[i][j].arr_node_ << " [tooltip=\"";
      for (auto const dg_arc : r.out_[i][j].dg_arcs_) {
        std::visit(
            utl::overloaded{
                [&](footpath const& fp) {
                  out << "probability of use=" << g.arcs_[dg_arc].use_prob_
                      << "\\nMEAT=" << g.arcs_[dg_arc].meat_
                      << "\\nFOOTPATH (duration=" << fp.duration().count()
                      << ")\\n\\n";
                },
                [&](journey::run_enter_exit const& run) {
                  out << "probability of use=" << g.arcs_[dg_arc].use_prob_
                      << "\\nMEAT=" << g.arcs_[dg_arc].meat_ << "\\n";
                  auto const fr = rt::frun{tt, nullptr, run.r_};
                  for (auto k = run.stop_range_.from_; k != run.stop_range_.to_;
                       ++k) {
                    if (!fr[k].is_canceled()) {
                      fr[k].print(out, k == run.stop_range_.from_,
                                  k == run.stop_range_.to_ - 1U);
                      out << "\\n";
                    }
                  }
                  out << "\\n\\n";
                }},
            g.arcs_[dg_arc].uses_);
      }
      out << "\"];\n";
    }
    //} else if (r.out_[i].size() == 1) {
    //  out << "\tnode" << i << " -> node" << r.out_[i][0].arr_node_ << ";\n";
    //}
  }

  out << '}' << std::endl;
}

}  // namespace nigiri::routing::meat
