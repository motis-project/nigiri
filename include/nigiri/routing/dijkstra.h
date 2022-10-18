#pragma once

#include "nigiri/common/dial.h"
#include "nigiri/footpath.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/query.h"
#include "nigiri/types.h"

namespace nigiri::routing {

using dist_t = duration_t;

struct label {
  label(location_idx_t const l, duration_t const d) : l_{l}, d_{d} {}
  friend bool operator>(label const& a, label const& b) { return a.d_ > b.d_; }
  location_idx_t l_;
  dist_t d_;
};

struct get_bucket {
  std::size_t operator()(label const& l) const { return l.d_.count(); }
};

void dijkstra(vecvec<location_idx_t, footpath> const& lb_graph,
              std::vector<dist_t>& dists,
              std::vector<offset> const& starts) {
  assert(dists.size() == lb_graph.size());
  std::fill(begin(dists), end(dists),
            duration_t{std::numeric_limits<duration_t::rep>::max()});

  dial<label, kMaxTravelTime, get_bucket> pq;
  for (auto const& start : starts) {
    pq.push(label{start.location_, start.offset_});
  }
  while (!pq.empty()) {
    auto l = pq.top();
    pq.pop();

    for (auto const& e : lb_graph[l.l_]) {
      auto const new_dist = l.d_ + e.duration_;
      if (new_dist < dists[to_idx(e.target_)] &&
          new_dist <= duration_t{kMaxTravelTime}) {
        dists[to_idx(e.target_)] = new_dist;
        pq.push(label(e.target_, new_dist));
      }
    }
  }
}

}  // namespace nigiri::routing
