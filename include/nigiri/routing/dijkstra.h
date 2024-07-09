#pragma once

#include "fmt/ranges.h"

#include "nigiri/common/dial.h"
#include "nigiri/footpath.h"
#include "nigiri/types.h"

namespace nigiri {
struct timetable;
}  // namespace nigiri

namespace nigiri::routing {

struct label {
  using dist_t = std::uint16_t;
  label(location_idx_t const l, dist_t const d) : l_{l}, d_{d} {}
  friend bool operator>(label const& a, label const& b) { return a.d_ > b.d_; }
  location_idx_t l_;
  dist_t d_;
};

inline auto format_as(label const& l) { return std::tuple{l.l_, l.d_}; }

struct get_bucket {
  label::dist_t operator()(label const& l) const { return l.d_; }
};

struct query;

template <typename NodeIdx, typename Edge, typename Label, typename GetBucketFn>
void dijkstra(vecvec<NodeIdx, Edge, std::uint64_t> const& graph,
              dial<Label, GetBucketFn>& pq,
              std::vector<typename Label::dist_t>& dists,
              typename Label::dist_t const max_dist =
                  std::numeric_limits<typename Label::dist_t>::max()) {
  using dist_t = typename Label::dist_t;

  while (!pq.empty()) {
    auto l = pq.top();
    pq.pop();

    if (dists[cista::to_idx(l.l_)] < l.d_) {
      continue;
    }

    for (auto const& e : graph[l.l_]) {
      auto const edge_target = cista::to_idx(e.target());
      auto const new_dist = l.d_ + e.duration().count();
      if (new_dist < dists[edge_target] && new_dist < pq.n_buckets() &&
          new_dist < max_dist) {
        dists[edge_target] = static_cast<dist_t>(new_dist);
        pq.push(Label{e.target(), static_cast<dist_t>(new_dist)});
      }
    }
  }
}

void dijkstra(timetable const&,
              query const&,
              vecvec<location_idx_t, footpath> const& lb_graph,
              std::vector<std::uint16_t>& dists);

}  // namespace nigiri::routing
