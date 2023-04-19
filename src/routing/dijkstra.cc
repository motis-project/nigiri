#include "nigiri/routing/dijkstra.h"

#include "fmt/core.h"

#include "utl/get_or_create.h"

#include "nigiri/common/dial.h"
#include "nigiri/footpath.h"
#include "nigiri/routing/for_each_meta.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/query.h"

// #define NIGIRI_DIJKSTRA_TRACING

#ifdef NIGIRI_DIJKSTRA_TRACING
#define trace(...) fmt::print(__VA_ARGS__)
#else
#define trace(...)
#endif

namespace nigiri::routing {

using dist_t = duration_t;

struct label {
  label(location_idx_t const l, duration_t const d) : l_{l}, d_{d} {}
  friend bool operator>(label const& a, label const& b) { return a.d_ > b.d_; }
  location_idx_t l_;
  dist_t d_;
};

struct get_bucket {
  std::size_t operator()(label const& l) const {
    return static_cast<std::size_t>(l.d_.count());
  }
};

void dijkstra(timetable const& tt,
              query const& q,
              vecvec<location_idx_t, footpath> const& lb_graph,
              std::vector<dist_t>& dists) {
  assert(dists.size() == tt.n_locations());
  std::fill(begin(dists), end(dists),
            duration_t{std::numeric_limits<duration_t::rep>::max()});

  std::map<location_idx_t, duration_t> min;
  for (auto const& start : q.destinations_.front()) {
    auto const p = tt.locations_.parents_[start.target_];
    auto const l = (p == location_idx_t::invalid()) ? start.target_ : p;
    auto& m = utl::get_or_create(min, l, [&]() { return dists[to_idx(l)]; });
    m = std::min(start.duration_, m);
  }

  dial<label, kMaxTravelTime, get_bucket> pq;
  for (auto const& [l, duration] : min) {
    for_each_meta(tt, q.start_match_mode_, l, [&](location_idx_t const meta) {
      pq.push(label{meta, duration});
      dists[to_idx(meta)] = std::min(duration, dists[to_idx(meta)]);
      trace("DIJKSTRA INIT @{}: {}\n", location{tt, meta}, duration);
    });
  }

  while (!pq.empty()) {
    auto l = pq.top();
    pq.pop();

    if (dists[to_idx(l.l_)] < l.d_) {
      continue;
    }

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
