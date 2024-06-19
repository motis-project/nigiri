#include "nigiri/routing/dijkstra.h"

#include "fmt/core.h"

#include "utl/get_or_create.h"

#include "nigiri/common/dial.h"
#include "nigiri/lb_entry.h"
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

template <typename Label, typename GetBucketFn>
void dijkstra(vecvec<location_idx_t, lb_entry> const& graph,
              dial<Label, GetBucketFn>& pq,
              transfer_time_settings const& tts,
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
      auto const new_dist = l.d_ + e.duration(tts).count();
      if (new_dist < dists[edge_target] && new_dist < pq.n_buckets() &&
          new_dist < max_dist) {
        dists[edge_target] = static_cast<dist_t>(new_dist);
        pq.push(Label{e.target(), static_cast<dist_t>(new_dist)});
      }
    }
  }
}

void dijkstra(timetable const& tt,
              query const& q,
              vecvec<location_idx_t, lb_entry> const& lb_graph,
              std::vector<label::dist_t>& dists) {
  dists.resize(tt.n_locations());
  utl::fill(dists, std::numeric_limits<label::dist_t>::max());

  std::map<location_idx_t, label::dist_t> min;
  for (auto const& start : q.destination_) {
    for_each_meta(
        tt, q.dest_match_mode_, start.target_, [&](location_idx_t const x) {
          auto const p = tt.locations_.parents_[x];
          auto const l = (p == location_idx_t::invalid()) ? x : p;
          auto& m =
              utl::get_or_create(min, l, [&]() { return dists[to_idx(l)]; });
          m = std::min(static_cast<label::dist_t>(start.duration().count()), m);
        });
  }

  auto pq = dial<label, get_bucket>{kMaxTravelTime.count()};
  for (auto const& [l, duration] : min) {
    auto const d = duration;
    for_each_meta(tt, q.start_match_mode_, l, [&](location_idx_t const meta) {
      pq.push(label{meta, d});
      dists[to_idx(meta)] = std::min(d, dists[to_idx(meta)]);
      trace("DIJKSTRA INIT @{}: {}\n", location{tt, meta}, duration);
    });
  }

  dijkstra(lb_graph, pq, q.transfer_time_settings_, dists);
}

}  // namespace nigiri::routing
