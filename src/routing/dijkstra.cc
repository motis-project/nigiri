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

void dijkstra(timetable const& tt,
              query const& q,
              vecvec<location_idx_t, footpath> const& lb_graph,
              std::vector<label::dist_t>& dists) {
  dists.resize(tt.n_locations());
  utl::fill(dists, std::numeric_limits<label::dist_t>::max());

  std::map<location_idx_t, label::dist_t> min;
  auto const update_min = [&](location_idx_t const x, duration_t const d) {
    auto const p = tt.locations_.parents_[x];
    auto const l = (p == location_idx_t::invalid()) ? x : p;
    auto& m = utl::get_or_create(min, l, [&]() { return dists[to_idx(l)]; });
    m = std::min(static_cast<label::dist_t>(d.count()), m);
  };

  for (auto const& start : q.destination_) {
    for_each_meta(
        tt, q.dest_match_mode_, start.target_,
        [&](location_idx_t const x) { update_min(x, start.duration()); });
  }

  for (auto const& [from, td] : q.td_dest_) {
    for (auto const& fp : td) {
      if (fp.duration_ != footpath::kMaxDuration &&
          fp.duration_ < kMaxTravelTime) {
        update_min(from, fp.duration_);
      }
    }
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

  dijkstra(lb_graph, pq, dists);
}

}  // namespace nigiri::routing
