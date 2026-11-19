#include "nigiri/routing/dijkstra.h"

#include "fmt/core.h"

#include "utl/get_or_create.h"

#include "nigiri/common/dial.h"
#include "nigiri/footpath.h"
#include "nigiri/for_each_meta.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/search_location.h"
#include "nigiri/rt/rt_timetable.h"

#define trace(...)
// #define trace(...) fmt::print(__VA_ARGS__)

namespace nigiri::routing {

void dijkstra(timetable const& tt,
              rt_timetable const* rtt,
              query const& q,
              direction const search_dir,
              std::vector<label::dist_t>& dists) {
  auto const fwd = search_dir == direction::kForward;
  dists.resize(tt.n_locations());
  utl::fill(dists, std::numeric_limits<label::dist_t>::max());

  std::map<location_idx_t, label::dist_t> min;
  auto const update_min = [&](location_idx_t const x, duration_t const d) {
    auto const l = tt.locations_.get_root_idx(x);
    auto& m = utl::get_or_create(min, l, [&]() { return dists[to_idx(l)]; });
    m = std::min(static_cast<label::dist_t>(d.count()), m);
  };

  for (auto const& start : q.destination_) {
    for_each_meta(
        tt, nullptr, q.dest_match_mode_, start.target_,
        [&](location_idx_t const x) { update_min(x, start.duration()); });
  }

  for (auto const& [from, td] : q.td_dest_) {
    if (to_idx(from) >= tt.n_locations()) {
      continue;
    }
    for (auto const& fp : td) {
      if (fp.duration_ != kTdFootpathBan && fp.duration_ < q.max_travel_time_) {
        update_min(from, fp.duration_);
      }
    }
  }

  auto pq = dial<label, get_bucket>{kMaxTravelTime.count()};
  for (auto const& [l, duration] : min) {
    auto const d = duration;
    for_each_meta(tt, nullptr, q.dest_match_mode_, l,
                  [&](location_idx_t const meta) {
                    pq.push(label{meta, d});
                    dists[to_idx(meta)] = std::min(d, dists[to_idx(meta)]);
                    trace("DIJKSTRA INIT @{}: {}\n", loc{tt, meta}, duration);
                  });
  }

  dijkstra(fwd ? tt.fwd_search_lb_graph_[q.prf_idx_]
               : tt.bwd_search_lb_graph_[q.prf_idx_],
           rtt == nullptr ? nullptr
                          : &(fwd ? rtt->fwd_search_lb_graph_has_edges_
                                  : rtt->bwd_search_lb_graph_has_edges_),
           rtt == nullptr
               ? nullptr
               : &(fwd ? rtt->fwd_search_lb_graph_ : rtt->bwd_search_lb_graph_),
           pq, dists);

  for (auto i = 0U; i != tt.n_locations(); ++i) {
    auto const lb = dists[i];
    for_each_meta(tt, nullptr, location_match_mode::kOnlyChildren,
                  location_idx_t{i}, [&](location_idx_t const c) {
                    dists[to_idx(c)] = std::min(lb, dists[to_idx(c)]);
                  });
  }

  if (rtt != nullptr) {
    extend_to_rt_virts(*rtt, dists);
  }
}

}  // namespace nigiri::routing
