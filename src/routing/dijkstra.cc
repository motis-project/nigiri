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

using travel_time_dist_t = std::uint16_t;

struct travel_time_label {
  travel_time_label(location_idx_t const l, travel_time_dist_t const d)
      : l_{l}, d_{d} {}

  friend bool operator>(travel_time_label const& a,
                        travel_time_label const& b) {
    return a.d_ > b.d_;
  }

  location_idx_t l_;
  travel_time_dist_t d_;
};

void dijkstra(timetable const& tt,
              query const& q,
              vecvec<location_idx_t, footpath> const& lb_graph,
              std::vector<std::uint16_t>& dists) {
  struct get_bucket {
    std::size_t operator()(travel_time_label const& l) const {
      return static_cast<std::size_t>(l.d_);
    }
  };

  dists.resize(tt.n_locations());
  utl::fill(dists, std::numeric_limits<travel_time_dist_t>::max());

  std::map<location_idx_t, travel_time_dist_t> min;
  for (auto const& start : q.destination_) {
    for_each_meta(
        tt, q.dest_match_mode_, start.target_, [&](location_idx_t const x) {
          auto const p = tt.locations_.parents_[x];
          auto const l = (p == location_idx_t::invalid()) ? x : p;
          auto& m =
              utl::get_or_create(min, l, [&]() { return dists[to_idx(l)]; });
          m = std::min(
              static_cast<travel_time_dist_t>(start.duration().count()), m);
        });
  }

  dial<travel_time_label, kMaxTravelTime.count(), get_bucket> pq;
  for (auto const& [l, duration] : min) {
    auto const d = duration;
    for_each_meta(tt, q.start_match_mode_, l, [&](location_idx_t const meta) {
      pq.push(travel_time_label{meta, d});
      dists[to_idx(meta)] = std::min(d, dists[to_idx(meta)]);
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
      auto const new_dist = l.d_ + e.duration().count();
      if (new_dist < dists[to_idx(e.target())] &&
          new_dist <= kMaxTravelTime.count()) {
        dists[to_idx(e.target())] = static_cast<travel_time_dist_t>(new_dist);
        pq.push(travel_time_label(e.target(),
                                  static_cast<travel_time_dist_t>(new_dist)));
      }
    }
  }
}

using transfers_dist_t = std::uint8_t;

struct transfers_label {
  transfers_label(component_idx_t const c, transfers_dist_t const d)
      : c_{c}, d_{d} {}

  friend bool operator>(transfers_label const& a, transfers_label const& b) {
    return a.d_ > b.d_;
  }

  component_idx_t c_;
  transfers_dist_t d_;
};

void dijkstra(timetable const& tt,
              query const& q,
              vecvec<component_idx_t, component_idx_t> const& lb_graph,
              std::vector<transfers_dist_t>& dists) {
  struct get_bucket {
    std::size_t operator()(transfers_label const& l) const {
      return static_cast<std::size_t>(l.d_);
    }
  };

  dists.resize(tt.n_locations());
  utl::fill(dists, std::numeric_limits<transfers_dist_t>::max());

  std::map<location_idx_t, transfers_dist_t> min;
  for (auto const& start : q.destination_) {
    for_each_meta(
        tt, q.dest_match_mode_, start.target_, [&](location_idx_t const x) {
          auto const p = tt.locations_.parents_[x];
          auto const l = (p == location_idx_t::invalid()) ? x : p;
          auto& m =
              utl::get_or_create(min, l, [&]() { return dists[to_idx(l)]; });
          m = std::min(static_cast<transfers_dist_t>(start.duration().count()),
                       m);
        });
  }

  dial<transfers_label, kMaxTransfers, get_bucket> pq;
  for (auto const& [l, duration] : min) {
    auto const d = duration;
    for_each_meta(tt, q.start_match_mode_, l, [&](location_idx_t const meta) {
      pq.push(transfers_label{tt.locations_.components_[meta], d});
      dists[to_idx(meta)] = std::min(d, dists[to_idx(meta)]);
      trace("DIJKSTRA INIT @{}: {}\n", location{tt, meta}, duration);
    });
  }

  while (!pq.empty()) {
    auto l = pq.top();
    pq.pop();

    if (dists[to_idx(l.c_)] < l.d_) {
      continue;
    }

    for (auto const& target : lb_graph[l.c_]) {
      auto const new_dist = l.d_ + 1U;
      if (new_dist < dists[to_idx(target)] &&
          new_dist <= kMaxTravelTime.count()) {
        dists[to_idx(target)] = static_cast<transfers_dist_t>(new_dist);
        pq.push(
            transfers_label(target, static_cast<transfers_dist_t>(new_dist)));
      }
    }
  }
}

}  // namespace nigiri::routing
