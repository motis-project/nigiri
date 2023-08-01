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
              std::vector<lower_bound>& dists) {
  struct get_bucket {
    std::size_t operator()(travel_time_label const& l) const {
      return static_cast<std::size_t>(l.d_);
    }
  };

  dists.resize(tt.n_locations());
  for (auto& d : dists) {
    d.travel_time_ = lower_bound::kTravelTimeUnreachable;
  }

  std::map<location_idx_t, travel_time_dist_t> min;
  for (auto const& start : q.destination_) {
    for_each_meta(
        tt, q.dest_match_mode_, start.target_, [&](location_idx_t const x) {
          auto const p = tt.locations_.parents_[x];
          auto const l = (p == location_idx_t::invalid()) ? x : p;
          auto& m = utl::get_or_create(
              min, l, [&]() { return dists[to_idx(l)].travel_time_; });
          m = std::min(
              static_cast<travel_time_dist_t>(start.duration().count()), m);
        });
  }

  dial<travel_time_label, kMaxTravelTime.count(), get_bucket> pq;
  for (auto const& [l, duration] : min) {
    auto const d = duration;
    for_each_meta(tt, q.start_match_mode_, l, [&](location_idx_t const meta) {
      pq.push(travel_time_label{meta, d});
      dists[to_idx(meta)].travel_time_ =
          std::min(d, dists[to_idx(meta)].travel_time_);
      trace("TRAVEL TIME DIJKSTRA INIT @{}: {}\n", location{tt, meta},
            duration);
    });
  }

  while (!pq.empty()) {
    auto l = pq.top();
    pq.pop();

    if (dists[to_idx(l.l_)].travel_time_ < l.d_) {
      continue;
    }

    for (auto const& e : lb_graph[l.l_]) {
      auto const new_dist = l.d_ + e.duration().count();
      if (new_dist < dists[to_idx(e.target())].travel_time_ &&
          new_dist <= kMaxTravelTime.count()) {
        dists[to_idx(e.target())].travel_time_ =
            static_cast<travel_time_dist_t>(new_dist);
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
              std::vector<lower_bound>& lb) {
  struct get_bucket {
    std::size_t operator()(transfers_label const& l) const {
      return static_cast<std::size_t>(l.d_);
    }
  };

  constexpr auto const kLimit = (kMaxTransfers + 1U) * 2U;
  constexpr auto const kUnreachable = std::numeric_limits<std::uint8_t>::max();

  auto dists = std::vector<transfers_dist_t>{};
  dists.resize(to_idx(tt.locations_.next_component_idx_) + tt.n_routes());
  utl::fill(dists, kUnreachable);

  auto pq = dial<transfers_label, kLimit, get_bucket>{};
  for (auto const& start : q.destination_) {
    for_each_meta(tt, q.dest_match_mode_, start.target_,
                  [&](location_idx_t const x) {
                    auto const c = tt.locations_.components_[x];
                    pq.push(transfers_label{c, 0U});
                    dists[to_idx(c)] = 0U;
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
      if (new_dist < dists[to_idx(target)] && new_dist <= kLimit) {
        dists[to_idx(target)] = static_cast<transfers_dist_t>(new_dist);
        pq.push(
            transfers_label(target, static_cast<transfers_dist_t>(new_dist)));
      }
    }
  }

  for (auto& dist : dists) {
    if (dist != kUnreachable) {
      dist = (dist == 0U) ? 0U : (dist / 2U) - 1U;
    }
  }

  lb.resize(tt.n_locations());
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    auto const d = dists[to_idx(tt.locations_.components_[l])];
    lb[to_idx(l)].transfers_ =
        d == kUnreachable ? lower_bound::kTransfersUnreachable : d;
  }
}

}  // namespace nigiri::routing
