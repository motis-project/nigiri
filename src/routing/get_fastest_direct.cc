#include "nigiri/routing/get_fastest_direct.h"

#include "utl/get_or_create.h"

#include "nigiri/common/dial.h"
#include "nigiri/routing/dijkstra.h"
#include "nigiri/routing/for_each_meta.h"
#include "nigiri/special_stations.h"

namespace nigiri::routing {

duration_t get_fastest_direct(timetable const& tt,
                              query const& q,
                              direction const dir,
                              label::dist_t const max_dist) {
  auto dists = hash_map<location_idx_t, label::dist_t>{};
  auto const get_dist = [&](location_idx_t const l) -> label::dist_t& {
    return utl::get_or_create(
        dists, l, []() { return std::numeric_limits<duration_t::rep>::max(); });
  };

  auto pq = dial<label, get_bucket>{};
  pq.n_buckets(max_dist);
  for (auto const& s : q.start_) {
    for_each_meta(
        tt, q.start_match_mode_, s.target_, [&](location_idx_t const start) {
          // auto& d = dists[start];
          auto& d = get_dist(start);
          d = std::min(static_cast<label::dist_t>(s.duration_.count()), d);
        });
  }
  for (auto const& [l, d] : dists) {
    pq.push(label{l, d});
  }

  auto dest_offsets = hash_map<location_idx_t, label::dist_t>{};
  for (auto const o : q.destination_) {
    auto const d = static_cast<label::dist_t>(o.duration().count());
    if (auto const it = dest_offsets.find(o.target_);
        it != end(dest_offsets) && it->second > d) {
      it->second = d;
    } else {
      dest_offsets.emplace(o.target(), d);
    }
  }

  auto end_dist = label::dist_t{std::numeric_limits<duration_t::rep>::max()};
  while (!pq.empty()) {
    auto l = pq.top();
    pq.pop();

    if (l.l_ == get_special_station(special_station::kEnd)) {
      break;
    }

    // if (dists[l.l_] < l.d_) {
    if (get_dist(l.l_) < l.d_) {
      continue;
    }

    auto const& footpaths =
        (dir == direction::kForward ? tt.locations_.footpaths_out_[q.prf_idx_]
                                    : tt.locations_.footpaths_in_[q.prf_idx_]);
    for (auto const& fp : footpaths[l.l_]) {
      auto const new_dist =
          l.d_ + static_cast<label::dist_t>(fp.duration().count());
      if (new_dist > max_dist) {
        continue;
      }

      if (!dest_offsets.contains(fp.target())) {
        continue;
      }

      // auto& target_dist = dists[fp.target()];
      auto& target_dist = get_dist(fp.target());
      if (new_dist < target_dist && new_dist < pq.n_buckets() &&
          new_dist < max_dist) {
        target_dist = static_cast<label::dist_t>(new_dist);
        pq.push(label{fp.target(), static_cast<label::dist_t>(new_dist)});
      }
    }

    if (auto const it = dest_offsets.find(l.l_); it != end(dest_offsets)) {
      auto const new_dist = l.d_ + static_cast<label::dist_t>(it->second);
      if (new_dist < max_dist && new_dist < end_dist) {
        end_dist = static_cast<label::dist_t>(new_dist);
      }
    }
  }

  return duration_t{end_dist};
}

}  // namespace nigiri::routing
