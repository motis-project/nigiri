#include "nigiri/loader/build_footpaths.h"

#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>

#include "geo/latlng.h"

#include "utl/helpers/algorithm.h"
#include "utl/parallel_for.h"

#include "nigiri/loader/link_nearby_stations.h"
#include "nigiri/loader/merge_duplicates.h"
#include "nigiri/loader/virtual_locations.h"
#include "nigiri/common/dial.h"
#include "nigiri/constants.h"
#include "nigiri/logging.h"
#include "nigiri/routing/dijkstra.h"
#include "nigiri/types.h"

namespace nigiri::loader {

duration_t max_with_transfer_times(timetable const& tt,
                                   location_idx_t const from,
                                   location_idx_t const to,
                                   duration_t const walk) {
  auto const transfer_time = [&](location_idx_t const l) {
    auto const t = tt.locations_.transfer_time_[l];
    return t == kNoTransferAllowed ? duration_t{0} : duration_t{t};
  };
  return std::max({walk, transfer_time(from), transfer_time(to)});
}

void add_if_not_exists(auto bucket, footpath const fp) {
  if (utl::none_of(bucket, [&](footpath const x) {
        return x.target() == fp.target();
      })) {
    bucket.push_back(fp);
  }
}

void add_equivalence_footpaths(
    timetable const& tt,
    mutable_fws_multimap<location_idx_t, footpath>& walks,
    std::uint16_t const max_footpath_length) {
  auto const max = std::min<int>(max_footpath_length, footpath::kMax.count());
  auto const add = [&](location_idx_t const a, location_idx_t const b) {
    auto const& pos = tt.locations_.coordinates_[a];
    auto const dist = std::sqrt(
        geo::approx_squared_distance(pos, tt.locations_.coordinates_[b],
                                     geo::approx_distance_lng_degrees(pos)));
    auto const minutes = std::max(2.0, std::ceil((dist / kWalkSpeed) / 60.0));
    if (minutes <= max) {
      add_if_not_exists(walks[a],
                        {b, duration_t{static_cast<duration_t::rep>(minutes)}});
    }
  };

  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    for (auto const eq : tt.locations_.equivalences_[l]) {
      if (eq != l) {
        add(l, eq);
        add(eq, l);
      }
    }
  }
}

// A footpath is one change: a walk A -> B -> C needs its own footpath. Adds
// the chains of the walks so far up to max_footpath_length walking minutes -
// transfer times come afterwards (walking through a stop is no change there).
void add_transitive_walks(timetable const& tt,
                          mutable_fws_multimap<location_idx_t, footpath>& fps,
                          std::uint16_t const max_footpath_length) {
  constexpr auto const kUnreached = std::numeric_limits<std::uint16_t>::max();
  struct search {
    std::vector<std::uint16_t> dist_;
    std::vector<location_idx_t> reached_;
    dial<routing::label, routing::get_bucket> pq_;
  };

  auto const n = tt.n_locations();
  auto const max = static_cast<std::uint16_t>(
      std::min<unsigned>(max_footpath_length, footpath::kMax.count()));
  auto chains = std::vector<std::vector<footpath>>(n);
  utl::parallel_for_run_threadlocal<search>(
      n, [&](search& s, std::size_t const i) {
        auto const from = location_idx_t{i};
        if (fps[from].empty()) {
          return;
        }
        s.dist_.resize(n, kUnreached);
        s.pq_.n_buckets(static_cast<std::uint16_t>(max + 1U));
        s.pq_.clear();  // a default constructed dial is uninitialized
        s.dist_[i] = 0U;
        s.reached_.push_back(from);
        s.pq_.push(routing::label{from, 0U});
        while (!s.pq_.empty()) {
          auto const l = s.pq_.top();
          s.pq_.pop();
          if (s.dist_[to_idx(l.l_)] < l.d_) {
            continue;
          }
          for (auto const fp : fps[l.l_]) {
            auto const d = l.d_ + fp.duration().count();
            auto& dist = s.dist_[to_idx(fp.target())];
            if (d <= max && d < dist) {
              if (dist == kUnreached) {
                s.reached_.push_back(fp.target());
              }
              dist = static_cast<std::uint16_t>(d);
              s.pq_.push(routing::label{fp.target(), dist});
            }
          }
        }
        for (auto const l : s.reached_) {
          if (l != from) {
            chains[i].emplace_back(l, duration_t{s.dist_[to_idx(l)]});
          }
          s.dist_[to_idx(l)] = kUnreached;
        }
        s.reached_.clear();
      });

  // Keep the shortest walk per pair.
  for (auto l = location_idx_t{0U}; l != n; ++l) {
    auto bucket = fps[l];
    for (auto const c : chains[to_idx(l)]) {
      auto const it = utl::find_if(
          bucket, [&](footpath const x) { return x.target() == c.target(); });
      if (it == end(bucket)) {
        bucket.push_back(c);
      } else if (c.duration() < it->duration()) {
        *it = c;
      }
    }
  }
}

void build_footpaths(timetable& tt,
                     finalize_options const opt,
                     shapes_storage* shapes) {
  auto walks = mutable_fws_multimap<location_idx_t, footpath>{};
  walks[location_idx_t{tt.n_locations() - 1U}];  // resize
  link_nearby_stations(tt, walks);  // the equivalences
  if (!opt.walks_) {
    auto const timer = scoped_timer{"loader.footpath.beelines"};

    // Walks between nearby and equivalent stops and their chains, in walking
    // minutes.
    add_equivalence_footpaths(tt, walks, opt.max_footpath_length_);
    add_transitive_walks(tt, walks, opt.max_footpath_length_);
  }

  if (opt.merge_dupes_intra_src_ || opt.merge_dupes_inter_src_) {
    merge_duplicates(tt, opt.merge_threshold_, opt.merge_dupes_intra_src_,
                     opt.merge_dupes_inter_src_, opt.merge_stats_dir_,
                     opt.src_tags_);
  }

  if (opt.walks_) {
    opt.walks_(tt, walks);
  }

  // Then transfer times.
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    auto bucket = walks[l];
    for (auto i = 0U; i != bucket.size(); ++i) {
      auto const t = bucket[i].target();
      bucket[i] =
          footpath{t, max_with_transfer_times(tt, l, t, bucket[i].duration())};
    }
  }
  apply_transfer_rules(tt, shapes, walks);
}

}  // namespace nigiri::loader
