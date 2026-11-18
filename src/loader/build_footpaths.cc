#include "nigiri/loader/build_footpaths.h"

#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>

#include "geo/latlng.h"

#include "utl/erase_if.h"
#include "utl/helpers/algorithm.h"
#include "utl/parallel_for.h"

#include "nigiri/loader/apply_transfer_rules.h"
#include "nigiri/loader/import_context.h"
#include "nigiri/loader/link_nearby_stations.h"
#include "nigiri/loader/merge_duplicates.h"
#include "nigiri/loader/station_footpaths.h"
#include "nigiri/loader/station_graph.h"
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

void add_equivalence_footpaths(
    timetable const& tt,
    import_context const& ctx,
    mutable_fws_multimap<location_idx_t, footpath>& walks,
    std::uint16_t const max_footpath_length) {
  auto const max = std::min<int>(max_footpath_length, footpath::kMax.count());
  auto const add = [&](location_idx_t const a, location_idx_t const b) {
    auto const& pos = tt.locations_.coordinates_[a];
    auto const dist = std::sqrt(
        geo::approx_squared_distance(pos, tt.locations_.coordinates_[b],
                                     geo::approx_distance_lng_degrees(pos)));
    auto const minutes = std::max(duration_t{2}, walk_duration(dist));
    if (minutes.count() <= max) {
      walks[a].emplace_back(b, minutes);
    }
  };

  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    for (auto const eq : ctx.equivalences_[l]) {
      if (eq != l) {
        add(l, eq);
        add(eq, l);
      }
    }
  }
}

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

  for (auto l = location_idx_t{0U}; l != n; ++l) {
    auto bucket = fps[l];
    bucket.clear();
    for (auto const c : chains[to_idx(l)]) {
      bucket.push_back(c);
    }
  }
}

void replace_station_walks(
    vector_map<location_idx_t, std::vector<station_footpath>> const&
        station_fps,
    mutable_fws_multimap<location_idx_t, footpath>& walks) {
  for (auto l = location_idx_t{0U}; l != station_fps.size(); ++l) {
    auto const& s = station_fps[l];
    if (s.empty()) {
      continue;
    }
    auto bucket = walks[l];
    utl::erase_if(bucket, [&](footpath const fp) {
      return utl::any_of(s, [&](station_footpath const& x) {
        return x.target_ == fp.target();
      });
    });
    for (auto i = 0U; i != s.size(); ++i) {
      if (i == 0U || s[i].target_ != s[i - 1U].target_) {
        bucket.push_back(footpath{s[i].target_, s[i].duration_});
      }
    }
  }
}

// Routed footpaths with restriction records join the station graph
// footpaths of their profile, except for pairs that a station graph covers.
void merge_routed_footpaths(timetable const& tt,
                            station_footpaths_t& routed,
                            station_footpaths_t& station_fps) {
  auto const& coverage = station_fps[kDefaultProfile];
  auto const is_covered = [&](location_idx_t const from,
                              location_idx_t const to) {
    return from < coverage.size() &&
           utl::any_of(
               coverage[from],
               [&](station_footpath const& fp) { return fp.target_ == to; });
  };
  auto n_merged = 0U;
  for (auto prf = profile_idx_t{1U}; prf != kNProfiles; ++prf) {
    auto& from_routing = routed[prf];
    if (from_routing.empty()) {
      continue;
    }
    auto& fps = station_fps[prf];
    fps.resize(tt.n_locations());
    for (auto l = location_idx_t{0U}; l != from_routing.size(); ++l) {
      if (from_routing[l].empty()) {
        continue;
      }
      for (auto const& fp : from_routing[l]) {
        if (!is_covered(l, fp.target_)) {
          fps[l].push_back(fp);
          ++n_merged;
        }
      }
      remove_dominated(fps[l]);
      from_routing[l] = {};
    }
  }
  if (n_merged != 0U) {
    log(log_lvl::info, "loader.footpaths", "{} routed restricted footpaths",
        n_merged);
  }
}

void build_footpaths(timetable& tt,
                     import_context& ctx,
                     finalize_options const& opt,
                     shapes_storage* shapes) {
  auto walks = mutable_fws_multimap<location_idx_t, footpath>{};
  walks[location_idx_t{tt.n_locations() - 1U}];  // resize
  link_nearby_stations(tt, ctx, walks);  // the equivalences
  if (!opt.compute_walks_) {
    auto const timer = scoped_timer{"loader.footpath.beelines"};
    add_equivalence_footpaths(tt, ctx, walks, opt.max_footpath_length_);
    add_transitive_walks(tt, walks, opt.max_footpath_length_);
  }

  if (opt.merge_dupes_intra_src_ || opt.merge_dupes_inter_src_) {
    merge_duplicates(tt, ctx, opt.merge_threshold_, opt.merge_dupes_intra_src_,
                     opt.merge_dupes_inter_src_, opt.merge_stats_dir_,
                     opt.src_tags_);
  }

  auto routed_restricted = station_footpaths_t{};
  if (opt.compute_walks_) {
    opt.compute_walks_(tt, walks, routed_restricted);
  }

  // Station graphs: the fastest route replaces all other walks between two
  // of their stops, the alternatives become restricted footpaths.
  resolve_refs(tt, ctx.station_graph_);
  auto const station_routes = get_station_routes(ctx.station_graph_);
  if (opt.station_paths_ != nullptr) {
    write_station_paths(ctx.station_graph_, station_routes,
                        *opt.station_paths_);
  }
  auto station_fps = get_station_footpaths(tt, station_routes);
  replace_station_walks(station_fps[kDefaultProfile], walks);
  merge_routed_footpaths(tt, routed_restricted, station_fps);

  // Then transfer times.
  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    auto bucket = walks[l];
    for (auto i = 0U; i != bucket.size(); ++i) {
      auto const t = bucket[i].target();
      bucket[i] =
          footpath{t, max_with_transfer_times(tt, l, t, bucket[i].duration())};
    }
  }

  // Then transfer rules.
  apply_transfer_rules(tt, ctx, shapes, walks, station_fps);
}

}  // namespace nigiri::loader
