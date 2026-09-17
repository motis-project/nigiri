#include "nigiri/loader/merge_duplicates.h"

#include "nigiri/loader/merge_stats.h"

#include <cassert>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <ranges>
#include <span>
#include <vector>

#include "utl/equal_ranges_linear.h"
#include "utl/erase_duplicates.h"
#include "utl/helpers/algorithm.h"
#include "utl/parallel_for.h"

#include "geo/latlng.h"
#include "geo/point_rtree.h"

#include "nigiri/logging.h"
#include "nigiri/timetable.h"

namespace nigiri::loader {

constexpr auto const kEqDist = 600.0;  // [m]

bool is_duplicate(timetable const& tt,
                  route_idx_t const a_route,
                  route_idx_t const b_route,
                  transport_idx_t const a,
                  transport_idx_t const b,
                  stop_idx_t const n_stops,
                  int const budget) {
  auto delta = 0;
  for (auto i = stop_idx_t{0U}; i != n_stops; ++i) {
    if (i != 0U) {
      delta += std::abs(tt.event_mam(a_route, a, i, event_type::kArr).count() -
                        tt.event_mam(b_route, b, i, event_type::kArr).count());
    }
    if (i != n_stops - 1U) {
      delta += std::abs(tt.event_mam(a_route, a, i, event_type::kDep).count() -
                        tt.event_mam(b_route, b, i, event_type::kDep).count());
    }
    if (delta > budget) {
      return false;
    }
  }
  return true;
}

void repoint_trips(timetable& tt,
                   transport_idx_t const from,
                   transport_idx_t const to) {
  for (auto const merged_trips_idx : tt.transport_to_trip_section_[from]) {
    for (auto const trp : tt.merged_trips_[merged_trips_idx]) {
      for (auto& [t, range] : tt.trip_transport_ranges_[trp]) {
        if (t == from) {
          t = to;
        }
      }
    }
  }
}

transport_idx_t merge(timetable& tt,
                      transport_idx_t const a,
                      transport_idx_t const b) {
  auto const& bf_a = tt.bitfields_[tt.transport_traffic_days_[a]];
  auto const& bf_b = tt.bitfields_[tt.transport_traffic_days_[b]];
  auto const shared = bf_a & bf_b;
  assert(shared.any());

  auto const rest_a = bf_a & ~shared;
  auto const rest_b = bf_b & ~shared;

  // CASE (1): A is subset of B => A disappears
  if (rest_a.none()) {
    tt.transport_traffic_days_[a] = kEmptyBitfieldIdx;
    repoint_trips(tt, a, b);
    return a;
  }

  // CASE (2): B is subset of A => B disappears
  if (rest_b.none()) {
    tt.transport_traffic_days_[b] = kEmptyBitfieldIdx;
    repoint_trips(tt, b, a);
    return b;
  }

  // CASE (3): Both trips survive (not 100% the same or subset relation)
  // => 3.1 Remove intersection from A (-> B now represents A for common days)
  // => 3.2 Point A's trip transport ranges to B

  // Remove intersection from A.
  tt.transport_traffic_days_[a] = tt.register_bitfield(rest_a);

  // Collect A's trips.
  auto a_trips = std::vector<std::pair<trip_idx_t, interval<stop_idx_t>>>{};
  for (auto const merged_trips_idx : tt.transport_to_trip_section_[a]) {
    for (auto const a_trp : tt.merged_trips_[merged_trips_idx]) {
      for (auto const& [t, a_stop_range] : tt.trip_transport_ranges_[a_trp]) {
        if (t == a) {
          a_trips.emplace_back(a_trp, a_stop_range);
        }
      }
    }
  }

  // Point A's trips to B.
  auto const has_b = [&](transport_range_t const& r) { return r.first == b; };
  for (auto const& [a_trp, a_stop_range] : a_trips) {
    auto ranges = tt.trip_transport_ranges_[a_trp];
    if (utl::none_of(ranges, has_b)) {
      ranges.push_back(transport_range_t{b, a_stop_range});
    }
  }

  return transport_idx_t::invalid();
}

void count_overlap(timetable const& tt,
                   transport_idx_t const a,
                   transport_idx_t const b,
                   merge_stats& stats) {
  auto const pa = tt.transport_section_providers_[a].front();
  auto const pb = tt.transport_section_providers_[b].front();
  stats.provider_overlap_.add(pa, pb, a);
  stats.provider_overlap_.add(pb, pa, b);

  auto const sa = tt.providers_[pa].src_;
  auto const sb = tt.providers_[pb].src_;
  stats.src_overlap_.add(sa, sb, a);
  stats.src_overlap_.add(sb, sa, b);
}

void merge_route_pair(timetable& tt,
                      route_idx_t const a_route,
                      route_idx_t const b_route,
                      stop_idx_t const n_stops,
                      duration_t const threshold,
                      merge_stats& stats) {
  auto const budget = static_cast<int>(2U * (n_stops - 1U)) * threshold.count();
  auto const a_range = tt.route_transport_ranges_[a_route];
  auto const b_range = tt.route_transport_ranges_[b_route];
  auto const a_dep = tt.event_times_at_stop(a_route, 0U, event_type::kDep);
  auto const b_dep = tt.event_times_at_stop(b_route, 0U, event_type::kDep);
  auto const is_reflexive = a_route == b_route;

  auto b_first = std::size_t{0U};
  for (auto ai = std::size_t{0U}; ai != a_dep.size(); ++ai) {
    auto const t_a = a_dep[ai].count();

    while (b_first != b_dep.size() && b_dep[b_first].count() + budget < t_a) {
      ++b_first;
    }

    auto const a_t = transport_idx_t{to_idx(a_range.from_) + ai};
    for (auto bi = is_reflexive ? std::max(b_first, ai + 1U) : b_first;
         bi != b_dep.size(); ++bi) {
      if (b_dep[bi].count() > t_a + budget) {
        break;
      }

      auto const b_t = transport_idx_t{to_idx(b_range.from_) + bi};
      auto const has_common_traffic_days =
          (tt.bitfields_[tt.transport_traffic_days_[a_t]] &
           tt.bitfields_[tt.transport_traffic_days_[b_t]])
              .any();
      if (!has_common_traffic_days) {
        continue;
      }

      if (!is_duplicate(tt, a_route, b_route, a_t, b_t, n_stops, budget)) {
        continue;
      }

      auto const absorbed = merge(tt, a_t, b_t);

      ++stats.n_merges_;
      if (absorbed != transport_idx_t::invalid()) {
        ++stats.n_absorbed_transports_;
        auto const p = tt.transport_section_providers_[absorbed].front();
        ++stats.provider_n_absorbed_transports_[p];
        ++stats.src_n_absorbed_transports_[tt.providers_[p].src_];
      }

      count_overlap(tt, a_t, b_t, stats);
    }
  }
}

void merge_duplicates(timetable& tt,
                      merge_threshold_t const& clasz_threshold,
                      bool const intra_src,
                      bool const inter_src,
                      std::filesystem::path const& stats_dir,
                      vector_map<source_idx_t, std::string> const& src_tags) {
  auto stats = merge_stats{tt};

  // Key (root location, stop sequence length) => routes
  struct route_start {
    bool operator<(route_start const& o) const {
      return std::tie(root_, n_stops_, route_) <
             std::tie(o.root_, o.n_stops_, o.route_);
    }

    location_idx_t root_;
    stop_idx_t n_stops_;
    route_idx_t route_;
  };

  auto const route_starts = [&]() {
    auto v = std::vector<route_start>{};
    v.reserve(tt.n_routes());
    for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
      auto const seq = tt.route_location_seq_[r];

      // Skip dummy stations.
      auto const root =
          tt.locations_.get_root_idx(stop{seq.front()}.location_idx());
      if (tt.locations_.src_[root] == source_idx_t::invalid()) {
        continue;
      }

      // Skip null island.
      auto const pos = tt.locations_.coordinates_[root];
      if (std::abs(pos.lat_) < 2.0 && std::abs(pos.lng_) < 2.0) {
        continue;
      }

      v.emplace_back(root, static_cast<stop_idx_t>(seq.size()), r);
    }
    utl::sort(v);
    return v;
  }();

  // Collect distinct roots that a route starts at.
  auto const roots = [&]() {
    auto v = std::vector<location_idx_t>{};
    utl::equal_ranges_linear(
        route_starts,
        [](route_start const& a, route_start const& b) {
          return a.root_ == b.root_;
        },
        [&](auto&& from, auto&&) { v.push_back(from->root_); });
    return v;
  }();

  // Root level neighbors: location_idx -> [location_idx, ...]
  auto const neighbours = [&]() {
    // Every root a route stops at: seq_matches looks up all of them, not
    // only the ones a route starts at.
    auto stop_roots = std::vector<location_idx_t>{};
    for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
      for (auto const x : tt.route_location_seq_[r]) {
        stop_roots.push_back(
            tt.locations_.get_root_idx(stop{x}.location_idx()));
      }
    }
    utl::erase_duplicates(stop_roots);

    // Prepare root indexed neighbor map.
    auto nb = std::vector<std::vector<location_idx_t>>(stop_roots.size());
    auto index = std::vector<geo::point_rtree::value_t>{};
    index.reserve(stop_roots.size());
    for (auto i = std::size_t{0U}; i != stop_roots.size(); ++i) {
      index.emplace_back(tt.locations_.coordinates_[stop_roots[i]], i);
    }

    // Fill root indexed neighbor map with R-tree results.
    auto const rtree = geo::point_rtree{index};
    utl::parallel_for_run(stop_roots.size(), [&](std::size_t const i) {
      for (auto const j : rtree.in_radius(
               tt.locations_.coordinates_[stop_roots[i]], kEqDist)) {
        if (j != i) {
          nb[i].emplace_back(stop_roots[j]);
        }
      }
      utl::sort(nb[i]);
    });

    // Convert root indexed to full location indexed neighbor map.
    auto v = vecvec<location_idx_t, location_idx_t>{};
    auto i = std::size_t{0U};
    for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
      if (i != stop_roots.size() && stop_roots[i] == l) {
        v.emplace_back(nb[i++]);  // indexed root
      } else {
        v.add_back_sized(0U);  // not a root
      }
    }
    return v;
  }();

  auto const seq_matches = [&](route_idx_t const a, route_idx_t const b) {
    auto const sa = tt.route_location_seq_[a];
    auto const sb = tt.route_location_seq_[b];

    for (auto i = 0U; i != sa.size(); ++i) {
      auto const ra = tt.locations_.get_root_idx(stop{sa[i]}.location_idx());
      auto const rb = tt.locations_.get_root_idx(stop{sb[i]}.location_idx());

      // Check exact match.
      if (ra == rb) {
        continue;
      }

      // Neighbor match.
      auto const nb = neighbours[ra];
      if (!std::binary_search(begin(nb), end(nb), rb)) {
        return false;  // mismatch -> early exit
      }
    }

    return true;
  };

  auto const route_src = [&](route_idx_t const r) {
    return tt.locations_
        .src_[stop{tt.route_location_seq_[r].front()}.location_idx()];
  };

  auto const attrs_match = [&](route_idx_t const a, route_idx_t const b) {
    for (auto f = 0U; f != route_flag::kNumRouteFlags; ++f) {
      if (tt.route_flags_[f][to_idx(a) * 2U] !=
              tt.route_flags_[f][to_idx(b) * 2U] ||
          tt.route_flags_[f][to_idx(a) * 2U + 1U] !=
              tt.route_flags_[f][to_idx(b) * 2U + 1U] ||
          !std::ranges::equal(tt.route_flags_per_section_[f][a],
                              tt.route_flags_per_section_[f][b])) {
        return false;
      }
    }

    auto const sa = tt.route_location_seq_[a];
    auto const sb = tt.route_location_seq_[b];
    for (auto i = 0U; i != sa.size(); ++i) {
      auto const x = stop{sa[i]};
      auto const y = stop{sb[i]};
      if (x.in_allowed() != y.in_allowed() ||
          x.out_allowed() != y.out_allowed() ||
          x.in_allowed_wheelchair() != y.in_allowed_wheelchair() ||
          x.out_allowed_wheelchair() != y.out_allowed_wheelchair()) {
        return false;
      }
    }

    return true;
  };

  auto const routes_of = [&](location_idx_t const root) {
    return std::ranges::equal_range(route_starts, root, {},
                                    &route_start::root_);
  };

  auto const merge_root_pair = [&](location_idx_t const a_root,
                                   location_idx_t const b_root) {
    auto const b_routes = routes_of(b_root);
    for (auto const& a : routes_of(a_root)) {
      auto const stop_matching_routes = std::ranges::equal_range(
          b_routes, a.n_stops_, {}, &route_start::n_stops_);
      for (auto const& b : stop_matching_routes) {
        auto const a_route = a.route_;
        auto const b_route = b.route_;

        // Skip lower triangle (keep diagonal + upper triangle).
        if (a_root == b_root && b_route < a_route) {
          continue;
        }

        // Respect merge flags.
        auto const same_src = route_src(a_route) == route_src(b_route);
        if ((same_src && !intra_src) || (!same_src && !inter_src)) {
          continue;
        }

        if (same_src && !attrs_match(a_route, b_route)) {
          continue;
        }

        // Compare stop sequence on root-neighbor level.
        if (!seq_matches(a_route, b_route)) {
          continue;
        }

        // Merge a vs b route transport pairs.
        ++stats.n_route_pairs_;
        auto const threshold = std::min(
            clasz_threshold[static_cast<unsigned>(tt.route_clasz_[a_route])],
            clasz_threshold[static_cast<unsigned>(tt.route_clasz_[b_route])]);
        merge_route_pair(tt, a_route, b_route, a.n_stops_, threshold, stats);
      }
    }
  };

  // Merge.
  for (auto const a_root : roots) {
    // Merge within same root.
    merge_root_pair(a_root, a_root);

    for (auto const b_root : neighbours[a_root]) {
      // Neighbors are sorted
      // -> merge each unordered pair exactly once.
      if (b_root > a_root) {
        merge_root_pair(a_root, b_root);
      }
    }
  }

  stats.provider_overlap_.finish(stats.provider_n_transports_.size());
  stats.src_overlap_.finish(stats.src_n_transports_.size());

  if (!stats_dir.empty()) {
    std::filesystem::create_directories(stats_dir);
    write_merge_stats(tt, stats, stats_dir / "merge_stats.json", src_tags);
    write_merge_html(tt, stats, stats_dir / "merge_stats.html", src_tags);
  }
}

}  // namespace nigiri::loader
