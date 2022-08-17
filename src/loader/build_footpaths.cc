#include "nigiri/loader/build_footpaths.h"

#include <optional>
#include <stack>

#include "geo/latlng.h"
#include "geo/point_rtree.h"

#include "utl/enumerate.h"
#include "utl/equal_ranges_linear.h"
#include "utl/parallel_for.h"
#include "utl/verify.h"

#include "nigiri/loader/floyd_warshall.h"
#include "nigiri/logging.h"
#include "nigiri/section_db.h"

namespace nigiri::loader {

constexpr const auto kWalkSpeed = 1.5;  // m/s
constexpr auto const kAdjustedMaxDuration = 15;  // [minutes]

constexpr auto kNoComponent = std::numeric_limits<uint32_t>::max();

// station_idx -> [footpath, ...]
using footgraph = vector<vector<footpath>>;

// (component, original station_idx)
using component_vec = std::vector<std::pair<uint32_t, uint32_t>>;
using component_it = component_vec::iterator;
using component_range = std::pair<component_it, component_it>;

void link_nearby_stations(timetable& tt) {
  constexpr auto const kLinkNearbyMaxDistance = 300;  // [m];

  auto const locations_rtree =
      geo::make_point_rtree(tt.locations_.coordinates_);

  for (auto from_idx = location_idx_t{0}; from_idx != tt.locations_.src_.size();
       ++from_idx) {
    auto const from_src = tt.locations_.src_[from_idx];
    auto const from_pos = tt.locations_.coordinates_[from_idx];

    if (from_src == source_idx_t::invalid()) {
      continue;  // no dummy stations
    }

    for (auto const& to_idx :
         locations_rtree.in_radius(from_pos, kLinkNearbyMaxDistance)) {
      if (from_idx == to_idx) {
        continue;
      }

      auto const to_l_idx = location_idx_t{static_cast<unsigned>(to_idx)};
      auto const to_src = tt.locations_.src_[to_l_idx];
      auto const to_pos = tt.locations_.coordinates_[to_l_idx];
      if (to_src == source_idx_t::invalid() /* no dummy stations */
          || from_src == to_src /* don't short-circuit */) {
        continue;
      }

      auto const from_transfer_time = tt.locations_.transfer_time_[from_idx];
      auto const to_transfer_time = tt.locations_.transfer_time_[to_l_idx];
      auto const walk_duration = duration_t{static_cast<unsigned>(
          std::round(geo::distance(from_pos, to_pos) / (60 * kWalkSpeed)))};
      auto const duration =
          std::max({from_transfer_time, to_transfer_time, walk_duration});

      auto const l_from_idx = location_idx_t{static_cast<unsigned>(from_idx)};
      auto const l_to_idx = location_idx_t{static_cast<unsigned>(to_idx)};

      tt.locations_.footpaths_out_[l_from_idx].emplace_back(l_to_idx, duration);
      tt.locations_.footpaths_in_[l_to_idx].emplace_back(l_from_idx, duration);
      tt.locations_.equivalences_[l_from_idx].emplace_back(l_to_idx);
    }
  }
}

footgraph get_footpath_graph(timetable& tt) {
  footgraph g;
  g.resize(to_idx(tt.locations_.src_.size()));
  for (auto i = 0U; i != tt.locations_.src_.size(); ++i) {
    auto const idx = location_idx_t{static_cast<unsigned>(i)};
    g[i].emplace_back(idx, tt.locations_.transfer_time_[idx]);
    g[i].insert(end(g[i]), begin(tt.locations_.footpaths_out_[idx]),
                end(tt.locations_.footpaths_out_[idx]));
  }
  return g;
}

static std::vector<std::pair<uint32_t, uint32_t>> find_components(
    footgraph const& fgraph) {
  std::vector<std::pair<uint32_t, uint32_t>> components(fgraph.size());
  std::generate(begin(components), end(components), [i = 0UL]() mutable {
    return std::pair<uint32_t, uint32_t>{kNoComponent, i++};
  });

  std::stack<uint32_t> stack;  // invariant: stack is empty
  for (auto i = 0UL; i < fgraph.size(); ++i) {
    if (components[i].first != kNoComponent || fgraph[i].empty()) {
      continue;
    }

    stack.emplace(i);
    while (!stack.empty()) {
      auto j = stack.top();
      stack.pop();

      if (components[j].first == i) {
        continue;
      }

      components[j].first = i;
      for (auto const& f : fgraph[j]) {
        if (components[to_idx(f.target_)].first != i) {
          stack.push(to_idx(f.target_));
        }
      }
    }
  }

  return components;
}

void process_component(timetable& tt,
                       component_it const lb,
                       component_it const ub,
                       footgraph const& fgraph) {
  if (lb->first == kNoComponent) {
    return;
  }

  auto const size = std::distance(lb, ub);
  if (size == 2) {
    auto const idx_a = lb->second;
    auto const idx_b = std::next(lb)->second;
    auto const l_idx_a = location_idx_t{static_cast<unsigned>(idx_a)};
    auto const l_idx_b = location_idx_t{static_cast<unsigned>(idx_b)};

    if (!fgraph[idx_a].empty()) {
      utl::verify_silent(fgraph[idx_a].size() == 1,
                         "invalid size (a): idx_a={}, size={}, data=[{}], "
                         "idx_b={}, size = {} ",
                         idx_a, fgraph[idx_a].size(), fgraph[idx_a], idx_b,
                         fgraph[idx_b].size(), fgraph[idx_b]);

      tt.locations_.footpaths_out_[l_idx_a].push_back(fgraph[idx_a].front());
      tt.locations_.footpaths_in_[l_idx_b].push_back(fgraph[idx_a].front());
    }
    if (!fgraph[idx_b].empty()) {
      utl::verify_silent(
          fgraph[idx_b].size() == 1,
          "invalid size (a): idx_a={}, size={}, idx_b={}, size={}", idx_a,
          fgraph[idx_a].size(), idx_b, fgraph[idx_b].size());
      tt.locations_.footpaths_out_[l_idx_b].push_back(fgraph[idx_b].front());
      tt.locations_.footpaths_in_[l_idx_a].push_back(fgraph[idx_b].front());
    }
    return;
  }
  utl::verify(size > 2, "invalid size");

  constexpr auto const kInvalidTime = std::numeric_limits<duration_t>::max();
  auto mat = make_matrix(size, size, kInvalidTime);
  for (auto i = 0; i < size; ++i) {
    auto it = lb;
    for (auto const& edge : fgraph[(lb + i)->second]) {  // precond.: sorted!
      while (it != ub && edge.target_ != it->second) {
        ++it;
      }
      auto j = static_cast<unsigned>(std::distance(lb, it));
      mat(i, j) = std::min(edge.duration_, mat(i, j));
      mat(j, i) = std::min(edge.duration_, mat(j, i));
    }
  }

  floyd_warshall(mat);

  for (auto i = 0; i < size; ++i) {
    for (auto j = 0; j < size; ++j) {
      if (mat(i, j) == kInvalidTime || i == j) {
        continue;
      }

      auto const idx_a = std::next(lb, i)->second;
      auto const idx_b = std::next(lb, j)->second;
      auto const l_idx_a = location_idx_t{static_cast<unsigned>(idx_a)};
      auto const l_idx_b = location_idx_t{static_cast<unsigned>(idx_b)};

      // each node only in one cluster -> no sync required
      tt.locations_.footpaths_out_[l_idx_a].emplace_back(l_idx_b, mat(i, j));
      tt.locations_.footpaths_in_[l_idx_b].emplace_back(l_idx_a, mat(i, j));
    }
  }
}

void transitivize_footpaths(timetable& tt) {
  scoped_timer timer("building transitively closed foot graph");

  auto const fgraph = get_footpath_graph(tt);

  auto components = find_components(fgraph);
  std::sort(begin(components), end(components));

  std::vector<component_range> ranges;
  utl::equal_ranges_linear(
      components,
      [](auto const& a, auto const& b) { return a.first == b.first; },
      [&](auto lb, auto ub) { ranges.emplace_back(lb, ub); });

  auto const errors = utl::parallel_for(
      ranges,
      [&](auto const& range) {
        process_component(tt, range.first, range.second, fgraph);
      },
      utl::parallel_error_strategy::CONTINUE_EXEC);
  if (!errors.empty()) {
    for (auto const& [idx, ex] : errors) {
      try {
        std::rethrow_exception(ex);
      } catch (std::exception const& e) {
        log(log_lvl::error, "nigiri.loader.footpaths",
            "footpath error: {} ({})", idx, e.what());
      }
    }
  }
}

void build_footpaths(timetable& tt) {
  link_nearby_stations(tt);
  transitivize_footpaths(tt);
}

}  // namespace nigiri::loader
