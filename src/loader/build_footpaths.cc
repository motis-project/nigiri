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

namespace nigiri::loader {

constexpr const auto kWalkSpeed = 1.5;  // m/s

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

  for (auto from_idx = location_idx_t{0U};
       from_idx != tt.locations_.src_.size(); ++from_idx) {
    auto const from_src = tt.locations_.src_[from_idx];
    auto const from_pos = tt.locations_.coordinates_[from_idx];

    if (from_src == source_idx_t::invalid()) {
      continue;  // no dummy stations
    }

    for (auto const& to_idx :
         locations_rtree.in_radius(from_pos, kLinkNearbyMaxDistance)) {
      auto const to_l_idx = location_idx_t{static_cast<unsigned>(to_idx)};
      if (from_idx == to_l_idx) {
        continue;
      }

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
  g.resize(tt.locations_.src_.size());
  for (auto i = 0U; i != tt.locations_.src_.size(); ++i) {
    auto const idx = location_idx_t{i};
    g[i].insert(end(g[i]), begin(tt.locations_.footpaths_out_[idx]),
                end(tt.locations_.footpaths_out_[idx]));
    std::sort(begin(g[i]), end(g[i]));
    //    for (auto const& fp : tt.locations_.footpaths_out_[idx]) {
    //      g[to_idx(fp.target_)].emplace_back(idx, fp.duration_);
    //    }
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
  for (auto i = 0U; i < fgraph.size(); ++i) {
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

  auto const size = static_cast<std::uint32_t>(std::distance(lb, ub));
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
  utl::verify(size > 2, "invalid size [id={}], first={}", lb->first,
              tt.locations_.ids_.at(location_idx_t{lb->second}).view());

  constexpr auto const kInvalidTime = std::numeric_limits<std::uint8_t>::max();
  auto mat = make_matrix(size, size, kInvalidTime);
  for (auto i = 0U; i != size; ++i) {
    auto it = lb;
    for (auto const& edge : fgraph[(lb + i)->second]) {  // precond.: sorted!
      while (it != ub && edge.target_ != it->second) {
        ++it;
      }
      auto j = static_cast<unsigned>(std::distance(lb, it));
      std::cerr << "adding entry " << i << " " << j << ": " << edge.duration_
                << "\n";
      mat(i, j) = std::min(static_cast<std::uint8_t>(edge.duration_.count()),
                           mat(i, j));
      mat(j, i) = std::min(static_cast<std::uint8_t>(edge.duration_.count()),
                           mat(j, i));
      std::cerr << "  " << static_cast<int>(mat(i, j)) << "\n";
    }
  }

  std::cerr << mat << "\n";

  floyd_warshall(mat);

  std::cerr << mat << "\n";

  std::cerr << "WRITING FOOTPATHS:\n";
  for (auto i = 0U; i < size; ++i) {
    auto const idx_a = std::next(lb, i)->second;
    auto const l_idx_a = location_idx_t{static_cast<unsigned>(idx_a)};

    for (auto j = 0U; j < size; ++j) {
      if (mat(i, j) == kInvalidTime || i == j) {
        continue;
      }

      auto const idx_b = std::next(lb, j)->second;
      auto const l_idx_b = location_idx_t{static_cast<unsigned>(idx_b)};

      tt.locations_.footpaths_out_[l_idx_a].emplace_back(l_idx_b,
                                                         duration_t{mat(i, j)});
      tt.locations_.footpaths_in_[l_idx_b].emplace_back(l_idx_a,
                                                        duration_t{mat(i, j)});

      std::cerr << i << "=" << tt.locations_.names_.at(l_idx_a).view() << " - "
                << j << "=" << tt.locations_.names_.at(l_idx_b).view()
                << ": duration=" << static_cast<int>(mat(i, j)) << "\n";
    }
  }
}

void transitivize_footpaths(timetable& tt) {
  scoped_timer timer("building transitively closed foot graph");

  auto const fgraph = get_footpath_graph(tt);

  auto components = find_components(fgraph);
  std::sort(begin(components), end(components));

  std::cerr << "FOOTPATHS\n";
  for (auto i = 0U; i != tt.locations_.footpaths_out_.size(); ++i) {
    auto const l = location_idx_t{i};
    std::cerr << tt.locations_.names_[l].view() << " ["
              << tt.locations_.ids_[l].view() << "]\n";
    for (auto const& fp : tt.locations_.footpaths_out_[l]) {
      std::cerr << "  " << tt.locations_.names_[fp.target_].view() << " ["
                << tt.locations_.ids_[fp.target_].view() << "]\n";
    }
  }
  std::cerr << "---\n";

  tt.locations_.footpaths_out_.clear();
  tt.locations_.footpaths_out_[location_idx_t{tt.locations_.src_.size() - 1}];
  tt.locations_.footpaths_in_.clear();
  tt.locations_.footpaths_in_[location_idx_t{tt.locations_.src_.size() - 1}];

  std::vector<component_range> ranges;
  utl::equal_ranges_linear(
      components,
      [](auto const& a, auto const& b) { return a.first == b.first; },
      [&](auto lb, auto ub) {
        std::cerr << "COMPONENT " << lb->second << "\n";
        for (auto const& x : it_range{lb, ub}) {
          std::cerr << "  i="
                    << tt.locations_.names_.at(location_idx_t{x.second}).view()
                    << " "
                    << tt.locations_.ids_.at(location_idx_t{x.second}).view()
                    << ", component_id=" << x.first << "\n";
          for (auto const& fp :
               tt.locations_.footpaths_out_.at(location_idx_t{x.second})) {
            std::cerr
                << "     " << fp.duration_ << " -> "
                << tt.locations_.names_.at(location_idx_t{fp.target_}).view()
                << " "
                << tt.locations_.ids_.at(location_idx_t{fp.target_}).view()
                << "\n";
          }
        }
        process_component(tt, lb, ub, fgraph);
      });

  std::cerr << "TRANSITTIVE FOOTPATHS\n";
  for (auto i = 0U; i != tt.locations_.footpaths_out_.size(); ++i) {
    auto const l = location_idx_t{i};
    std::cerr << tt.locations_.names_[l].view() << " ["
              << tt.locations_.ids_[l].view() << "]\n";
    for (auto const& fp : tt.locations_.footpaths_out_[l]) {
      std::cerr << "  " << tt.locations_.names_[fp.target_].view() << " ["
                << tt.locations_.ids_[fp.target_].view() << "]\n";
    }
  }
  std::cerr << "---\n";
}

void add_links_to_and_between_children(timetable& tt) {
  for (auto const [l, children] : utl::enumerate(tt.locations_.children_)) {
    auto const parent = location_idx_t{l};
    auto const t = tt.locations_.transfer_time_[parent];
    for (auto i = 0U; i != children.size(); ++i) {
      tt.locations_.footpaths_out_[parent].emplace_back(children[i], t);
      tt.locations_.footpaths_out_[children[i]].emplace_back(parent, t);
      for (auto j = 0U; j != children.size(); ++j) {
        if (i != j) {
          tt.locations_.footpaths_out_[children[i]].emplace_back(children[j],
                                                                 t);
        }
      }
    }
  }
}

void build_footpaths(timetable& tt) {
  add_links_to_and_between_children(tt);
  link_nearby_stations(tt);
  transitivize_footpaths(tt);
}

}  // namespace nigiri::loader
