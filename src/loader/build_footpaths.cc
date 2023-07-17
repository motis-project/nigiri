#include "nigiri/loader/build_footpaths.h"

#include <optional>
#include <stack>

#include "fmt/format.h"
#include "fmt/ostream.h"

#include "geo/latlng.h"
#include "geo/point_rtree.h"

#include "utl/enumerate.h"
#include "utl/equal_ranges_linear.h"
#include "utl/parallel_for.h"
#include "utl/verify.h"

#include "nigiri/loader/floyd_warshall.h"
#include "nigiri/logging.h"
#include "utl/erase_duplicates.h"
#include "utl/erase_if.h"

#define NIGIRI_BUILD_FOOTPATHS_DEBUG
#if defined(NIGIRI_BUILD_FOOTPATHS_DEBUG)
template <typename T>
struct fmt::formatter<nigiri::matrix<T>> {
  constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(nigiri::matrix<T> const& m, FormatContext& ctx) const
      -> decltype(ctx.out()) {
    for (auto i = 0U; i != m.n_rows_; ++i) {
      for (auto j = 0U; j != m.n_columns_; ++j) {
        if (m[i][j] == std::numeric_limits<T>::max()) {
          fmt::format_to(ctx.out(), "** ");
        } else {
          fmt::format_to(ctx.out(), "{:2} ", m[i][j]);
        }
      }
      fmt::format_to(ctx.out(), "\n");
    }
    return ctx.out();
  }
};

template <typename... Args>
void trace(char const* fmt_str, Args... args) {
  fmt::print(std::cout, fmt_str, std::forward<Args&&>(args)...);
}
#else
#define print_dbg(...)
#define trace(...)
#endif

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
    auto const from_pos = tt.locations_.coordinates_[from_idx];
    if (std::abs(from_pos.lat_) < 2.0 && std::abs(from_pos.lng_) < 2.0) {
      continue;
    }

    auto const from_src = tt.locations_.src_[from_idx];
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

      auto const from_transfer_time =
          duration_t{tt.locations_.transfer_time_[from_idx]};
      auto const to_transfer_time =
          duration_t{tt.locations_.transfer_time_[to_l_idx]};
      auto const walk_duration = duration_t{static_cast<unsigned>(
          std::round(geo::distance(from_pos, to_pos) / (60 * kWalkSpeed)))};
      auto const duration =
          std::max({from_transfer_time, to_transfer_time, walk_duration});

      auto const l_from_idx = location_idx_t{static_cast<unsigned>(from_idx)};
      auto const l_to_idx = location_idx_t{static_cast<unsigned>(to_idx)};

      tt.locations_.preprocessing_footpaths_out_[l_from_idx].emplace_back(
          l_to_idx, duration);
      tt.locations_.preprocessing_footpaths_in_[l_to_idx].emplace_back(
          l_from_idx, duration);
      tt.locations_.equivalences_[l_from_idx].emplace_back(l_to_idx);
    }
  }
}

footgraph get_footpath_graph(timetable& tt) {
  footgraph g;
  g.resize(tt.locations_.src_.size());
  for (auto i = 0U; i != tt.locations_.src_.size(); ++i) {
    auto const idx = location_idx_t{i};
    g[i].insert(end(g[i]),
                begin(tt.locations_.preprocessing_footpaths_out_[idx]),
                end(tt.locations_.preprocessing_footpaths_out_[idx]));
    utl::erase_if(g[i],
                  [&](auto&& fp) { return fp.target() == location_idx_t{i}; });
    utl::erase_duplicates(
        g[i], [](auto&& a, auto&& b) { return a.target_ < b.target_; },
        [](auto&& a, auto&& b) {
          return a.target_ == b.target_;
        });  // also sorts
  }
  return g;
}

std::vector<std::pair<uint32_t, uint32_t>> find_components(
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
        if (components[to_idx(f.target())].first != i) {
          stack.push(to_idx(f.target()));
        }
      }
    }
  }

  return components;
}

void process_component(timetable& tt,
                       component_it const lb,
                       component_it const ub,
                       footgraph const& fgraph,
                       matrix<std::uint16_t>& matrix_memory) {
  if (lb->first == kNoComponent) {
    return;
  }

  auto const size = static_cast<std::uint32_t>(std::distance(lb, ub));

#if defined(NIGIRI_BUILD_FOOTPATHS_DEBUG)
  auto dbg = false;
  auto const print_dbg = [&](auto... args) {
    if (dbg) {
      trace(args...);
    }
  };

  auto const id = std::string_view{"de:08317:14414:1:2"};
  auto const needle =
      std::find_if(begin(tt.locations_.ids_), end(tt.locations_.ids_),
                   [&](auto&& x) { return x.view() == id; });

  if (needle != end(tt.locations_.ids_)) {
    auto const needle_l =
        location_idx_t{std::distance(begin(tt.locations_.ids_), needle)};
    for (auto i = 0U; i != size; ++i) {
      if (location_idx_t{(lb + i)->second} == needle_l) {
        trace("FOUND\n");
        dbg = true;
        goto next;
      }
      for (auto const& edge : fgraph[(lb + i)->second]) {
        if (edge.target() == needle_l) {
          trace("FOUND\n");
          dbg = true;
          goto next;
        }
      }
    }
  } else {
    trace("NEEDLE NOT FOUND\n");
  }
next:
#endif

  if (size == 2) {
    auto const idx_a = lb->second;
    auto const idx_b = std::next(lb)->second;
    auto const l_idx_a = location_idx_t{static_cast<unsigned>(idx_a)};
    auto const l_idx_b = location_idx_t{static_cast<unsigned>(idx_b)};

    if (!fgraph[idx_a].empty()) {
      utl_verify(fgraph[idx_a].size() == 1,
                 "invalid size (a): idx_a={}, size={}, data=[{}], "
                 "idx_b={}, size = {} ",
                 idx_a, fgraph[idx_a].size(), fgraph[idx_a], idx_b,
                 fgraph[idx_b].size(), fgraph[idx_b]);

      auto const duration =
          std::max({u8_minutes{fgraph[idx_a].front().duration_},
                    tt.locations_.transfer_time_[l_idx_a],
                    tt.locations_.transfer_time_[l_idx_b]});

      print_dbg("INPUT: {} --{}--> {}\n", location{tt, l_idx_a}, duration,
                location{tt, l_idx_b});

      tt.locations_.preprocessing_footpaths_out_[l_idx_a].emplace_back(
          l_idx_b, duration);
      tt.locations_.preprocessing_footpaths_in_[l_idx_b].emplace_back(l_idx_a,
                                                                      duration);
    }
    if (!fgraph[idx_b].empty()) {
      utl::verify_silent(
          fgraph[idx_b].size() == 1,
          "invalid size (a): idx_a={}, size={}, idx_b={}, size={}", idx_a,
          fgraph[idx_a].size(), idx_b, fgraph[idx_b].size());

      auto const duration =
          std::max({u8_minutes{fgraph[idx_b].front().duration_},
                    tt.locations_.transfer_time_[l_idx_a],
                    tt.locations_.transfer_time_[l_idx_b]});

      print_dbg("INPUT: {} --{}--> {}\n", location{tt, l_idx_b}, duration,
                location{tt, l_idx_a});

      tt.locations_.preprocessing_footpaths_out_[l_idx_b].emplace_back(
          l_idx_a, duration);
      tt.locations_.preprocessing_footpaths_in_[l_idx_a].emplace_back(l_idx_b,
                                                                      duration);
    }
    return;
  }
  utl::verify(size > 2, "invalid size [id={}], first={}", lb->first,
              tt.locations_.ids_.at(location_idx_t{lb->second}).view());

  print_dbg("INPUT\n");
  constexpr auto const kInvalidTime = std::numeric_limits<std::uint16_t>::max();
  auto& mat = matrix_memory;
  mat.resize(size, size);
  mat.reset(kInvalidTime);
  for (auto i = 0U; i != size; ++i) {
    auto it = lb;
    for (auto const& edge : fgraph[(lb + i)->second]) {  // precond.: sorted!
      while (it != ub && edge.target() != it->second) {
        ++it;
      }
      auto const j = static_cast<unsigned>(std::distance(lb, it));
      auto const from_l = location_idx_t{(lb + i)->second};
      auto const to_l = edge.target();
      mat(i, j) = std::max({tt.locations_.transfer_time_[from_l].count(),
                            tt.locations_.transfer_time_[to_l].count(),
                            u8_minutes{edge.duration()}.count()});
      print_dbg("INPUT: {} --{}={}--> {}\n", location{tt, from_l},
                edge.duration(), mat(i, j), location{tt, to_l});
    }
  }

  print_dbg("NIGIRI STATIONS:\n");
  for (auto i = 0U; i != size; ++i) {
    print_dbg("{} = {} \n", i, location{tt, location_idx_t{(lb + i)->second}});
  }

  print_dbg("NIGIRI MAT BEFORE\n{}\n", mat);

  floyd_warshall(mat);

  print_dbg("NIGIRI MAT AFTER\n{}", mat);

  print_dbg("\n\nOUTPUT\n");
  for (auto i = 0U; i < size; ++i) {
    auto const idx_a = std::next(lb, i)->second;
    auto const l_idx_a = location_idx_t{static_cast<unsigned>(idx_a)};

    for (auto j = 0U; j < size; ++j) {
      if (mat(i, j) == kInvalidTime || i == j) {
        continue;
      }

      auto const idx_b = std::next(lb, j)->second;
      auto const l_idx_b = location_idx_t{static_cast<unsigned>(idx_b)};

      if (mat(i, j) > std::numeric_limits<u8_minutes::rep>::max()) {
        std::cout << "ERROR: " << mat(i, j) << " > "
                  << std::numeric_limits<u8_minutes::rep>::max() << "\n";
        log(log_lvl::error, "loader.footpath", "footpath {}>256 too long",
            mat(i, j));
        continue;
      }

      auto const duration = std::max({u8_minutes{mat(i, j)},
                                      tt.locations_.transfer_time_[l_idx_a],
                                      tt.locations_.transfer_time_[l_idx_b]});
      tt.locations_.preprocessing_footpaths_out_[l_idx_a].emplace_back(
          l_idx_b, duration);
      tt.locations_.preprocessing_footpaths_in_[l_idx_b].emplace_back(l_idx_a,
                                                                      duration);
    }
  }
}

void transitivize_footpaths(timetable& tt) {
  auto const timer = scoped_timer{"building transitively closed foot graph"};

  auto const fgraph = get_footpath_graph(tt);

  auto components = find_components(fgraph);
  std::sort(begin(components), end(components));

  tt.locations_.preprocessing_footpaths_out_.clear();
  tt.locations_.preprocessing_footpaths_out_[location_idx_t{
      tt.locations_.src_.size() - 1}];
  tt.locations_.preprocessing_footpaths_in_.clear();
  tt.locations_.preprocessing_footpaths_in_[location_idx_t{
      tt.locations_.src_.size() - 1}];

  auto matrix_memory = make_flat_matrix(0, 0, std::uint16_t{0});
  utl::equal_ranges_linear(
      components,
      [](auto const& a, auto const& b) { return a.first == b.first; },
      [&](auto lb, auto ub) {
        process_component(tt, lb, ub, fgraph, matrix_memory);
      });
}

void add_links_to_and_between_children(timetable& tt) {
  mutable_fws_multimap<location_idx_t, footpath> fp_out;
  for (auto l = location_idx_t{0U};
       l != tt.locations_.preprocessing_footpaths_out_.size(); ++l) {
    for (auto const& fp : tt.locations_.preprocessing_footpaths_out_[l]) {
      for (auto const& neighbor_child : tt.locations_.children_[fp.target()]) {
        if (tt.locations_.types_[neighbor_child] ==
            location_type::kGeneratedTrack) {
          trace("  l -> neighbor child: {} -> {}: {}\n", location{tt, l},
                location{tt, neighbor_child}, fp.duration());
          fp_out[l].emplace_back(footpath{neighbor_child, fp.duration()});
        }

        for (auto const& child : tt.locations_.children_[l]) {
          if (tt.locations_.types_[child] == location_type::kGeneratedTrack) {
            trace("  child -> neighbor child: {} -> {}: {}\n",
                  location{tt, child}, location{tt, neighbor_child},
                  fp.duration());
            fp_out[child].emplace_back(footpath{neighbor_child, fp.duration()});
          }
        }
      }

      for (auto const& child : tt.locations_.children_[l]) {
        if (tt.locations_.types_[child] == location_type::kGeneratedTrack) {
          trace("  child -> neighbor child: {} -> {}: {}\n",
                location{tt, child}, location{tt, fp.target()}, fp.duration());
          fp_out[child].emplace_back(footpath{fp.target(), fp.duration()});
        }
      }
    }
  }

  for (auto l = location_idx_t{0U};
       l != tt.locations_.preprocessing_footpaths_out_.size(); ++l) {
    for (auto const& fp : fp_out[l]) {
      tt.locations_.preprocessing_footpaths_out_[l].emplace_back(fp);
    }
  }

  for (auto const [l, children] : utl::enumerate(tt.locations_.children_)) {
    auto const parent = location_idx_t{l};

    auto const t = tt.locations_.transfer_time_[parent];
    for (auto i = 0U; i != children.size(); ++i) {
      auto const child_i = children[i];
      if (tt.locations_.types_[child_i] != location_type::kGeneratedTrack) {
        continue;
      }
      tt.locations_.preprocessing_footpaths_out_[parent].emplace_back(child_i,
                                                                      t);
      tt.locations_.preprocessing_footpaths_out_[child_i].emplace_back(parent,
                                                                       t);
      for (auto j = 0U; j != children.size(); ++j) {
        if (i != j) {
          tt.locations_.preprocessing_footpaths_out_[child_i].emplace_back(
              children[j], t);
        }
      }
    }
  }
}

void write_footpaths(timetable& tt) {
  assert(tt.locations_.footpaths_out_.empty());
  assert(tt.locations_.footpaths_in_.empty());
  assert(tt.locations_.preprocessing_footpaths_out_.size() == tt.n_locations());
  assert(tt.locations_.preprocessing_footpaths_in_.size() == tt.n_locations());

  for (auto i = location_idx_t{0U}; i != tt.n_locations(); ++i) {
    tt.locations_.footpaths_out_.emplace_back(
        tt.locations_.preprocessing_footpaths_out_[i]);
  }

  for (auto i = location_idx_t{0U}; i != tt.n_locations(); ++i) {
    tt.locations_.footpaths_in_.emplace_back(
        tt.locations_.preprocessing_footpaths_in_[i]);
  }

  tt.locations_.preprocessing_footpaths_in_.clear();
  tt.locations_.preprocessing_footpaths_out_.clear();
}

void build_footpaths(timetable& tt) {
  add_links_to_and_between_children(tt);
  link_nearby_stations(tt);
  transitivize_footpaths(tt);
  write_footpaths(tt);
}

}  // namespace nigiri::loader
