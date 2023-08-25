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
#include "nigiri/common/day_list.h"
#include "nigiri/logging.h"
#include "nigiri/rt/frun.h"
#include "utl/erase_duplicates.h"
#include "utl/erase_if.h"
#include "utl/pairwise.h"
#include "utl/progress_tracker.h"

// #define NIGIRI_BUILD_FOOTPATHS_DEBUG
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
using match_set_t = hash_set<pair<location_idx_t, location_idx_t>>;

pair<location_idx_t, location_idx_t> make_match_pair(location_idx_t const a,
                                                     location_idx_t const b) {
  return {std::min(a, b), std::max(a, b)};
}

unsigned get_delta(timetable const& tt,
                   route_idx_t const a_route,
                   route_idx_t const b_route,
                   transport_idx_t const a,
                   transport_idx_t const b) {
  auto const size = tt.route_location_seq_[a_route].size();

  auto delta = 0U;
  for (auto i = stop_idx_t{0U}; i != size; ++i) {
    if (i != 0U) {
      delta += static_cast<unsigned>(
          std::abs(tt.event_mam(a_route, a, i, event_type::kArr).count() -
                   tt.event_mam(b_route, b, i, event_type::kArr).count()));
    }
    if (i != size - 1U) {
      delta += static_cast<unsigned>(
          std::abs(tt.event_mam(a_route, a, i, event_type::kDep).count() -
                   tt.event_mam(b_route, b, i, event_type::kDep).count()));
    }
  }

  return delta;
}

bool merge(timetable& tt,
           stop_idx_t const size,
           transport_idx_t const a,
           transport_idx_t const b) {
  auto const bf_a = tt.bitfields_[tt.transport_traffic_days_[a]];
  auto const bf_b = tt.bitfields_[tt.transport_traffic_days_[b]];
  if ((bf_a & bf_b).none()) {
    return false;
  }

  if ((bf_a & bf_b) == bf_a) {
    tt.transport_traffic_days_[b] = bitfield_idx_t{0U};  // disable trip 'b'

    for (auto const merged_trips_idx_b : tt.transport_to_trip_section_[b]) {
      for (auto const b_trp : tt.merged_trips_[merged_trips_idx_b]) {
        for (auto& [t, range] : tt.trip_transport_ranges_[b_trp]) {
          if (t == b) {
            t = a;  // replace b with a in b's trip transport ranges
          }
        }
      }
    }
  } else {
    tt.transport_traffic_days_[a] = tt.register_bitfield(bf_a | bf_b);
    tt.transport_traffic_days_[b] = tt.register_bitfield(bf_b & ~bf_a);

    hash_set<trip_idx_t> b_trips;
    for (auto const merged_trips_idx_b : tt.transport_to_trip_section_[b]) {
      for (auto const b_trp : tt.merged_trips_[merged_trips_idx_b]) {
        for (auto& [t, range] : tt.trip_transport_ranges_[b_trp]) {
          if (t == b) {
            b_trips.emplace(b_trp);
          }
        }
      }
    }

    for (auto const b_trp : b_trips) {
      tt.trip_transport_ranges_[b_trp].push_back(
          transport_range_t{a, {0U, size}});
    }
  }

  return true;
}

unsigned find_duplicates(timetable& tt,
                         match_set_t const& matches,
                         location_idx_t const a,
                         location_idx_t const b) {
  auto merged = 0U;
  for (auto const a_route : tt.location_routes_[a]) {
    auto const first_stop_a_route =
        stop{tt.route_location_seq_[a_route].front()}.location_idx();
    if (first_stop_a_route != a) {
      continue;
    }

    auto const a_loc_seq = tt.route_location_seq_[a_route];
    for (auto const& b_route : tt.location_routes_[b]) {
      auto const first_stop_b_route =
          stop{tt.route_location_seq_[b_route].front()}.location_idx();
      if (first_stop_b_route != b) {
        continue;
      }

      auto const b_loc_seq = tt.route_location_seq_[b_route];
      if (a_loc_seq.size() != b_loc_seq.size()) {
        continue;
      }

      for (auto const [x, y] : utl::zip(a_loc_seq, b_loc_seq)) {
        if (!matches.contains(make_match_pair(stop{x}.location_idx(),
                                              stop{y}.location_idx()))) {
          continue;
        }
      }

      auto const a_transport_range = tt.route_transport_ranges_[a_route];
      auto const b_transport_range = tt.route_transport_ranges_[b_route];
      auto a_t = begin(a_transport_range), b_t = begin(b_transport_range);

      while (a_t != end(a_transport_range) && b_t != end(b_transport_range)) {
        auto const time_a = tt.event_mam(a_route, *a_t, 0U, event_type::kDep);
        auto const time_b = tt.event_mam(b_route, *b_t, 0U, event_type::kDep);

        if (time_a == time_b) {
          if (get_delta(tt, a_route, b_route, *a_t, *b_t) < a_loc_seq.size()) {
            if (merge(tt, static_cast<stop_idx_t>(a_loc_seq.size()), *a_t,
                      *b_t)) {
              ++merged;
            }
          }
          ++a_t;
          ++b_t;
        } else if (time_a < time_b) {
          ++a_t;
        } else /* time_a > time_b */ {
          ++b_t;
        }
      }
    }
  }
  return merged;
}

void link_nearby_stations(timetable& tt, bool const merge_duplicates) {
  constexpr auto const kLinkNearbyMaxDistance = 300;  // [m];

  auto const locations_rtree =
      geo::make_point_rtree(tt.locations_.coordinates_);

  auto matches = match_set_t{};
  for (auto l_from_idx = location_idx_t{0U};
       l_from_idx != tt.locations_.src_.size(); ++l_from_idx) {
    auto const from_pos = tt.locations_.coordinates_[l_from_idx];
    if (std::abs(from_pos.lat_) < 2.0 && std::abs(from_pos.lng_) < 2.0) {
      continue;
    }

    auto const from_src = tt.locations_.src_[l_from_idx];
    if (from_src == source_idx_t::invalid()) {
      continue;  // no dummy stations
    }

    for (auto const& to_idx :
         locations_rtree.in_radius(from_pos, kLinkNearbyMaxDistance)) {
      auto const l_to_idx = location_idx_t{static_cast<unsigned>(to_idx)};
      if (l_from_idx == l_to_idx) {
        continue;
      }

      auto const to_src = tt.locations_.src_[l_to_idx];
      auto const to_pos = tt.locations_.coordinates_[l_to_idx];
      if (to_src == source_idx_t::invalid() /* no dummy stations */
          || from_src == to_src /* don't short-circuit */) {
        continue;
      }

      auto const from_transfer_time =
          duration_t{tt.locations_.transfer_time_[l_from_idx]};
      auto const to_transfer_time =
          duration_t{tt.locations_.transfer_time_[l_to_idx]};
      auto const walk_duration = duration_t{static_cast<unsigned>(
          std::round(geo::distance(from_pos, to_pos) / (60 * kWalkSpeed)))};
      auto const duration =
          std::max({from_transfer_time, to_transfer_time, walk_duration});

      tt.locations_.preprocessing_footpaths_out_[l_from_idx].emplace_back(
          l_to_idx, duration);
      tt.locations_.preprocessing_footpaths_in_[l_to_idx].emplace_back(
          l_from_idx, duration);
      tt.locations_.equivalences_[l_from_idx].emplace_back(l_to_idx);

      if (merge_duplicates) {
        matches.emplace(make_match_pair(l_from_idx, l_to_idx));
      }
    }
  }

  if (merge_duplicates) {
    for (auto const& [a, b] : matches) {
      find_duplicates(tt, matches, a, b);
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
                       matrix<std::uint16_t>& matrix_memory,
                       bool const adjust_footpaths) {
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
        log(log_lvl::error, "loader.footpath", "footpath {}>256 too long",
            mat(i, j));
        continue;
      }

      auto const duration = std::max({u8_minutes{mat(i, j)},
                                      tt.locations_.transfer_time_[l_idx_a],
                                      tt.locations_.transfer_time_[l_idx_b]});

      auto adjusted = duration;
      if (adjust_footpaths) {
        auto const distance =
            geo::distance(tt.locations_.coordinates_[l_idx_a],
                          tt.locations_.coordinates_[l_idx_b]);
        auto const adjusted_int =
            std::max(static_cast<duration_t::rep>(duration.count()),
                     static_cast<duration_t::rep>(distance / kWalkSpeed / 60));
        if (adjusted_int > std::numeric_limits<u8_minutes::rep>::max()) {
          log(log_lvl::error, "loader.footpath.adjust",
              "too long after adjust: {}>256", adjusted_int);
        }
        adjusted = u8_minutes{adjusted_int};
      }

      tt.locations_.preprocessing_footpaths_out_[l_idx_a].emplace_back(
          l_idx_b, adjusted);
      tt.locations_.preprocessing_footpaths_in_[l_idx_b].emplace_back(l_idx_a,
                                                                      adjusted);
    }
  }
}

void transitivize_footpaths(timetable& tt, bool const adjust_footpaths) {
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
        process_component(tt, lb, ub, fgraph, matrix_memory, adjust_footpaths);
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

void build_footpaths(timetable& tt,
                     bool const adjust_footpaths,
                     bool const merge_duplicates) {
  add_links_to_and_between_children(tt);
  link_nearby_stations(tt, merge_duplicates);
  transitivize_footpaths(tt, adjust_footpaths);
  write_footpaths(tt);
}

}  // namespace nigiri::loader
