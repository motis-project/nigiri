#include "nigiri/loader/build_footpaths.h"

#include <mutex>
#include <optional>
#include <stack>

#include "utl/enumerate.h"
#include "utl/equal_ranges_linear.h"
#include "utl/erase_duplicates.h"
#include "utl/erase_if.h"
#include "utl/pairwise.h"
#include "utl/parallel_for.h"
#include "utl/progress_tracker.h"
#include "utl/verify.h"

#include "nigiri/loader/link_nearby_stations.h"
#include "nigiri/loader/merge_duplicates.h"
#include "nigiri/common/day_list.h"
#include "nigiri/constants.h"
#include "nigiri/logging.h"
#include "nigiri/routing/dijkstra.h"
#include "nigiri/rt/frun.h"
#include "nigiri/types.h"

namespace nigiri::loader {

using component_idx_t = cista::strong<std::uint32_t, struct component_idx_>;

// station_idx -> [footpath, ...]
using footgraph = vector<vector<footpath>>;

struct assignment {
  CISTA_FRIEND_COMPARABLE(assignment)

  component_idx_t c_;
  location_idx_t l_;
};
using component_vec = std::vector<assignment>;
using component_it = component_vec::iterator;

struct component {
  component(component_it from, component_it to) : from_{from}, to_{to} {}
  component_idx_t idx() const { return from_->c_; }
  bool invalid() const { return from_->c_ == component_idx_t::invalid(); }
  std::size_t size() const {
    return static_cast<std::size_t>(std::distance(from_, to_));
  }
  location_idx_t location_idx(std::size_t const i) const {
    assert(i < size());
    return std::next(from_, static_cast<component_it::difference_type>(i))->l_;
  }
  void verify() const {
    for (auto i = location_idx_t{0U}; i != graph_.size(); ++i) {
      auto const bucket = graph_[i];
      for (auto j = 0U; j != bucket.size(); ++j) {
        auto const fp = bucket[j];
        utl_verify(fp.target() < graph_.size(),
                   "fp.target={}, graph.size={}, i={}, j={}", fp.target(),
                   graph_.size(), i, j);
      }
    }
  }

  component_it from_, to_;
  vecvec<location_idx_t, footpath> graph_;
};

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
  }
  // mirror all edges: directional inputs (e.g. one-way transfers.txt rows)
  // must not break the component enumeration, which walks out-edges and
  // assumes mutual reachability -- an asymmetric edge can otherwise strand
  // nodes in bogus tiny components. the transitive closure treats every
  // edge as undirected anyway (build_component_graph inserts both
  // directions), so mirroring here does not change its result.
  for (auto i = 0U; i != tt.locations_.src_.size(); ++i) {
    for (auto const fp :
         tt.locations_.preprocessing_footpaths_out_[location_idx_t{i}]) {
      if (fp.target() != location_idx_t{i}) {
        g[to_idx(fp.target())].emplace_back(location_idx_t{i}, fp.duration());
      }
    }
  }
  for (auto i = 0U; i != tt.locations_.src_.size(); ++i) {
    utl::erase_duplicates(
        g[i],
        [](auto&& a, auto&& b) {
          return a.target_ != b.target_ ? a.target_ < b.target_
                                        : a.duration_ < b.duration_;
        },
        [](auto&& a, auto&& b) {
          return a.target_ == b.target_;
        });  // also sorts; keeps the shorter of asymmetric pair durations
  }
  return g;
}

std::vector<assignment> find_components(footgraph const& fgraph) {
  auto components = std::vector<assignment>{};
  components.resize(fgraph.size());
  std::generate(begin(components), end(components),
                [i = location_idx_t{0U}]() mutable {
                  return assignment{component_idx_t::invalid(), i++};
                });

  std::stack<uint32_t> stack;  // invariant: stack is empty
  for (auto i = 0U; i < fgraph.size(); ++i) {
    if (components[i].c_ != component_idx_t::invalid() || fgraph[i].empty()) {
      continue;
    }

    stack.emplace(i);
    while (!stack.empty()) {
      auto j = stack.top();
      stack.pop();

      if (components[j].c_ == i) {
        continue;
      }

      components[j].c_ = component_idx_t{i};
      for (auto const& f : fgraph[j]) {
        if (components[to_idx(f.target())].c_ != i) {
          stack.push(to_idx(f.target()));
        }
      }
    }
  }

  return components;
}

std::optional<u8_minutes> adjust_to_walk_speed(timetable const& tt,
                                               location_idx_t const a,
                                               location_idx_t const b,
                                               u8_minutes const duration) {

  constexpr auto const kMaxWalkDistance =
      std::numeric_limits<u8_minutes::rep>::max() * 60.0 * kWalkSpeed;

  auto const distance = geo::distance(tt.locations_.coordinates_[a],
                                      tt.locations_.coordinates_[b]);
  if (distance > kMaxWalkDistance) {
    log(log_lvl::error, "loader.footpath.adjust",
        "dropping footpath {} -> {}: {:.1f} km apart, not walkable",
        tt.locations_.ids_[a].view(), tt.locations_.ids_[b].view(),
        distance / 1000.0);
    return std::nullopt;
  }

  return u8_minutes{
      std::max(static_cast<duration_t::rep>(duration.count()),
               static_cast<duration_t::rep>(distance / kWalkSpeed / 60))};
}

void process_2_node_component(timetable& tt,
                              component const& c,
                              footgraph const& fgraph,
                              bool const adjust_footpaths) {
  auto const l_idx_a = c.from_->l_;
  auto const l_idx_b = std::next(c.from_)->l_;
  auto const idx_a = to_idx(l_idx_a);
  auto const idx_b = to_idx(l_idx_b);

  auto const write = [&](location_idx_t const from_l, location_idx_t const to_l,
                         u8_minutes const raw) {
    auto const duration = std::max({raw, tt.locations_.transfer_time_[l_idx_a],
                                    tt.locations_.transfer_time_[l_idx_b]});

    auto adjusted = duration;
    if (adjust_footpaths) {
      auto const a = adjust_to_walk_speed(tt, from_l, to_l, duration);
      if (!a.has_value()) {
        return;
      }
      adjusted = *a;
    }

    tt.locations_.preprocessing_footpaths_out_[from_l].emplace_back(to_l,
                                                                    adjusted);
    tt.locations_.preprocessing_footpaths_in_[to_l].emplace_back(from_l,
                                                                 adjusted);
  };

  if (!fgraph[idx_a].empty()) {
    write(l_idx_a, l_idx_b, u8_minutes{fgraph[idx_a].front().duration_});
  }

  if (!fgraph[idx_b].empty()) {
    write(l_idx_b, l_idx_a, u8_minutes{fgraph[idx_b].front().duration_});
  }
}

void build_component_graph(
    timetable& tt,
    component& c,
    footgraph const& fgraph,
    cista::raw::mutable_fws_multimap<location_idx_t, footpath>& tmp_graph) {
  if (c.invalid()) {
    return;
  }

  auto const size = c.size();

  utl::verify(size > 2, "invalid size [component={}], first={}", c.idx(),
              tt.locations_.ids_.at(c.from_->l_).view());

  tmp_graph.clear();
  for (auto i = 0U; i != size; ++i) {
    auto it = c.from_;
    auto const from_l = (c.from_ + i)->l_;

    for (auto const& edge : fgraph[to_idx(from_l)]) {
      while (it != c.to_ && edge.target() != it->l_) {
        ++it;  // precond.: component and fgraph are sorted!
      }
      auto const j = static_cast<unsigned>(std::distance(c.from_, it));
      assert(it != c.to_);

      auto const to_l = edge.target();
      auto const fp_duration = std::max({tt.locations_.transfer_time_[from_l],
                                         tt.locations_.transfer_time_[to_l],
                                         u8_minutes{edge.duration()}});

      tmp_graph[location_idx_t{i}].push_back(
          footpath{location_idx_t{j}, fp_duration});
      tmp_graph[location_idx_t{j}].push_back(
          footpath{location_idx_t{i}, fp_duration});
    }
  }

  for (auto n : tmp_graph) {
    utl::erase_duplicates(n);
    c.graph_.emplace_back(n);
  }

  c.verify();
}

void connect_components(timetable& tt,
                        std::uint16_t const max_footpath_length,
                        bool adjust_footpaths) {
  // ==========================
  // Find Connected Components
  // --------------------------
  auto const timer = scoped_timer{"building transitively closed foot graph"};

  auto const fgraph = get_footpath_graph(tt);
  auto assignments = find_components(fgraph);
  utl::sort(assignments);

  tt.locations_.preprocessing_footpaths_out_.clear();
  tt.locations_.preprocessing_footpaths_out_[location_idx_t{
      tt.locations_.src_.size() - 1}];
  tt.locations_.preprocessing_footpaths_in_.clear();
  tt.locations_.preprocessing_footpaths_in_[location_idx_t{
      tt.locations_.src_.size() - 1}];

  auto tmp_graph = cista::raw::mutable_fws_multimap<location_idx_t, footpath>{};
  auto components = std::vector<component>{};
  utl::equal_ranges_linear(
      assignments,
      [](assignment const& a, assignment const& b) { return a.c_ == b.c_; },
      [&](component_it const& from, component_it const& to) {
        auto c = component{from, to};
        if (c.invalid()) {
          return;
        } else if (c.size() == 2U) {
          process_2_node_component(tt, c, fgraph, adjust_footpaths);
        } else {
          build_component_graph(tt, c, fgraph, tmp_graph);
          components.emplace_back(std::move(c));
        }
      });

  // =====================
  // Shortest Path Search
  // ---------------------
  struct task {
    component const* c_;
    std::size_t idx_;
    std::vector<footpath> results_;
  };

  struct dijkstra_data {
    std::vector<routing::label::dist_t> dists_;
    dial<routing::label, routing::get_bucket> pq_;
  };

  constexpr auto const kUnreachable =
      std::numeric_limits<routing::label::dist_t>::max();

  auto tasks = std::vector<task>{};
  for (auto const& c : components) {
    for (auto i = 0U; i != c.size(); ++i) {
      tasks.push_back(task{.c_ = &c, .idx_ = i, .results_ = {}});
    }
  }

  utl::parallel_for_run_threadlocal<dijkstra_data>(
      tasks.size(), [&](dijkstra_data& dd, std::size_t const idx) {
        auto const& c = *tasks[idx].c_;
        auto const& node_idx = tasks[idx].idx_;

        dd.pq_.clear();
        dd.pq_.n_buckets(max_footpath_length);
        dd.pq_.push(routing::label{location_idx_t{node_idx}, 0U});

        dd.dists_.resize(c.size());
        utl::fill(dd.dists_,
                  std::numeric_limits<routing::label::dist_t>::max());
        dd.dists_[node_idx] = 0U;

        bitvec_map<location_idx_t> const* has_rt = nullptr;
        vecvec<location_idx_t, footpath> const* rt = nullptr;
        routing::dijkstra(c.graph_, has_rt, rt, dd.pq_, dd.dists_,
                          max_footpath_length);
        for (auto const [target, duration] : utl::enumerate(dd.dists_)) {
          if (duration == kUnreachable || target == node_idx) {
            continue;
          }
          tasks[idx].results_.emplace_back(
              footpath{c.location_idx(target), duration_t{duration}});
        }
      });

  // ================
  // Write Footpaths
  // ----------------
  for (auto const& t : tasks) {
    auto const& c = *t.c_;
    auto const from_l = c.location_idx(t.idx_);

    for (auto const& fp : t.results_) {
      auto const to_l = fp.target();

      auto const duration =
          std::max({std::chrono::duration_cast<u8_minutes>(fp.duration()),
                    tt.locations_.transfer_time_[from_l],
                    tt.locations_.transfer_time_[to_l]});

      auto adjusted = duration;
      if (adjust_footpaths) {
        auto const a = adjust_to_walk_speed(tt, from_l, to_l, duration);
        if (!a.has_value()) {
          continue;
        }
        adjusted = *a;
      }

      tt.locations_.preprocessing_footpaths_out_[from_l].emplace_back(to_l,
                                                                      adjusted);
      tt.locations_.preprocessing_footpaths_in_[to_l].emplace_back(from_l,
                                                                   adjusted);
    }
  }
}

bool is_generated(location_type const t) {
  return t == location_type::kGeneratedTrack || t == location_type::kVirt;
}

// Walking transfers between equivalent stops (e.g. GTFS same-name / nearby /
// parent-child stops, HRDF meta stations): the loaders only collect the
// equivalences, the beeline footpaths are derived here for pairs without a
// footpath from the input data. With street routing, these are later replaced
// by routed footpaths (except where a transfer rule fixes the duration).
void add_missing_equivalence_footpaths(timetable& tt) {
  auto const add_if_not_exists = [](auto bucket, footpath const fp) {
    for (auto const existing : bucket) {
      if (existing.target() == fp.target()) {
        return;
      }
    }
    bucket.emplace_back(fp);
  };

  for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
    if (tt.locations_.equivalences_[l].empty()) {
      continue;
    }
    auto const& pos = tt.locations_.coordinates_[l];
    auto const dist_lng_degrees = geo::approx_distance_lng_degrees(pos);
    for (auto const eq : tt.locations_.equivalences_[l]) {
      auto const dist = std::sqrt(geo::approx_squared_distance(
          pos, tt.locations_.coordinates_[eq], dist_lng_degrees));
      auto const duration = duration_t{
          std::max(2, static_cast<int>(std::ceil((dist / kWalkSpeed) / 60.0)))};

      if (duration > footpath::kMaxDuration) {
        continue;
      }

      add_if_not_exists(tt.locations_.preprocessing_footpaths_out_[l],
                        {eq, duration});
      add_if_not_exists(tt.locations_.preprocessing_footpaths_in_[eq],
                        {l, duration});
      add_if_not_exists(tt.locations_.preprocessing_footpaths_out_[eq],
                        {l, duration});
      add_if_not_exists(tt.locations_.preprocessing_footpaths_in_[l],
                        {eq, duration});
    }
  }
}

void add_links_to_and_between_children(timetable& tt) {
  // propagated copies involving kVirt members are never materialized:
  // same-station pairs are covered by the transfer rule matrix (derived
  // via the door bits, or materialized as deviating cells), cross-station
  // pairs are uniform duplicates of the base edge and derived at query
  // time through the walk channel (raptor update_footpaths).
  auto const implicit_pair = [&](location_idx_t const a,
                                 location_idx_t const b) {
    return tt.locations_.types_[a] == location_type::kVirt ||
           tt.locations_.types_[b] == location_type::kVirt;
  };

  auto fp_out = mutable_fws_multimap<location_idx_t, footpath>{};
  for (auto l = location_idx_t{0U};
       l != tt.locations_.preprocessing_footpaths_out_.size(); ++l) {
    for (auto const& fp : tt.locations_.preprocessing_footpaths_out_[l]) {
      for (auto const& neighbor_child : tt.locations_.children_[fp.target()]) {
        if (is_generated(tt.locations_.types_[neighbor_child]) &&
            !implicit_pair(l, neighbor_child)) {
          fp_out[l].emplace_back(footpath{neighbor_child, fp.duration()});
        }

        for (auto const& child : tt.locations_.children_[l]) {
          if (is_generated(tt.locations_.types_[child]) &&
              !implicit_pair(child, neighbor_child)) {
            fp_out[child].emplace_back(footpath{neighbor_child, fp.duration()});
          }
        }
      }

      for (auto const& child : tt.locations_.children_[l]) {
        if (is_generated(tt.locations_.types_[child]) &&
            !implicit_pair(child, fp.target())) {
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

void sort_footpaths(timetable& tt) {
  // sorted by target: consumers can set-operate footpaths against other
  // target-sorted sequences (e.g. the transfer-group expansion diffing
  // against children_)
  auto const cmp_fp_dur = [](auto const& a, auto const& b) {
    return std::tie(a.target_, a.duration_) < std::tie(b.target_, b.duration_);
  };
  for (auto i = location_idx_t{0U}; i != tt.n_locations(); ++i) {
    utl::sort(tt.locations_.preprocessing_footpaths_out_[i], cmp_fp_dur);
  }
  for (auto i = location_idx_t{0U}; i != tt.n_locations(); ++i) {
    utl::sort(tt.locations_.preprocessing_footpaths_in_[i], cmp_fp_dur);
  }
}

void write_footpaths(timetable& tt) {
  assert(tt.locations_.footpaths_out_.size() == kNProfiles);
  assert(tt.locations_.footpaths_in_.size() == kNProfiles);
  assert(tt.locations_.preprocessing_footpaths_out_.size() == tt.n_locations());
  assert(tt.locations_.preprocessing_footpaths_in_.size() == tt.n_locations());

  profile_idx_t const prf_idx{0};

  for (auto i = location_idx_t{0U}; i != tt.n_locations(); ++i) {
    tt.locations_.footpaths_out_[prf_idx].emplace_back(
        tt.locations_.preprocessing_footpaths_out_[i]);
  }

  for (auto i = location_idx_t{0U}; i != tt.n_locations(); ++i) {
    tt.locations_.footpaths_in_[prf_idx].emplace_back(
        tt.locations_.preprocessing_footpaths_in_[i]);
  }

  tt.locations_.preprocessing_footpaths_in_.clear();
  tt.locations_.preprocessing_footpaths_out_.clear();
}

// Overwrite/insert the directed same-station transfer edges emitted from
// transfer time rules. These are authoritative: any transitively computed or
// generic parent/child footpath between the same pair is replaced. The rule
// duration is kept as-is (a rule fixes the transfer time), but unwalkable
// pairs (wormholes, see adjust_to_walk_speed) are still dropped: a transfer
// time between stops hundreds of km apart is a data error, not a rule.
void apply_transfer_rules(timetable& tt, bool const adjust_footpaths) {
  auto const n = location_idx_t{
      std::min(static_cast<location_idx_t::value_t>(
                   tt.locations_.transfer_rule_fps_.size()),
               static_cast<location_idx_t::value_t>(tt.n_locations()))};
  for (auto l = location_idx_t{0U}; l != n; ++l) {
    for (auto const fp : tt.locations_.transfer_rule_fps_[l]) {
      if (adjust_footpaths &&
          !adjust_to_walk_speed(tt, l, fp.target(), fp.duration())
               .has_value()) {
        continue;
      }
      auto replaced = false;
      for (auto& existing : tt.locations_.preprocessing_footpaths_out_[l]) {
        if (existing.target() == fp.target()) {
          existing = fp;
          replaced = true;
        }
      }
      if (!replaced) {
        tt.locations_.preprocessing_footpaths_out_[l].emplace_back(fp);
      }

      auto const reverse = footpath{l, fp.duration()};
      auto replaced_in = false;
      for (auto& existing :
           tt.locations_.preprocessing_footpaths_in_[fp.target()]) {
        if (existing.target() == l) {
          existing = reverse;
          replaced_in = true;
        }
      }
      if (!replaced_in) {
        tt.locations_.preprocessing_footpaths_in_[fp.target()].emplace_back(
            reverse);
      }
    }
  }
}

// build the hub edge lists for one profile from the (final) footpaths of
// that profile: one broadcast + collect + walk-out + walk-in hub per
// transfer-group station (station or platform with kVirt children). all
// classification is derived from the materialized cells: a member is
// "loud" if its own transfer time exceeds the group default (it never
// gathers -- its echo would undercut the route scan's same-stop value),
// row/col-clean if it has no same-station cell above the default.
void build_hubs(timetable& tt, profile_idx_t const prf) {
  auto const n = tt.n_locations();
  auto const& fps_out = tt.locations_.footpaths_out_[prf];
  auto const& fps_in = tt.locations_.footpaths_in_[prf];

  auto in = vecvec<hub_idx_t, footpath>{};
  auto out = vecvec<hub_idx_t, footpath>{};
  auto in_t = std::vector<std::vector<hub_ref>>(n);
  auto out_t = std::vector<std::vector<hub_ref>>(n);

  auto next_hub = hub_idx_t{0U};
  auto const add_hub = [&](std::vector<footpath> const& hin,
                           std::vector<footpath> const& hout) {
    if (hin.empty() || hout.empty()) {
      return;
    }
    auto const h = next_hub++;
    in.emplace_back(hin);
    out.emplace_back(hout);
    for (auto const& e : hin) {
      in_t[to_idx(e.target())].push_back({h, e.duration()});
    }
    for (auto const& e : hout) {
      out_t[to_idx(e.target())].push_back({h, e.duration()});
    }
  };

  auto is_group_parent = std::vector<bool>(n, false);
  for (auto l = location_idx_t{0U}; l != location_idx_t{n}; ++l) {
    if (tt.locations_.types_[l] == location_type::kVirt) {
      is_group_parent[to_idx(tt.locations_.parents_[l])] = true;
    }
  }

  for (auto l = location_idx_t{0U}; l != location_idx_t{n}; ++l) {
    if (!is_group_parent[to_idx(l)]) {
      continue;
    }
    auto members = std::vector<location_idx_t>{l};
    for (auto const c : tt.locations_.children_[l]) {
      if (tt.locations_.types_[c] == location_type::kVirt) {
        members.push_back(c);
      }
    }
    auto const d = tt.locations_.transfer_time_[l].count();
    auto const same_station = [&](location_idx_t const t) {
      return t == l || tt.locations_.parents_[t] == l;
    };
    // cleanliness only counts cells between group members: the transfer
    // hubs never derive a pair targeting a non-member, so only member
    // cells can be undercut. real stops (GTFS) can carry same-station
    // edges to non-member children (e.g. equivalence beelines) that must
    // not flip the classification the loader elided against.
    auto const is_member = [&](location_idx_t const t) {
      return t == l || (tt.locations_.parents_[t] == l &&
                        tt.locations_.types_[t] == location_type::kVirt);
    };

    auto bcast_in = std::vector<footpath>{};
    auto coll_in = std::vector<footpath>{};
    auto bcast_out = std::vector<footpath>{};
    auto coll_out = std::vector<footpath>{};
    auto walk_in_out = std::vector<footpath>{};
    auto walk_out_in = std::vector<footpath>{};
    for (auto const m : members) {
      auto const quiet = m == l || tt.locations_.transfer_time_[m].count() <= d;
      auto const clean = [&](auto const& fps) {
        for (auto const& fp : fps[m]) {
          if (is_member(fp.target()) && fp.duration().count() > d) {
            return false;
          }
        }
        return true;
      };
      auto const row_clean = clean(fps_out);
      auto const col_clean = clean(fps_in);
      if (quiet && row_clean) {
        bcast_in.emplace_back(m, duration_t{0});
      } else if (quiet && col_clean) {
        coll_in.emplace_back(m, duration_t{0});
      }
      bcast_out.emplace_back(m, duration_t{d});
      if (col_clean) {
        coll_out.emplace_back(m, duration_t{d});
      }
      walk_out_in.emplace_back(m, duration_t{0});
      if (tt.locations_.types_[m] == location_type::kVirt) {
        walk_in_out.emplace_back(m, duration_t{0});
      }
    }

    auto walk_out_out = std::vector<footpath>{};
    for (auto const& fp : fps_out[l]) {
      auto const t = fp.target();
      if (same_station(t)) {
        continue;  // same-station cells are the transfer channel
      }
      walk_out_out.emplace_back(t, fp.duration());
      for (auto const c : tt.locations_.children_[t]) {
        auto const ct = tt.locations_.types_[c];
        if (ct == location_type::kVirt ||
            ct == location_type::kGeneratedTrack) {
          walk_out_out.emplace_back(c, fp.duration());
        }
      }
    }
    auto walk_in_in = std::vector<footpath>{};
    for (auto const& fp : fps_in[l]) {
      if (same_station(fp.target())) {
        continue;
      }
      walk_in_in.emplace_back(fp.target(), fp.duration());
    }

    add_hub(bcast_in, bcast_out);
    add_hub(coll_in, coll_out);
    add_hub(walk_out_in, walk_out_out);
    add_hub(walk_in_in, walk_in_out);
  }

  auto in_by_loc = vecvec<location_idx_t, hub_ref>{};
  auto out_by_loc = vecvec<location_idx_t, hub_ref>{};
  for (auto l = 0U; l != n; ++l) {
    in_by_loc.emplace_back(in_t[l]);
    out_by_loc.emplace_back(out_t[l]);
  }
  tt.locations_.hub_in_[prf] = std::move(in);
  tt.locations_.hub_out_[prf] = std::move(out);
  tt.locations_.hub_in_by_loc_[prf] = std::move(in_by_loc);
  tt.locations_.hub_out_by_loc_[prf] = std::move(out_by_loc);
}

void build_footpaths(timetable& tt, finalize_options const opt) {
  add_missing_equivalence_footpaths(tt);
  add_links_to_and_between_children(tt);

  if (!opt.transitive_footpaths_) {
    // direct footpaths only (e.g. single HRDF input): no transitive closure,
    // input footpaths + parent/child links + transfer rule edges
    apply_transfer_rules(tt, opt.adjust_footpaths_);

    // rebuild incoming footpaths as the mirror of the outgoing footpaths
    tt.locations_.preprocessing_footpaths_in_.clear();
    tt.locations_.preprocessing_footpaths_in_[location_idx_t{
        tt.locations_.src_.size() - 1}];
    for (auto l = location_idx_t{0U};
         l != tt.locations_.preprocessing_footpaths_out_.size(); ++l) {
      for (auto const fp : tt.locations_.preprocessing_footpaths_out_[l]) {
        tt.locations_.preprocessing_footpaths_in_[fp.target()].emplace_back(
            l, fp.duration());
      }
    }
    tt.locations_.preprocessing_footpaths_out_[location_idx_t{
        tt.locations_.src_.size() - 1}];

    sort_footpaths(tt);
    write_footpaths(tt);
    build_hubs(tt, kDefaultProfile);
    return;
  }

  link_nearby_stations(tt);
  if (opt.merge_dupes_intra_src_ || opt.merge_dupes_inter_src_) {
    for (auto l = location_idx_t{0U}; l != tt.n_locations(); ++l) {
      if (tt.locations_.src_[l] == source_idx_t{source_idx_t::invalid()}) {
        continue;
      }
      for (auto e : tt.locations_.equivalences_[l]) {
        if (tt.locations_.src_[e] == source_idx_t{source_idx_t::invalid()} ||
            (!opt.merge_dupes_intra_src_ &&
             tt.locations_.src_[l] == tt.locations_.src_[e]) ||
            (!opt.merge_dupes_inter_src_ &&
             tt.locations_.src_[l] != tt.locations_.src_[e])) {
          continue;
        }

        find_duplicates(tt, l, e);
      }
    }
  }
  connect_components(tt, opt.max_footpath_length_, opt.adjust_footpaths_);
  apply_transfer_rules(tt, opt.adjust_footpaths_);
  sort_footpaths(tt);
  write_footpaths(tt);
  build_hubs(tt, kDefaultProfile);
}

}  // namespace nigiri::loader
