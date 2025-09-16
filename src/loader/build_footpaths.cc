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
    utl::erase_duplicates(
        g[i], [](auto&& a, auto&& b) { return a.target_ < b.target_; },
        [](auto&& a, auto&& b) {
          return a.target_ == b.target_;
        });  // also sorts
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

void process_2_node_component(timetable& tt,
                              component const& c,
                              footgraph const& fgraph) {
  auto const l_idx_a = c.from_->l_;
  auto const l_idx_b = std::next(c.from_)->l_;
  auto const idx_a = to_idx(l_idx_a);
  auto const idx_b = to_idx(l_idx_b);

  if (!fgraph[idx_a].empty()) {
    auto const duration = std::max({u8_minutes{fgraph[idx_a].front().duration_},
                                    tt.locations_.transfer_time_[l_idx_a],
                                    tt.locations_.transfer_time_[l_idx_b]});
    tt.locations_.preprocessing_footpaths_out_[l_idx_a].emplace_back(l_idx_b,
                                                                     duration);
    tt.locations_.preprocessing_footpaths_in_[l_idx_b].emplace_back(l_idx_a,
                                                                    duration);
  }

  if (!fgraph[idx_b].empty()) {
    auto const duration = std::max({u8_minutes{fgraph[idx_b].front().duration_},
                                    tt.locations_.transfer_time_[l_idx_a],
                                    tt.locations_.transfer_time_[l_idx_b]});
    tt.locations_.preprocessing_footpaths_out_[l_idx_b].emplace_back(l_idx_a,
                                                                     duration);
    tt.locations_.preprocessing_footpaths_in_[l_idx_a].emplace_back(l_idx_b,
                                                                    duration);
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
          process_2_node_component(tt, c, fgraph);
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

        routing::dijkstra(c.graph_, dd.pq_, dd.dists_, max_footpath_length);
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
        auto const distance = geo::distance(tt.locations_.coordinates_[from_l],
                                            tt.locations_.coordinates_[to_l]);
        auto const adjusted_int = static_cast<int>(distance / kWalkSpeed / 60);
        if (adjusted_int > std::numeric_limits<u8_minutes::rep>::max()) {
          log(log_lvl::error, "loader.footpath.adjust",
              "too long after adjust: from={} at {}, to={} at {} {}>256",
              location{tt, from_l}, tt.locations_.coordinates_[from_l],
              location{tt, to_l}, tt.locations_.coordinates_[to_l],
              adjusted_int);
        } else {
          adjusted = u8_minutes{
              std::max(static_cast<duration_t::rep>(duration.count()),
                       static_cast<duration_t::rep>(adjusted_int))};
        }
      }

      tt.locations_.preprocessing_footpaths_out_[from_l].emplace_back(to_l,
                                                                      adjusted);
      tt.locations_.preprocessing_footpaths_in_[to_l].emplace_back(from_l,
                                                                   adjusted);
    }
  }
}

void add_links_to_and_between_children(timetable& tt) {
  auto fp_out = mutable_fws_multimap<location_idx_t, footpath>{};
  for (auto l = location_idx_t{0U};
       l != tt.locations_.preprocessing_footpaths_out_.size(); ++l) {
    for (auto const& fp : tt.locations_.preprocessing_footpaths_out_[l]) {
      for (auto const& neighbor_child : tt.locations_.children_[fp.target()]) {
        if (tt.locations_.types_[neighbor_child] ==
            location_type::kGeneratedTrack) {
          fp_out[l].emplace_back(footpath{neighbor_child, fp.duration()});
        }

        for (auto const& child : tt.locations_.children_[l]) {
          if (tt.locations_.types_[child] == location_type::kGeneratedTrack) {
            fp_out[child].emplace_back(footpath{neighbor_child, fp.duration()});
          }
        }
      }

      for (auto const& child : tt.locations_.children_[l]) {
        if (tt.locations_.types_[child] == location_type::kGeneratedTrack) {
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
  auto const cmp_fp_dur = [](auto const& a, auto const& b) {
    return a.duration_ < b.duration_;
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

void build_footpaths(timetable& tt, finalize_options const opt) {
  add_links_to_and_between_children(tt);
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
  sort_footpaths(tt);
  write_footpaths(tt);
}

}  // namespace nigiri::loader
