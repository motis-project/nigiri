#include "nigiri/routing/component_graph.h"

#include <algorithm>
#include <functional>
#include <numeric>

#include "utl/enumerate.h"

#include "nigiri/timetable.h"

namespace nigiri::routing {

namespace {

struct union_find {
  explicit union_find(std::size_t const n) : parent_(n) {
    std::iota(begin(parent_), end(parent_), 0U);
  }

  std::uint32_t find(std::uint32_t x) {
    while (parent_[x] != x) {
      parent_[x] = parent_[parent_[x]];  // path halving
      x = parent_[x];
    }
    return x;
  }

  void merge(std::uint32_t const a, std::uint32_t const b) {
    auto const ra = find(a);
    auto const rb = find(b);
    if (ra != rb) {
      parent_[std::max(ra, rb)] = std::min(ra, rb);
    }
  }

  std::vector<std::uint32_t> parent_;
};

std::uint16_t sat_u16(int const x) {
  return static_cast<std::uint16_t>(
      std::clamp(x, 0, static_cast<int>(component_lb::kUnreachableTt)));
}

}  // namespace

component_graph build_component_graph(timetable const& tt) {
  auto const n = tt.n_locations();

  // --- components: union over footpaths (all profiles) + parent/child
  auto uf = union_find{n};
  for (auto p = profile_idx_t{0U}; p != kNProfiles; ++p) {
    for (auto const& fps :
         {std::cref(tt.locations_.footpaths_out_[p]),
          std::cref(tt.locations_.footpaths_in_[p])}) {
      if (fps.get().size() != n) {
        continue;  // profile not built
      }
      for (auto l = location_idx_t{0U}; l != n; ++l) {
        for (auto const& fp : fps.get()[l]) {
          uf.merge(to_idx(l), to_idx(fp.target()));
        }
      }
    }
  }
  for (auto l = location_idx_t{0U}; l != n; ++l) {
    if (tt.locations_.parents_[l] != location_idx_t::invalid()) {
      uf.merge(to_idx(l), to_idx(tt.locations_.parents_[l]));
    }
  }

  auto g = component_graph{};

  // densify component ids
  auto comp_of_root =
      std::vector<component_idx_t>(n, component_idx_t::invalid());
  g.location_component_.resize(n);
  auto n_components = std::uint32_t{0U};
  for (auto l = 0U; l != n; ++l) {
    auto const root = uf.find(l);
    if (comp_of_root[root] == component_idx_t::invalid()) {
      comp_of_root[root] = component_idx_t{n_components++};
    }
    g.location_component_[location_idx_t{l}] = comp_of_root[root];
  }
  g.n_components_ = n_components;

  // --- compressed component sequences + fastest-trip segment durations,
  //     deduped over routes sharing the same component sequence
  auto seq_map = hash_map<std::vector<component_idx_t>, std::uint32_t>{};
  auto seqs = std::vector<std::vector<component_idx_t>>{};
  auto durations = std::vector<std::vector<std::uint16_t>>{};

  auto comp_seq = std::vector<component_idx_t>{};
  auto first_stop = std::vector<stop_idx_t>{};  // first stop idx of each run
  auto last_stop = std::vector<stop_idx_t>{};  // last stop idx of each run
  auto segs = std::vector<std::uint16_t>{};
  for (auto r = route_idx_t{0U}; r != tt.n_routes(); ++r) {
    auto const loc_seq = tt.route_location_seq_[r];

    comp_seq.clear();
    first_stop.clear();
    last_stop.clear();
    for (auto const [i, s] : utl::enumerate(loc_seq)) {
      auto const c = g.location_component_[stop{s}.location_idx()];
      if (comp_seq.empty() || comp_seq.back() != c) {
        comp_seq.push_back(c);
        first_stop.push_back(static_cast<stop_idx_t>(i));
      }
      last_stop.resize(comp_seq.size());
      last_stop.back() = static_cast<stop_idx_t>(i);
    }
    if (comp_seq.size() < 2U) {
      continue;  // route entirely inside one component
    }

    // segment p: depart at the last stop of run p, arrive at the first stop
    // of run p+1 (within a run, boarding/alighting is free anyway)
    segs.clear();
    for (auto p = 0U; p != comp_seq.size() - 1U; ++p) {
      auto min_dur = static_cast<int>(component_lb::kUnreachableTt);
      for (auto const t : tt.route_transport_ranges_[r]) {
        auto const dep =
            tt.event_mam(r, t, last_stop[p], event_type::kDep).count();
        auto const arr =
            tt.event_mam(r, t, first_stop[p + 1U], event_type::kArr).count();
        min_dur = std::min(min_dur, arr - dep);
      }
      segs.push_back(sat_u16(min_dur));
    }

    if (auto const it = seq_map.find(comp_seq); it != end(seq_map)) {
      auto& d = durations[it->second];
      for (auto p = 0U; p != segs.size(); ++p) {
        d[p] = std::min(d[p], segs[p]);
      }
    } else {
      seq_map.emplace(comp_seq, static_cast<std::uint32_t>(seqs.size()));
      seqs.push_back(comp_seq);
      durations.push_back(segs);
    }
  }

  for (auto i = 0U; i != seqs.size(); ++i) {
    g.seqs_.emplace_back(seqs[i]);
    g.durations_.emplace_back(durations[i]);
  }

  // --- inverted index component -> component routes
  auto comp_route_lists =
      std::vector<std::vector<comp_route_idx_t>>(n_components);
  for (auto const [i, seq] : utl::enumerate(seqs)) {
    auto uniq = seq;
    utl::sort(uniq);
    uniq.erase(std::unique(begin(uniq), end(uniq)), end(uniq));
    for (auto const c : uniq) {
      comp_route_lists[to_idx(c)].push_back(
          comp_route_idx_t{static_cast<std::uint32_t>(i)});
    }
  }
  for (auto const& l : comp_route_lists) {
    g.comp_routes_.emplace_back(l);
  }

  return g;
}

component_lb compute_component_lb(
    component_graph const& g,
    direction const dir,
    std::vector<std::pair<component_idx_t, std::uint16_t>> const& seeds,
    unsigned const max_rounds) {
  constexpr auto const kInf = component_lb::kUnreachableTt;
  auto const fwd = dir == direction::kForward;
  auto const n = g.n_components_;
  auto const n_routes = static_cast<std::uint32_t>(g.seqs_.size());

  auto lb = component_lb{};
  lb.tt_.assign(n, kInf);
  lb.ic_.assign(n, component_lb::kUnreachableIc);

  auto comp_marked = std::vector<std::uint8_t>(n, 0U);
  auto route_marked = std::vector<std::uint8_t>(n_routes, 0U);
  auto prev = std::vector<std::uint16_t>{};

  for (auto const& [c, d] : seeds) {
    if (d < lb.tt_[to_idx(c)]) {
      lb.tt_[to_idx(c)] = d;
      lb.ic_[to_idx(c)] = 0U;
      comp_marked[to_idx(c)] = 1U;
    }
  }

  auto const sat_add = [](std::uint16_t const a, std::uint16_t const b) {
    return (a == kInf || b == kInf)
               ? kInf
               : sat_u16(static_cast<int>(a) + static_cast<int>(b));
  };

  for (auto k = 1U; k <= max_rounds; ++k) {
    // mark routes of components improved in the previous round
    std::fill(begin(route_marked), end(route_marked), 0U);
    auto any_route = false;
    for (auto c = component_idx_t{0U}; c != n; ++c) {
      if (comp_marked[to_idx(c)]) {
        for (auto const cr : g.comp_routes_[c]) {
          route_marked[to_idx(cr)] = 1U;
          any_route = true;
        }
      }
    }
    if (!any_route) {
      break;
    }
    std::fill(begin(comp_marked), end(comp_marked), 0U);
    prev = lb.tt_;  // exact boarding counts: read round k-1 values only

    auto const relax = [&](component_idx_t const c, std::uint16_t const val) {
      if (val < lb.tt_[to_idx(c)]) {
        if (lb.tt_[to_idx(c)] == kInf) {
          lb.ic_[to_idx(c)] = static_cast<std::uint8_t>(std::min(k, 254U));
        }
        lb.tt_[to_idx(c)] = val;
        comp_marked[to_idx(c)] = 1U;
      }
    };

    for (auto cr = comp_route_idx_t{0U}; cr != n_routes; ++cr) {
      if (!route_marked[to_idx(cr)]) {
        continue;
      }
      auto const seq = g.seqs_[cr];
      auto const durs = g.durations_[cr];
      auto const len = static_cast<unsigned>(seq.size());
      auto carry = kInf;
      if (fwd) {
        // distance TO the seeds: board at j, ride forward, alight at j' > j
        for (auto j = len - 1U; j != 0U; --j) {
          carry = sat_add(durs[j - 1U], std::min(prev[to_idx(seq[j])], carry));
          relax(seq[j - 1U], carry);
        }
      } else {
        // distance FROM the seeds: mirrored
        for (auto j = 0U; j != len - 1U; ++j) {
          carry = sat_add(durs[j], std::min(prev[to_idx(seq[j])], carry));
          relax(seq[j + 1U], carry);
        }
      }
    }
  }

  return lb;
}

}  // namespace nigiri::routing
