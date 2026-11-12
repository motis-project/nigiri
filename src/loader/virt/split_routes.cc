#include "nigiri/loader/virt/split_routes.h"

#include <cstdint>
#include <algorithm>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "utl/expect.h"
#include "utl/helpers/algorithm.h"
#include "utl/zip.h"

#include "nigiri/loader/virt/route_stop_pairs.h"
#include "nigiri/logging.h"
#include "nigiri/timetable.h"

namespace nigiri::loader {

using group_idx_t = cista::strong<std::uint32_t, struct group_>;
using conflict_idx_t = cista::strong<std::uint32_t, struct conflict_>;
using split_route_idx_t = cista::strong<std::uint32_t, struct split_route_>;

route_idx_t transport_routes::get_route(timetable const& tt,
                                        route_idx_t const r,
                                        route_transport_idx_t const t) const {
  return transport_route_[tt.route_transport_ranges_[base_route_[r]].from_ +
                          to_idx(t)];
}

transport_routes split_routes(timetable const& tt, route_stop_pairs& pairs) {
  // Initialize:
  // - Map each route to iself.
  // - Map each transport to its base route.
  auto routes = transport_routes{};
  routes.base_route_.resize(tt.n_routes());
  std::iota(begin(routes.base_route_), end(routes.base_route_),
            route_idx_t{0U});
  routes.transport_route_ = tt.transport_route_;

  auto const reassign_transfers = [&](route_stop_pair_idx_t const p) {
    auto const route_of = [&](route_stop const& s,
                              route_transport_idx_t const t) {
      return s.is_regular() ? routes.get_route(tt, s.route_, t) : s.route_;
    };

    auto const from = pairs.route_stops_[pairs.pairs_[p].from_];
    auto const to = pairs.route_stops_[pairs.pairs_[p].to_];
    auto const durations = pairs.durations_[p];
    auto transfers = std::move(pairs.transfers_[p]);
    auto const n_moved = std::erase_if(transfers, [&](transfer const& t) {
      // Check if from route or to route moved to a split route.
      auto const from_route = route_of(from, t.from_);
      auto const to_route = route_of(to, t.to_);
      if (from_route == from.route_ && to_route == to.route_) {
        return false;
      }

      // Create new route stops and add the transfer.
      auto const new_from = pairs.get_or_create_route_stop(
          route_stop{from_route, from.stop_idx_, from.base_, from.sig_});
      auto const new_to = pairs.get_or_create_route_stop(
          route_stop{to_route, to.stop_idx_, to.base_, to.sig_});
      pairs.add_transfer(pairs.get_or_create_pair(new_from, new_to, durations),
                         t);

      return true;
    });

    // Update window.
    if (n_moved != 0U) {
      auto& w = pairs.windows_[p];
      w = durations.hull_;
      for (auto const& t : transfers) {
        narrow(w, t);
      }
    }

    pairs.transfers_[p] = std::move(transfers);
  };

  struct transport_groups {
    bool has_conflicts() const { return n_groups_ != 0U; }

    vector_map<route_transport_idx_t, group_idx_t> transport_group_;
    std::uint32_t n_groups_{0U};
  };
  auto const get_groups = [&](route_stop_pair_idx_t const p,
                              bool const is_from) -> transport_groups {
    auto const [from, to] = pairs.pairs_[p];
    auto const& rs = pairs.route_stops_[is_from ? from : to];

    if (!rs.is_regular()) {
      // Base and signature route stops never split.
      return {};
    }

    // For each transport: determine transfer interval based on all transfers.
    auto const n_transports = static_cast<std::uint16_t>(
        tt.route_transport_ranges_[routes.base_route_[rs.route_]].size());
    auto w = vector_map<route_transport_idx_t, std::optional<window_t>>(
        n_transports);
    for (auto const& t : pairs.transfers_[p]) {
      auto& tw = w[is_from ? t.from_ : t.to_];
      if (!tw.has_value()) {
        tw = pairs.durations_[p].hull_;
      }
      narrow(*tw, t);
    }

    // Sort transports by "restrictiveness" of their transfer window:
    // - most restrictive interval end (lowest)
    // - most restrictive interval begin (highest)
    // - for stable ordering: by transport index
    auto transports = std::vector<route_transport_idx_t>{};
    for (auto i = route_transport_idx_t{0U}; i != n_transports; ++i) {
      if (w[i].has_value()) {
        transports.push_back(i);
      }
    }
    utl::sort(transports, [&](auto&& a, auto&& b) {
      return std::tie(w[a]->to_, w[b]->from_, a) <
             std::tie(w[b]->to_, w[a]->from_, b);
    });

    // Interval stabbing: Assign each transport to a group so that each group
    // shares an admissible interval, with as few groups as possible.
    // Greedy algorithm is optimal locally (least groups per pair) based on the
    // sort order of transports.
    auto groups = transport_groups{};
    groups.transport_group_.resize(n_transports, group_idx_t::invalid());
    auto window = window_t{};
    for (auto const t : transports) {
      auto const& tw = *w[t];
      auto const both = window.intersect(tw);
      if (is_admissible(both)) {
        window = both;
      } else {
        window = tw;
        ++groups.n_groups_;
      }
      groups.transport_group_[t] = group_idx_t{groups.n_groups_ - 1U};
    }
    return groups.n_groups_ < 2U ? transport_groups{} : groups;
  };

  auto conflict_transport_group = vecvec<conflict_idx_t, group_idx_t>{};
  auto route_conflict_group = vecvec<split_route_idx_t, group_idx_t>{};
  auto split_route = std::vector<route_idx_t>{};

  // Enqueue all pairs that have conflicts (i.e. >= 2 transports that would need
  // different transfer durations of this pair).
  auto q = std::vector<route_stop_pair_idx_t>{};
  for (auto p = route_stop_pair_idx_t{0U}; p != pairs.pairs_.size(); ++p) {
    if (!is_admissible(pairs.windows_[p])) {
      q.push_back(p);
    }
  }

  auto n_split = 0U;
  while (!q.empty()) {
    auto const p = q.back();
    q.pop_back();

    // Conflict got solved by a previous split between insert and pop of p.
    if (is_admissible(pairs.windows_[p])) {
      continue;
    }

    auto const from_groups = get_groups(p, true);
    auto const to_groups = get_groups(p, false);
    auto const from_can_split = from_groups.has_conflicts();
    auto const to_can_split = to_groups.has_conflicts();
    if (!utl_expect(from_can_split || to_can_split, "loader.virtual_locations",
                    "split_routes: inadmissible pair without groups")) {
      continue;
    }

    // Split preference:
    // - if only one side can split: the side that can split
    // - if both sides can split: the side with fewer groups
    // - if both sides can split and have the same number of groups: from side
    auto const [from, to] = pairs.pairs_[p];
    auto const rs = std::tie(from_can_split, to_groups.n_groups_) <
                            std::tie(to_can_split, from_groups.n_groups_)
                        ? to
                        : from;

    auto const r = pairs.route_stops_[rs].route_;
    auto const base_route = routes.base_route_[r];
    auto const range = tt.route_transport_ranges_[base_route];
    auto const n_transports = static_cast<std::uint16_t>(range.size());

    // Collect (transport -> group) assignments for each conflict.
    conflict_transport_group.clear();
    for (auto const x : pairs.route_pairs_[r]) {
      if (is_admissible(pairs.windows_[x])) {
        continue;
      }

      auto const [x_from, x_to] = pairs.pairs_[x];
      for (auto const is_from : {true, false}) {
        if (pairs.route_stops_[is_from ? x_from : x_to].route_ == r) {
          auto g = get_groups(x, is_from);
          if (g.has_conflicts()) {
            conflict_transport_group.emplace_back(g.transport_group_);
          }
        }
      }
    }

    // Split this route along all its conflicts in which its side has several
    // groups. Conflicts with only one group on this route's side can only be
    // solved by splitting the other side.
    route_conflict_group.clear();
    split_route.clear();
    auto transport_conflict_group = vector_map<conflict_idx_t, group_idx_t>{};
    for (auto t = route_transport_idx_t{0U}; t != n_transports; ++t) {
      if (routes.get_route(tt, r, t) != r) {
        // Transport already got assigned to another route by a previous split.
        continue;
      }

      // Map (conflict_idx -> group_idx) of transport t in that conflict.
      transport_conflict_group.clear();
      for (auto const g : conflict_transport_group) {
        transport_conflict_group.push_back(g[to_idx(t)]);
      }

      if (utl::all_of(transport_conflict_group,
                      [](auto g) { return g == group_idx_t::invalid(); })) {
        // This transport has no transfers.
        continue;
      }

      // Find the first compatible split route for this transport.
      // Requires: All transports in the existing split route were assigned
      //           to the same group in each conflict as this transport.
      // Wildcard: group_idx::invalid transport has no transfer in this conflict
      auto const it = utl::find_if(route_conflict_group, [&](auto const& rg) {
        return std::ranges::equal(rg, transport_conflict_group,
                                  [](group_idx_t const a, group_idx_t const b) {
                                    return a == group_idx_t::invalid() ||
                                           b == group_idx_t::invalid() ||
                                           a == b;
                                  });
      });
      auto const k = split_route_idx_t{it - begin(route_conflict_group)};
      if (it == end(route_conflict_group)) {
        // Create new split route for this transport.
        route_conflict_group.emplace_back(transport_conflict_group);
        split_route.push_back(
            split_route.empty() ? r : route_idx_t{routes.base_route_.size()});
        if (split_route.back() != r) {
          routes.base_route_.push_back(base_route);
          pairs.route_pairs_.emplace_back();
        }
      } else {
        // Existing compatible split route found.
        // -> Merge requirements of the new transport
        //   (wildcards may be tightened to concrete group_idx).
        for (auto [route_g, transport_g] :
             utl::zip(*it, transport_conflict_group)) {
          if (transport_g != group_idx_t::invalid()) {
            route_g = transport_g;
          }
        }
      }

      // Update (transport -> split route) mapping.
      routes.transport_route_[range.from_ + to_idx(t)] = split_route[to_idx(k)];
    }

    // Repartition all pairs.
    auto const n_pairs_before = pairs.pairs_.size();
    // !! Copy to allow modifications in reassign_transfers(x).
    auto const route_pairs = pairs.route_pairs_[r];
    for (auto const x : route_pairs) {
      reassign_transfers(x);
    }

    // Push newly created pairs with conflicts.
    for (auto x = route_stop_pair_idx_t{n_pairs_before};
         x != pairs.pairs_.size(); ++x) {
      if (!is_admissible(pairs.windows_[x])) {
        q.push_back(x);
      }
    }
    q.push_back(p);
    ++n_split;
  }

  log(log_lvl::info, "loader.virtual_locations",
      "{} route stops, {} pairs, {} splits", pairs.route_stops_.size(),
      pairs.pairs_.size(), n_split);

  return routes;
}

}  // namespace nigiri::loader
