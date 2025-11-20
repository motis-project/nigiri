#pragma once

#include "nigiri/routing/ch_query.h"
#include "utl/insert_sorted.h"
#include "utl/pairwise.h"

#include "nigiri/common/dial.h"
#include "nigiri/for_each_meta.h"
#include "nigiri/logging.h"
#include "nigiri/routing/limits.h"
#include "nigiri/td_footpath.h"

#include "nigiri/timetable.h"
#include "nigiri/types.h"
#include <algorithm>
#include <vector>

namespace nigiri::routing {

void obtain_relevant_stops(timetable const& tt,
                           routing::query const& q,
                           profile_idx_t const prf_idx,
                           bitvec& relevant_stops) {

  std::array<vector_map<location_idx_t, dist>, 2> dists;
  dists[0].resize(tt.n_locations());
  dists[1].resize(tt.n_locations());
  auto pq = dial<label, get_bucket>{routing::kMaxTravelTime.count()};

  auto const init = [&](std::vector<routing::offset> offsets,
                        std::uint8_t dir) {
    for (auto const& start : offsets) {  // TODO correct offsets
      for_each_meta(
          tt, q.dest_match_mode_, start.target_, [&](location_idx_t const x) {
            auto const d = static_cast<dist::dist_t>(start.duration().count());
            dists[dir][x].d_[kMax] = d;
            dists[dir][x].d_[kMin] = d;
            pq.push(label{x, {d, d}, dir});
          });
    }
  };
  init(q.start_, 0);
  init(q.destination_, 1);
  auto min_max_dist = std::numeric_limits<dist::dist_t>::max();
  auto mode = kMax;
  auto meetpoints = std::vector<location_idx_t>{};
  while (!pq.empty()) {
    auto l = pq.top();
    pq.pop();
    auto const l_dir = l.dir_ % kModeOffset;
    auto const other_dir = (l_dir) ^ 1U;

    if (dists[l_dir][l.l_].d_[kMax] < l.d_[kMax] &&
        dists[l_dir][l.l_].d_[kMin] < l.d_[kMin]) {
      continue;
    }
    if (dists[other_dir][l.l_].d_[kMax] !=
        std::numeric_limits<dist::dist_t>::max()) {
      if (l.d_[kMax] + dists[other_dir][l.l_].d_[kMax] < min_max_dist) {
        min_max_dist = l.d_[kMax] + dists[other_dir][l.l_].d_[kMax];
        meetpoints.emplace_back(l.l_);
      } else if (l.d_[kMin] + dists[other_dir][l.l_].d_[kMin] <= min_max_dist) {
        meetpoints.emplace_back(l.l_);
      }
    }
    if (l.d_[mode] > min_max_dist) {
      if (mode == kMax) {
        auto buffer = std::vector<label>{};
        while (!pq.empty()) {
          auto b = pq.top();
          b.dir_ += kModeOffset;
          buffer.emplace_back(b);
          pq.pop();
        }
        l.dir_ += kModeOffset;
        pq.push(l);
        for (auto const& b : buffer) {
          pq.push(b);
        }
        mode = kMin;
        continue;
      } else {
        break;
      }
    }

    auto const& graph = l_dir == kForward ? tt.fwd_search_ch_graph_[prf_idx]
                                          : tt.bwd_search_ch_graph_[prf_idx];

    for (auto const& e_idx : graph[l.l_]) {
      auto const e = tt.ch_graph_edges_[prf_idx][e_idx];
      auto const edge_target = l_dir == kForward ? e.to_ : e.from_;
      if (tt.ch_levels_[l.l_] > tt.ch_levels_[edge_target]) {
        continue;
      }
      auto const new_max_dist = l.d_[kMax] + e.max_dur_.count();
      auto const new_min_dist = l.d_[kMin] + e.min_dur_.count();
      if ((new_max_dist < dists[l_dir][edge_target].d_[kMax] ||
           new_min_dist < dists[l_dir][edge_target].d_[kMin]) &&
          new_max_dist < pq.n_buckets() &&
          new_max_dist < routing::kMaxTravelTime.count()) {
        dists[l_dir][edge_target].d_[kMax] =
            std::min(static_cast<dist::dist_t>(new_max_dist),
                     dists[l_dir][edge_target].d_[kMax]);
        dists[l_dir][edge_target].d_[kMin] =
            std::min(static_cast<dist::dist_t>(new_min_dist),
                     dists[l_dir][edge_target].d_[kMin]);
        pq.push(label{edge_target,
                      {static_cast<dist::dist_t>(new_max_dist),
                       static_cast<dist::dist_t>(new_min_dist)},
                      static_cast<std::uint8_t>(l_dir)});
      }
    }
  }
  pq.clear();
  for (auto const m : meetpoints) {
    if (dists[kForward][m].d_[kMin] + dists[kReverse][m].d_[kMin] >
        min_max_dist) {
      continue;
    }
    for (auto const dir : {kForward, kReverse}) {
      pq.push(label{m,
                    {static_cast<dist::dist_t>(-dists[dir][m].d_[kMax]),
                     static_cast<dist::dist_t>(-dists[dir][m].d_[kMin])},
                    static_cast<std::uint8_t>(dir)});
    }
  }
  while (!pq.empty()) {
    auto l = pq.top();
    pq.pop();
    if (dists[l.dir_][l.l_].d_[kMin] >= -l.d_[kMax]) {
      continue;
    }
    dists[l.dir_][l.l_].d_[kMin] = -l.d_[kMax];
    relevant_stops.set(l.l_.v_);
    auto const& graph = l.dir_ == kReverse ? tt.fwd_search_ch_graph_[prf_idx]
                                           : tt.bwd_search_ch_graph_[prf_idx];

    for (auto const& e_idx : graph[l.l_]) {
      auto const e = tt.ch_graph_edges_[prf_idx][e_idx];
      auto const edge_target = l.dir_ == kReverse ? e.to_ : e.from_;
      if (tt.ch_levels_[l.l_] < tt.ch_levels_[edge_target]) {
        continue;
      }
      auto const& prev_label = dists[l.dir_][edge_target];
      if (prev_label.d_[kMin] == std::numeric_limits<dist::dist_t>::max()) {
        continue;
      }
      auto const min_dist_via_prev = prev_label.d_[kMin] + e.min_dur_.count();
      if (min_dist_via_prev <=
          -l.d_[kMax]) {  // todo stopping criterion, cutoff?
        for (auto const mark : tt.ch_graph_transfers_[prf_idx][e_idx]) {
          relevant_stops.set(mark.v_);
        }
        pq.push(label{
            edge_target,
            {static_cast<dist::dist_t>(-(-l.d_[kMax] - e.min_dur_.count())),
             static_cast<dist::dist_t>(0)},
            static_cast<std::uint8_t>(l.dir_)});
      }
    }
    std::cout << "marked stops: " << relevant_stops.count() << "/"
              << relevant_stops.size() << std::endl;
  }
}

}  // namespace nigiri::routing