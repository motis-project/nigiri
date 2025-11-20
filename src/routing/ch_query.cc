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

  if (tt.fwd_search_ch_graph_[prf_idx].size() != tt.n_locations()) {
    std::cout << "no ch for profile, skipping" << std::endl;
    relevant_stops.one_out();
    return;
  }

  std::cout << "upsearch" << std::endl;

  std::array<vector_map<location_idx_t, ch_dist>, 2> dists;
  dists[0].resize(tt.n_locations());
  dists[1].resize(tt.n_locations());
  auto pq = dial<ch_label, ch_get_bucket>{kMaxTravelTime.count()};

  auto const init = [&](std::vector<routing::offset> offsets,
                        std::uint8_t dir) {
    for (auto const& start : offsets) {  // TODO correct offsets
      for_each_meta(
          tt, q.dest_match_mode_, start.target_, [&](location_idx_t const x) {
            auto const d =
                static_cast<ch_dist::dist_t>(start.duration().count());
            dists[dir].at(x).d_[kMax] = d;
            dists[dir].at(x).d_[kMin] = d;
            pq.push(ch_label{x, {d, d}, dir});
          });
    }
  };
  init(q.start_, 0);
  init(q.destination_, 1);
  auto min_max_dist = std::numeric_limits<ch_dist::dist_t>::max();
  auto mode = kMax;
  auto meetpoints = std::vector<location_idx_t>{};
  auto counter = 0;
  while (!pq.empty()) {
    ++counter;
    auto l = pq.top();
    pq.pop();
    auto const l_dir = l.dir_ % kModeOffset;
    auto const other_dir = l_dir ^ 1U;

    if (dists[l_dir].at(l.l_).d_[kMax] < l.d_[kMax] &&
        dists[l_dir].at(l.l_).d_[kMin] < l.d_[kMin]) {
      continue;
    }
    if (dists[other_dir][l.l_].d_[kMax] !=
        std::numeric_limits<ch_dist::dist_t>::max()) {
      if (l.d_[kMax] + dists[other_dir][l.l_].d_[kMax] < min_max_dist) {
        min_max_dist = l.d_[kMax] + dists[other_dir][l.l_].d_[kMax];
        meetpoints.emplace_back(l.l_);
      } else if (l.d_[kMin] + dists[other_dir][l.l_].d_[kMin] <= min_max_dist) {
        meetpoints.emplace_back(l.l_);
      }
    }
    if (l.d_[mode] > min_max_dist) {
      if (mode == kMax) {
        auto buffer = std::vector<ch_label>{};
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
        std::cout << "switching to min mode " << counter << std::endl;
        mode = kMin;
        continue;
      } else {
        std::cout << "reached ḿax with min " << counter << std::endl;
        break;
      }
    }

    auto const& graph = l_dir == kForward ? tt.fwd_search_ch_graph_[prf_idx]
                                          : tt.bwd_search_ch_graph_[prf_idx];

    for (auto const& e_idx : graph.at(l.l_)) {
      auto const e = tt.ch_graph_edges_[prf_idx].at(e_idx);
      auto const edge_target = l_dir == kForward ? e.to_ : e.from_;
      if (tt.ch_levels_.at(l.l_) > tt.ch_levels_.at(edge_target)) {
        continue;
      }
      auto const new_max_dist = l.d_[kMax] + e.max_dur_.count();
      auto const new_min_dist = l.d_[kMin] + e.min_dur_.count();
      if ((new_max_dist < dists[l_dir].at(edge_target).d_[kMax] ||
           new_min_dist < dists[l_dir].at(edge_target).d_[kMin]) &&
          new_max_dist < kMaxTravelTime.count() &&
          new_min_dist < kMaxTravelTime.count()) {
        dists[l_dir][edge_target].d_[kMax] =
            std::min(static_cast<ch_dist::dist_t>(new_max_dist),
                     dists[l_dir][edge_target].d_[kMax]);
        dists[l_dir][edge_target].d_[kMin] =
            std::min(static_cast<ch_dist::dist_t>(new_min_dist),
                     dists[l_dir][edge_target].d_[kMin]);
        pq.push(ch_label{edge_target,
                         {static_cast<ch_dist::dist_t>(new_max_dist),
                          static_cast<ch_dist::dist_t>(new_min_dist)},
                         static_cast<std::uint8_t>(l_dir)});
      } else if (new_max_dist >= kMaxTravelTime.count() ||
                 new_min_dist >= kMaxTravelTime.count()) {
        std::cout << "weird" << new_max_dist << " " << new_min_dist
                  << std::endl;
      }
    }
  }
  pq.clear();
  auto const invert = [&](ch_dist::dist_t d) {
    return static_cast<ch_dist::dist_t>(kMaxTravelTime.count() - d);
  };
  std::cout << "downsearch " << counter << std::endl;
  for (auto const m : meetpoints) {
    std::cout << m << " " << dists[kForward][m].d_[kMin] << " "
              << dists[kForward][m].d_[kMax] << " "
              << dists[kReverse][m].d_[kMin] << dists[kReverse][m].d_[kMax]
              << std::endl;
    if (dists[kForward][m].d_[kMin] + dists[kReverse][m].d_[kMin] >
        min_max_dist) {
      continue;
    }
    for (auto const dir : {kForward, kReverse}) {
      pq.push(ch_label{m,
                       {invert(dists[dir][m].d_[kMax]), 0},
                       static_cast<std::uint8_t>(dir)});
    }
  }
  std::cout << "starting pq" << std::endl;
  while (!pq.empty()) {
    auto l = pq.top();
    auto const l_d_max = invert(l.d_[kMax]);
    pq.pop();

    if (dists[l.dir_][l.l_].d_[kMin] >= l_d_max) {
      continue;
    }
    dists[l.dir_][l.l_].d_[kMin] = l_d_max;
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
      if (prev_label.d_[kMin] == std::numeric_limits<ch_dist::dist_t>::max()) {
        continue;
      }
      auto const min_dist_via_prev = prev_label.d_[kMin] + e.min_dur_.count();
      if (min_dist_via_prev >= routing::kMaxTravelTime.count()) {
        std::cout << "weird" << min_dist_via_prev << " " << l_d_max << " "
                  << e.min_dur_.count() << std::endl;
        continue;
      }
      if (min_dist_via_prev <= l_d_max) {  // todo stopping criterion, cutoff?
        for (auto const mark : tt.ch_graph_transfers_[prf_idx][e_idx]) {
          relevant_stops.set(mark.v_);
        }
        pq.push(ch_label{
            edge_target,
            {static_cast<ch_dist::dist_t>(invert(
                 static_cast<ch_dist::dist_t>(l_d_max - e.min_dur_.count()))),
             static_cast<ch_dist::dist_t>(0)},
            static_cast<std::uint8_t>(l.dir_)});
      }
    }
    std::cout << "marked stops: " << relevant_stops.count() << "/"
              << relevant_stops.size() << std::endl;
  }
}

}  // namespace nigiri::routing