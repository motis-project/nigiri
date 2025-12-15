#include "nigiri/routing/ch_query.h"
#include "utl/helpers/algorithm.h"
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
#include <limits>
#include <vector>

namespace nigiri::routing {

static constexpr auto const kChMaxTravelTime = kMaxTravelTime * 5;  // TODO

void obtain_relevant_stops(timetable const& tt,
                           routing::query const& q,
                           profile_idx_t const prf_idx,
                           bitvec& relevant_stops) {

  if (tt.fwd_search_ch_graph_[prf_idx].size() != tt.n_locations()) {
    std::cout << "no ch for profile, skipping" << std::endl;
    relevant_stops.one_out();
    return;
  }

  std::cout << "max max: "
            << utl::max_element(tt.ch_graph_edges_[prf_idx],
                                [](auto const& a, auto const& b) {
                                  return a.max_dur_ < b.max_dur_;
                                })
                   ->max_dur_
            << " max min: "
            << utl::max_element(tt.ch_graph_edges_[prf_idx],
                                [](auto const& a, auto const& b) {
                                  return a.min_dur_ < b.min_dur_;
                                })
                   ->min_dur_
            << std::endl;
  std::cout << "min max: "
            << utl::max_element(tt.ch_graph_edges_[prf_idx],
                                [](auto const& a, auto const& b) {
                                  return a.max_dur_ > b.max_dur_;
                                })
                   ->max_dur_
            << " min min: "
            << utl::max_element(tt.ch_graph_edges_[prf_idx],
                                [](auto const& a, auto const& b) {
                                  return a.min_dur_ > b.min_dur_;
                                })
                   ->min_dur_
            << std::endl;

  std::cout << "upsearch" << std::endl;

  std::array<vector_map<location_idx_t, ch_dist>, 2> dists;
  dists[0].resize(tt.n_locations());
  dists[1].resize(tt.n_locations());

  auto pq = dial<ch_label, ch_get_bucket>{kChMaxTravelTime.count()};

  auto const init = [&](std::vector<routing::offset> offsets,
                        std::uint8_t dir) {
    for (auto const& start : offsets) {  // TODO correct offsets
      for_each_meta(
          tt, dir == kForward ? q.start_match_mode_ : q.dest_match_mode_,
          start.target_, [&](location_idx_t const x) {
            auto const d =
                static_cast<ch_dist::dist_t>(start.duration().count());
            dists[dir].at(x).d_[kMax] = d;
            dists[dir].at(x).d_[kMin] = d;
            pq.push(ch_label{x, {d, d}, dir});
            std::cout << "input" << x << " " << d << " "
                      << (dir == kForward ? "fw" : "bw") << std::endl;
          });
    }
  };
  init(q.start_, kForward);
  init(q.destination_, kReverse);
  auto min_max_dist = std::numeric_limits<ch_dist::dist_t>::max();
  auto min_min_dist = std::numeric_limits<ch_dist::dist_t>::max();
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
    /* std::cout << "steop " << l.l_ << " " << tt.locations_.names_[l.l_].view()
     << " min: " << l.d_[kMin] << " " << " max: " << l.d_[kMax] << " other:"
               << dists[other_dir][l.l_].d_[kMax] << " " << l_dir << " l:" <<
     tt.ch_levels_.at(l.l_)
               << std::endl;*/
    if (dists[other_dir][l.l_].d_[kMax] !=
        std::numeric_limits<ch_dist::dist_t>::max()) {
      if (dists[l_dir].at(l.l_).d_[kMax] + dists[other_dir][l.l_].d_[kMax] <
          min_max_dist) {
        min_max_dist =
            dists[l_dir].at(l.l_).d_[kMax] + dists[other_dir][l.l_].d_[kMax];
        min_min_dist =
            dists[l_dir].at(l.l_).d_[kMin] + dists[other_dir][l.l_].d_[kMin];
        meetpoints.emplace_back(l.l_);
      } else if (dists[l_dir].at(l.l_).d_[kMin] +
                     dists[other_dir][l.l_].d_[kMin] <=
                 min_max_dist) {
        meetpoints.emplace_back(l.l_);
      }
    }
    // std::cout << "mmd" << min_max_dist << std::endl;
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
        std::cout << "switching to min mode " << counter << " "
                  << "minmax: " << min_max_dist << " minmin: " << min_min_dist
                  << " infty: " << std::numeric_limits<ch_dist::dist_t>::max()
                  << std::endl;
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
      auto const new_max_dist =
          dists[l_dir].at(l.l_).d_[kMax] + e.max_dur_.count();
      auto const new_min_dist =
          dists[l_dir].at(l.l_).d_[kMin] + e.min_dur_.count();
      // std::cout << "tar" << edge_target << " " << new_max_dist << " ld " <<
      //  l.d_[kMax] << " em " << e.max_dur_.count() << " " << new_min_dist <<
      //  std::endl;
      if ((new_max_dist < dists[l_dir].at(edge_target).d_[kMax] ||
           new_min_dist < dists[l_dir].at(edge_target).d_[kMin]) &&
          new_max_dist < kChMaxTravelTime.count() &&
          new_min_dist < kChMaxTravelTime.count()) {
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
      } else if (new_max_dist >= kChMaxTravelTime.count() ||
                 new_min_dist >= kChMaxTravelTime.count()) {
        std::cout << "extra weird" << new_max_dist << " " << new_min_dist << " "
                  << dists[l_dir][edge_target].d_[kMax] << " "
                  << dists[l_dir][edge_target].d_[kMin] << std::endl;
      }
    }
  }
  pq.clear();
  auto const invert = [&](ch_dist::dist_t d) {
    return static_cast<ch_dist::dist_t>(kChMaxTravelTime.count() - d);
  };
  std::cout << "downsearch " << counter << std::endl;
  for (auto const m : meetpoints) {
    /*std::cout << m << " " << dists[kForward][m].d_[kMin] << " "
              << dists[kForward][m].d_[kMax] << " "
              << dists[kReverse][m].d_[kMin] << dists[kReverse][m].d_[kMax]
              << std::endl;*/
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
      if (min_dist_via_prev >= kChMaxTravelTime.count()) {
        std::cout << "weird" << min_dist_via_prev << " " << l_d_max << " "
                  << e.min_dur_.count() << std::endl;
        continue;
      }
      if (min_dist_via_prev <= l_d_max) {  // todo stopping criterion, cutoff?
        for (auto const mark : tt.ch_graph_transfers_[prf_idx].at(e_idx)) {
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
  }

  // relevant_stops.one_out();
  /*relevant_stops.zero_out();
  for (auto l : {66733, 66707,
    66707, 14037,
    14037, 24022,
    24022, 44946,
    44946, 60390,
    66733, 66707,
    66707, 14037,
    14037, 17360,
    17360, 17346,
    17346, 23630,
    23630, 23631,
    23631, 24051,
    24051, 45331,
    45331, 60392,
    66731, 67390,
    67390, 67386,
    67386, 66705,
    66705, 14037,
    14037, 24022,
    24022, 44946,
    44946, 60390,
    66731, 67390,
    67390, 67386,
    67386, 66705,
    66705, 14037,
    14037, 17360,
    17360, 17346,
    17346, 23630,
    23630, 23631,
    23631, 24051,
    24051, 45331,
    45331, 60392,
    66733, 66707,
    66707, 14038,
    14038, 24022,
    24022, 44946,
    44946, 60390,
    66733, 66707,
    66707, 14038,
    14038, 17360,
    17360, 17370,
    17370, 23630,
    23630, 23631,
    23631, 24051,
    24051, 45331,
    45331, 60392}) {
    relevant_stops.set(static_cast<unsigned>(l));
  }*/
  std::cout << "marked stops: " << relevant_stops.count() << "/"
            << relevant_stops.size() << std::endl;
}

}  // namespace nigiri::routing