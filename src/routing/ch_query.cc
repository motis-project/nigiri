#include "utl/helpers/algorithm.h"
#include "utl/insert_sorted.h"
#include "utl/pairwise.h"
#include "utl/zip.h"

#include "nigiri/loader/build_lb_graph.h"
#include "nigiri/common/dial.h"
#include "nigiri/for_each_meta.h"
#include "nigiri/logging.h"
#include "nigiri/routing/ch/ch_data.h"
#include "nigiri/routing/ch/ch_query.h"
#include "nigiri/routing/ch/saw.h"
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

  std::cout << "upsearch" << std::endl;

  std::array<vector_map<location_idx_t, ch_dist>, 2> dists;
  dists[0].resize(tt.n_locations());
  dists[1].resize(tt.n_locations());

  auto pq = dial<ch_label, ch_get_bucket>{kChMaxTravelTime.count()};

  auto const mark_relevant_stop = [&](location_idx_t const parent) {
    if (loader::kChGroupParents && !relevant_stops.test(parent.v_)) {
      for (auto const& c : tt.locations_.children_[parent]) {
        relevant_stops.set(c.v_);
        for (auto const& cc : tt.locations_.children_[c]) {
          relevant_stops.set(cc.v_);
        }
      }
    }
    relevant_stops.set(parent.v_);
  };

  auto new_max_dist = std::vector<tooth>{};
  auto new_min_dist = std::vector<tooth>{};
  auto tmp_saw = std::vector<tooth>{};

  auto const follow_edges = [&](location_idx_t const l, unsigned const l_dir,
                                u16_minutes const const_dist) {
    auto const other_dir = l_dir ^ 1U;
    auto const& graph = l_dir == kForward ? tt.fwd_search_ch_graph_[prf_idx]
                                          : tt.bwd_search_ch_graph_[prf_idx];

    std::cout << "weird" << graph.at(l).size() << std::endl;
    for (auto const& e_idx : graph.at(l)) {
      auto const e = tt.ch_graph_edges_[prf_idx].at(e_idx);
      auto const edge_target = l_dir == kForward ? e.to_ : e.from_;
      std::cout << "const_dist1" << const_dist << std::endl;
      if (tt.ch_levels_.at(l) > tt.ch_levels_.at(edge_target)) {
        continue;
      }
      if (const_dist != u16_minutes::max()) {
        std::cout << "const_dist2" << const_dist << std::endl;
        auto const d = owning_saw<saw_type::kConstant>{
            saw<saw_type::kConstant>::of(const_dist), {}};

        saw<kChSawType>{tt.ch_graph_max_[prf_idx].at(e_idx),
                        tt.ch_traffic_days_}
            .concat_const(other_dir, d.to_saw(tt.ch_traffic_days_),
                          new_max_dist);

        saw<kChSawType>{tt.ch_graph_min_[prf_idx].at(e_idx),
                        tt.ch_traffic_days_}
            .concat_const(other_dir, d.to_saw(tt.ch_traffic_days_),
                          new_min_dist);
      } else {
        dists[l_dir]
            .at(l)
            .d_[kMax]
            .to_saw(tt.ch_traffic_days_)
            .concat(l_dir,
                    saw<kChSawType>{tt.ch_graph_max_[prf_idx].at(e_idx),
                                    tt.ch_traffic_days_},
                    true, new_max_dist);
        dists[l_dir]
            .at(l)
            .d_[kMin]
            .to_saw(tt.ch_traffic_days_)
            .concat(l_dir,
                    saw<kChSawType>{tt.ch_graph_min_[prf_idx].at(e_idx),
                                    tt.ch_traffic_days_},
                    false, new_min_dist);
      }
      // std::cout << "tar" << edge_target << " " << new_max_dist << " ld " <<
      //  l.d_[kMax] << " em " << e.max_dur_.count() << " " << new_min_dist <<
      //  std::endl;
      auto const max_true =
          saw<kChSawType>{new_max_dist, tt.ch_traffic_days_}.leq(
              dists[l_dir].at(edge_target).d_[kMax].to_saw(tt.ch_traffic_days_),
              true);
      auto const min_true =
          saw<kChSawType>{new_min_dist, tt.ch_traffic_days_}.leq(
              dists[l_dir].at(edge_target).d_[kMin].to_saw(tt.ch_traffic_days_),
              true);
      if (max_true || min_true) {
        saw<kChSawType>{new_max_dist, tt.ch_traffic_days_}.simplify(
            dists[l_dir][edge_target].d_[kMax].to_saw(tt.ch_traffic_days_),
            tmp_saw);
        std::cout
            << "push pq" << edge_target << " " << max_true << " " << min_true
            << " new: " << saw<kChSawType>{new_max_dist, tt.ch_traffic_days_}
            << std::endl
            << " old: "
            << dists[l_dir].at(edge_target).d_[kMax].to_saw(tt.ch_traffic_days_)
            << std::endl
            << "simpl:" << saw<kChSawType>{tmp_saw, tt.ch_traffic_days_}
            << std::endl;
        std::swap(dists[l_dir][edge_target].d_[kMax].saw_, tmp_saw);
        tmp_saw.clear();
        saw<kChSawType>{new_min_dist, tt.ch_traffic_days_}.simplify(
            dists[l_dir][edge_target].d_[kMin].to_saw(tt.ch_traffic_days_),
            tmp_saw);
        std::swap(dists[l_dir][edge_target].d_[kMin].saw_, tmp_saw);
        tmp_saw.clear();

        auto const const_max = dists[l_dir][edge_target]
                                   .d_[kMax]
                                   .to_saw(tt.ch_traffic_days_)
                                   .max();
        auto const const_min = dists[l_dir][edge_target]
                                   .d_[kMin]
                                   .to_saw(tt.ch_traffic_days_)
                                   .min();
        utl::verify(const_max.count() < kChMaxTravelTime.count() &&
                        const_min.count() < kChMaxTravelTime.count(),
                    "extra weird {} {}", const_max, const_min);

        pq.push(ch_label{edge_target,
                         {const_max.count(), const_min.count()},
                         static_cast<std::uint8_t>(l_dir)});
      }
      new_max_dist.clear();
      new_min_dist.clear();
    }
  };

  auto const init = [&](std::vector<routing::offset> offsets,
                        std::uint8_t dir) {
    for (auto const& start : offsets) {  // TODO correct offsets
      for_each_meta(
          tt, dir == kForward ? q.start_match_mode_ : q.dest_match_mode_,
          start.target_, [&](location_idx_t const x) {
            relevant_stops.set(
                x.v_);  // TODO only mark locations actually on shortest paths?
            follow_edges(x, dir, start.duration());
            std::cout << "input" << x << " " << start.duration() << " "
                      << (dir == kForward ? "fw" : "bw") << std::endl;
          });
    }
  };

  init(q.start_, kForward);
  init(q.destination_, kReverse);
  auto min_max_dist = std::vector<tooth>{};
  auto mode = kMax;
  auto meetpoints = std::vector<location_idx_t>{};
  auto counter = 0;
  while (!pq.empty()) {
    ++counter;
    auto l = pq.top();
    pq.pop();
    auto const l_dir = l.dir_ % kModeOffset;
    auto const other_dir = l_dir ^ 1U;

    if (dists[l_dir]
                .at(l.l_)
                .d_[kMax]
                .to_saw(tt.ch_traffic_days_)
                .max()
                .count() < l.d_[kMax] &&  // TODO extrema
        dists[l_dir]
                .at(l.l_)
                .d_[kMin]
                .to_saw(tt.ch_traffic_days_)
                .min()
                .count() < l.d_[kMin]) {  // TODO nonce?
      continue;
    }
    std::cout
        << "steop " << l.l_  // << " " << tt.locations_.names_[l.l_]..view()
        << " min: " << l.d_[kMin] << " " << " max: " << l.d_[kMax] << " other:"
        << dists[other_dir][l.l_].d_[kMax].to_saw(tt.ch_traffic_days_).max()
        << " " << l_dir << " l:" << tt.ch_levels_.at(l.l_) << std::endl;
    if (!dists[other_dir][l.l_].d_[kMax].saw_.empty()) {
      auto max_concat =
          dists[l_dir]
              .at(l.l_)
              .d_[kMax]
              .to_saw(tt.ch_traffic_days_)
              .concat(
                  l_dir,
                  dists[other_dir][l.l_].d_[kMax].to_saw(tt.ch_traffic_days_),
                  true, tmp_saw);
      if (max_concat <= saw<kChSawType>{min_max_dist, tt.ch_traffic_days_}) {
        max_concat.simplify(saw<kChSawType>{min_max_dist, tt.ch_traffic_days_},
                            new_max_dist);
        std::swap(min_max_dist, new_max_dist);
        new_max_dist.clear();
        meetpoints.emplace_back(l.l_);
      } else if (tmp_saw.clear();
                 dists[l_dir]
                     .at(l.l_)
                     .d_[kMin]
                     .to_saw(tt.ch_traffic_days_)
                     .concat(l_dir,
                             dists[other_dir][l.l_].d_[kMin].to_saw(
                                 tt.ch_traffic_days_),
                             false, tmp_saw) <=
                 saw<kChSawType>{min_max_dist, tt.ch_traffic_days_}) {
        meetpoints.emplace_back(l.l_);
      }
      std::cout << "mp found "
                << saw<kChSawType>{min_max_dist, tt.ch_traffic_days_}.max()
                << std::endl;

      tmp_saw.clear();
    }
    // std::cout << "mmd" << min_max_dist << std::endl;
    if (dists[l_dir].at(l.l_).d_[mode].to_saw(tt.ch_traffic_days_) >
        saw<kChSawType>{min_max_dist, tt.ch_traffic_days_}) {
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
                  << "minmax: "
                  << saw<kChSawType>{min_max_dist, tt.ch_traffic_days_}.max()
                  << " infty: " << std::numeric_limits<ch_label::dist_t>::max()
                  << std::endl;
        mode = kMin;
        continue;
      } else {
        std::cout << "reached ḿax with min " << counter << std::endl;
        break;
      }
    }

    follow_edges(l.l_, l_dir, u16_minutes::max());
  }
  pq.clear();

  auto const invert = [&](ch_label::dist_t d) {
    return static_cast<ch_label::dist_t>(kChMaxTravelTime.count() - d);
  };
  auto nonce_map = vector_map<location_idx_t, std::uint32_t>{};
  nonce_map.resize(tt.n_locations());
  auto const const_min_max_dist =
      saw<kChSawType>{min_max_dist, tt.ch_traffic_days_}.max();
  std::cout << "downsearch " << counter << " " << meetpoints.size()
            << std::endl;
  for (auto const m : meetpoints) {
    dists[kForward][m]
        .d_[kMin]
        .to_saw(tt.ch_traffic_days_)
        .concat(dists[kReverse][m].d_[kMin].to_saw(tt.ch_traffic_days_), false,
                tmp_saw);
    std::cout << "mp filter "
              << saw<kChSawType>{tmp_saw, tt.ch_traffic_days_}.min() << " "
              << saw<kChSawType>{min_max_dist, tt.ch_traffic_days_}.max()
              << std::endl;
    for (auto const& e : tmp_saw) {
      std::cout << e.mam_ << " " << e.travel_dur_ << std::endl;
    }
    std::cout << "min_max <" << std::endl;
    for (auto const& e : tmp_saw) {
      std::cout << e.mam_ << " " << e.travel_dur_ << std::endl;
    }
    if (saw<kChSawType>{tmp_saw, tt.ch_traffic_days_} >
        saw<kChSawType>{min_max_dist, tt.ch_traffic_days_}) {
      tmp_saw.clear();
      continue;
    }
    tmp_saw.clear();
    for (auto const dir : {kForward, kReverse}) {
      auto const other_dir = dir ^ 1U;

      saw<kChSawType>{min_max_dist, tt.ch_traffic_days_}.simplify(
          dists[dir][m].d_[kMax].to_saw(tt.ch_traffic_days_), tmp_saw);
      std::swap(dists[dir][m].d_[kMax].saw_, tmp_saw);
      tmp_saw.clear();
      auto const d = std::min(
          static_cast<ch_label::dist_t>(const_min_max_dist.count() -
                                        dists[other_dir][m]
                                            .d_[kMin]
                                            .to_saw(tt.ch_traffic_days_)
                                            .min()
                                            .count()),
          dists[dir][m].d_[kMax].to_saw(tt.ch_traffic_days_).max().count());
      pq.push(ch_label{m, {invert(d), 1}, static_cast<std::uint8_t>(dir)});
      std::cout << "added mp " << d << std::endl;
    }
  }
  std::cout << "starting pq" << std::endl;
  auto stack = std::vector<std::pair<ch_edge_idx_t, ch_label::dist_t>>{};
  auto visited = vector_map<ch_edge_idx_t, ch_label::dist_t>{};
  visited.resize(tt.ch_graph_edges_[prf_idx].size());

  while (!pq.empty()) {
    auto l = pq.top();
    auto const l_d_max = static_cast<int>(invert(l.d_[kMax]));
    pq.pop();

    if (l.d_[kMin] <= nonce_map.at(l.l_)) {
      continue;
    }
    std::cout << "down " << l.l_
              << " "  // << tt.locations_.names_[l.l_].view()
                      // << " min: " << dists[l.dir_][l.l_].d_[kMin] << " "
              << " max: " << l_d_max
              << " dir:" << (l.dir_ == kForward ? "fwd" : "bwd")
              << "| l:" << tt.ch_levels_.at(l.l_) << std::endl;
    nonce_map.at(l.l_) = l.d_[kMin];
    saw<kChSawType>{min_max_dist, tt.ch_traffic_days_}.simplify(
        dists[l.dir_][l.l_].d_[kMax].to_saw(tt.ch_traffic_days_), tmp_saw);
    std::swap(dists[l.dir_][l.l_].d_[kMax].saw_,
              tmp_saw);  // TODO min with l_d_max-e.min_dur
    tmp_saw.clear();
    mark_relevant_stop(l.l_);
    auto const& graph = l.dir_ == kReverse ? tt.fwd_search_ch_graph_[prf_idx]
                                           : tt.bwd_search_ch_graph_[prf_idx];

    for (auto const& e_idx : graph[l.l_]) {
      auto const e = tt.ch_graph_edges_[prf_idx][e_idx];
      auto const edge_target = l.dir_ == kReverse ? e.to_ : e.from_;
      if (tt.ch_levels_[l.l_] < tt.ch_levels_[edge_target]) {
        continue;
      }
      auto const& prev_label = dists[l.dir_][edge_target];
      if (prev_label.d_[kMin].saw_.empty()) {
        continue;
      }
      prev_label.d_[kMin]
          .to_saw(tt.ch_traffic_days_)
          .concat(l.dir_,
                  saw<kChSawType>{tt.ch_graph_min_[prf_idx].at(e_idx),
                                  tt.ch_traffic_days_},
                  false, tmp_saw);
      auto min_dist_via_prev = saw<kChSawType>{tmp_saw, tt.ch_traffic_days_};
      if (min_dist_via_prev.max().count() >=
          kChMaxTravelTime.count()) {  // TODO expensive
        std::cout << "weird" << min_dist_via_prev.max().count() << " "
                  << l_d_max << " " << std::endl;
        tmp_saw.clear();
        continue;
      }
      if (min_dist_via_prev <=
              dists[l.dir_][l.l_].d_[kMax].to_saw(tt.ch_traffic_days_) &&
          min_dist_via_prev.min().count() <=
              l_d_max) {  // TODO l_d_max cheat, stopping criterion, cutoff?
        // TODO move to queue pop?
        stack.push_back(
            {e_idx, std::min(static_cast<ch_label::dist_t>(
                                 l_d_max - prev_label.d_[kMin]
                                               .to_saw(tt.ch_traffic_days_)
                                               .min()
                                               .count()),
                             static_cast<ch_label::dist_t>(saw<kChSawType>{
                                 tt.ch_graph_max_[prf_idx].at(e_idx),
                                 tt.ch_traffic_days_}
                                                               .max()
                                                               .count()))});
        while (!stack.empty()) {
          auto [child_edge_idx, child_max_dur] = stack.back();
          stack.pop_back();
          if (visited.at(child_edge_idx) >= child_max_dur) {
            continue;
          }
          visited[child_edge_idx] = child_max_dur;
          auto const child_max_dur_saw = saw<kChSawType>{
              tt.ch_graph_max_[prf_idx].at(child_edge_idx),
              tt.ch_traffic_days_};  // TODO at least min with min_max_dur?

          for (auto const [unpack, transfer] :
               utl::zip(tt.ch_graph_unpack_[prf_idx].at(child_edge_idx),
                        tt.ch_graph_transfers_[prf_idx].at(child_edge_idx))) {
            if (unpack.first == ch_edge_idx_t::invalid()) {
              if (transfer != location_idx_t::invalid()) {
                mark_relevant_stop(transfer);
              }
              continue;
            }
            auto const arr_min_saw =
                saw<kChSawType>{tt.ch_graph_min_[prf_idx].at(unpack.first),
                                tt.ch_traffic_days_};
            auto const dep_min_saw =
                saw<kChSawType>{tt.ch_graph_min_[prf_idx].at(unpack.second),
                                tt.ch_traffic_days_};
            auto const arr_min = arr_min_saw.min().count();
            auto const dep_min = dep_min_saw.min().count();
            auto const arr_max =
                std::min(static_cast<ch_label::dist_t>(child_max_dur - dep_min),
                         static_cast<ch_label::dist_t>(saw<kChSawType>{
                             tt.ch_graph_max_[prf_idx].at(unpack.first),
                             tt.ch_traffic_days_}
                                                           .max()
                                                           .count()));
            auto const dep_max =
                std::min(static_cast<ch_label::dist_t>(
                             child_max_dur - arr_min),  // TODO deconcat?
                         static_cast<ch_label::dist_t>(saw<kChSawType>{
                             tt.ch_graph_max_[prf_idx].at(unpack.second),
                             tt.ch_traffic_days_}
                                                           .max()
                                                           .count()));

            if (arr_min_saw > child_max_dur_saw ||
                dep_min_saw > child_max_dur_saw || arr_min > arr_max ||
                dep_min > dep_max) {
              continue;  // TODO count occurs
            }
            if (transfer != location_idx_t::invalid()) {
              mark_relevant_stop(transfer);
            }
            stack.push_back({unpack.first, arr_max});
            stack.push_back({unpack.second, dep_max});
          }
        }
        pq.push(ch_label{
            edge_target,
            {static_cast<ch_label::dist_t>(invert(std::min(
                 static_cast<ch_label::dist_t>(
                     l_d_max -
                     saw<kChSawType>{tt.ch_graph_min_[prf_idx].at(e_idx),
                                     tt.ch_traffic_days_}  // TODO deconcat?
                         .min()
                         .count()),
                 dists[l.dir_][edge_target]
                     .d_[kMax]
                     .to_saw(tt.ch_traffic_days_)
                     .max()
                     .count()))),
             static_cast<ch_label::dist_t>(nonce_map[l.l_] +
                                           1)},  // TODO is this correct?
            static_cast<std::uint8_t>(l.dir_)});
      }
      tmp_saw.clear();
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