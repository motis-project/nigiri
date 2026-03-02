#include "utl/helpers/algorithm.h"
#include "utl/insert_sorted.h"
#include "utl/pairwise.h"
#include "utl/verify.h"
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
#include <queue>
#include <vector>

namespace nigiri::routing {

static constexpr auto const kChMaxTravelTime = kMaxTravelTime * 5;  // TODO
static constexpr auto const kToothUnpackMode = true;

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

  auto bf = bitvec{};
  bf.resize(tt.ch_traffic_days_[prf_idx].size());
  for (auto const e : tt.ch_graph_min_[prf_idx]) {
    for (auto const& t : e) {
      if (t.traffic_days_ != bitfield_idx_t::invalid()) {

        bf.set(t.traffic_days_.v_);
      }
    }
  }
  std::cout << bf.count() << "/" << bf.size() << std::endl;
  for (auto const e : tt.ch_graph_max_[prf_idx]) {
    for (auto const& t : e) {
      if (t.traffic_days_ != bitfield_idx_t::invalid()) {
        bf.set(t.traffic_days_.v_);
      }
    }
  }
  std::cout << bf.count() << "/" << bf.size() << std::endl;

  std::array<vector_map<location_idx_t, ch_dist>, 2> dists;
  dists[0].resize(tt.n_locations());
  dists[1].resize(tt.n_locations());
  auto nonce_map = vector_map<location_idx_t, std::uint32_t>{};
  nonce_map.resize(tt.n_locations());

  auto pq = dial<ch_label, ch_get_bucket>{kChMaxTravelTime.count()};
  auto ch_traffic_days =
      traffic_days{tt.ch_traffic_days_[prf_idx], {}};  // TODO avoid copy

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
  // auto mode = kMin;

  auto const follow_edges = [&](location_idx_t const l, unsigned const l_dir,
                                u16_minutes const const_dist,
                                std::array<ch_label::dist_t, 2> const& dist) {
    auto const other_dir = l_dir ^ 1U;
    auto const& graph = l_dir == kForward ? tt.fwd_search_ch_graph_[prf_idx]
                                          : tt.bwd_search_ch_graph_[prf_idx];

    // std::cout << "weird" << graph.at(l).size() << std::endl;
    for (auto const& e_idx : graph.at(l)) {
      auto const e = tt.ch_graph_edges_[prf_idx].at(e_idx);
      auto const edge_target = l_dir == kForward ? e.to_ : e.from_;
      // std::cout << "const_dist1" << const_dist << std::endl;
      if (tt.ch_levels_[prf_idx].at(l) >
          tt.ch_levels_[prf_idx].at(edge_target)) {
        continue;
      }
      if (const_dist != u16_minutes::max()) {
        // TODO dead code (?)
        std::cout << "const_dist2" << const_dist << std::endl;
        auto const d = owning_saw<saw_type::kConstant>{
            saw<saw_type::kConstant>::of(const_dist), {}};

        saw<kChSawType>{tt.ch_graph_max_[prf_idx].at(e_idx), ch_traffic_days}
            .concat_const(other_dir, d.to_saw(ch_traffic_days),
                          ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid(),
                          new_max_dist);

        saw<kChSawType>{tt.ch_graph_min_[prf_idx].at(e_idx), ch_traffic_days}
            .concat_const(other_dir, d.to_saw(ch_traffic_days),
                          ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid(),
                          new_min_dist);
      } else {
        dists[l_dir]
            .at(l)
            .d_[kMax]
            .to_saw(ch_traffic_days)
            .concat(l_dir,
                    saw<kChSawType>{tt.ch_graph_max_[prf_idx].at(e_idx),
                                    ch_traffic_days},
                    ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid(), true,
                    new_max_dist);
        dists[l_dir]
            .at(l)
            .d_[kMin]
            .to_saw(ch_traffic_days)
            .concat(l_dir,
                    saw<kChSawType>{tt.ch_graph_min_[prf_idx].at(e_idx),
                                    ch_traffic_days},
                    ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid(), false,
                    new_min_dist);
      }
      // std::cout << "tar" << edge_target << " " << new_max_dist << " ld " <<
      //  l.d_[kMax] << " em " << e.max_dur_.count() << " " << new_min_dist <<
      //  std::endl;

      /*auto const max_true = saw<kChSawType>{new_max_dist,
      ch_traffic_days}.leq(
          dists[l_dir].at(edge_target).d_[kMax].to_saw(ch_traffic_days), true);
      auto const min_true = saw<kChSawType>{new_min_dist, ch_traffic_days}.leq(
          dists[l_dir].at(edge_target).d_[kMin].to_saw(ch_traffic_days), true);
      if (max_true || min_true) {*/
      saw<kChSawType>{new_max_dist, ch_traffic_days}.simplify(
          dists[l_dir][edge_target].d_[kMax].to_saw(ch_traffic_days), true,
          tmp_saw);  // TODO move to leq again? or detect within simplify if not
      // equal (new bitfields created etc)
      /*std::cout << "push pq" << edge_target << " " << 0 << " " << 0
                << " new: " << saw<kChSawType>{new_max_dist, ch_traffic_days}
                << std::endl
                << " old: "
                << dists[l_dir].at(edge_target).d_[kMax].to_saw(ch_traffic_days)
                << std::endl
                << "simpl:" << saw<kChSawType>{tmp_saw, ch_traffic_days}
                << std::endl;*/
      new_max_dist.clear();
      saw<kChSawType>{tmp_saw, ch_traffic_days}.min(
          new_max_dist);  // TODO should not be necessary
      auto max_leq = false;
      auto min_leq = false;
      if (dists[l_dir].at(edge_target).d_[kMax].to_saw(ch_traffic_days) !=
          saw<kChSawType>{new_max_dist, ch_traffic_days}) {
        std::swap(dists[l_dir][edge_target].d_[kMax].saw_, new_max_dist);
        max_leq = true;
      }
      new_max_dist.clear();
      tmp_saw.clear();
      saw<kChSawType>{new_min_dist, ch_traffic_days}.simplify(
          dists[l_dir][edge_target].d_[kMin].to_saw(ch_traffic_days), false,
          tmp_saw);
      new_min_dist.clear();
      saw<kChSawType>{tmp_saw, ch_traffic_days}.min(
          new_min_dist);  // TODO should not be necessary
      if (dists[l_dir].at(edge_target).d_[kMin].to_saw(ch_traffic_days) !=
          saw<kChSawType>{new_min_dist, ch_traffic_days}) {
        std::swap(dists[l_dir][edge_target].d_[kMin].saw_, new_min_dist);
        min_leq = true;
      }
      new_min_dist.clear();
      tmp_saw.clear();
      if (!min_leq && !max_leq) {
        /*std::cout << "skip " << e.from_ << " " << e.to_ << " " << edge_target
                  << std::endl;*/
        continue;
      }
      auto const const_max =
          dists[l_dir][edge_target].d_[kMax].to_saw(ch_traffic_days).max();
      auto const const_min =
          dists[l_dir][edge_target].d_[kMin].to_saw(ch_traffic_days).min();
      utl::verify(const_max.count() < kChMaxTravelTime.count() &&
                      const_min.count() < kChMaxTravelTime.count(),
                  "extra weird {} {}", const_max, const_min);
      std::cout << "push " << e.from_ << " " << e.to_ << " " << edge_target
                << " minmay " << const_min << " " << const_max << std::endl;

      auto const const_min_edge =
          saw<kChSawType>{tt.ch_graph_min_[prf_idx].at(e_idx), ch_traffic_days}
              .min();
      if (dist[kMax] + const_min_edge.count() >= kChMaxTravelTime.count()) {

        std::cout << "extra extra weird " << dist[kMax] << " "
                  << const_min_edge.count() << std::endl;
        continue;
      }
      pq.push(ch_label{
          edge_target,
          {static_cast<ch_label::dist_t>(dist[kMax] + const_min_edge.count()),
           static_cast<ch_label::dist_t>(nonce_map.at(edge_target) + 1)},
          static_cast<std::uint8_t>(l_dir)});
      //}
    }
  };

  for (auto i = location_idx_t{0U}; i < tt.n_locations(); ++i) {
    std::cout << i << " "
              << tt.get_default_translation(tt.locations_.names_.at(i))
              << " l:" << tt.ch_levels_[prf_idx].at(i) << std::endl;
  }

  auto const init = [&](std::vector<routing::offset> offsets,
                        std::uint8_t dir) {
    for (auto const& start : offsets) {  // TODO correct offsets
      for_each_meta(
          tt, dir == kForward ? q.start_match_mode_ : q.dest_match_mode_,
          start.target_, [&](location_idx_t const x) {
            auto const d =
                static_cast<ch_label::dist_t>(start.duration().count());
            dists[dir].at(x).d_[kMax] = owning_saw<kChSawType>{
                saw<saw_type::kConstant>::of(start.duration()), {}};
            dists[dir].at(x).d_[kMin] = owning_saw<kChSawType>{
                saw<saw_type::kConstant>::of(start.duration()), {}};
            pq.push(ch_label{
                x,
                {d, static_cast<ch_label::dist_t>(nonce_map.at(x) + 1)},
                dir});
            std::cout << "input" << x << " " << start.duration() << " "
                      << (dir == kForward ? "fw" : "bw") << std::endl;
          });
    }
  };

  init(q.start_, kForward);
  init(q.destination_, kReverse);
  auto min_max_dist = std::vector<tooth>{};
  auto meetpoints = hash_set<location_idx_t>{};  // TODO other way of dedup?
  auto counter = 0;

  while (!pq.empty()) {
    ++counter;
    auto l = pq.top();
    pq.pop();
    auto const l_dir = l.dir_ % kModeOffset;
    auto const other_dir = l_dir ^ 1U;

    if (l.d_[kMin] <= nonce_map.at(l.l_)) {
      continue;
    }
    nonce_map.at(l.l_) = l.d_[kMin];

    /*if (dists[l_dir].at(l.l_).d_[kMax].to_saw(ch_traffic_days).max().count() <
            l.d_[kMax] &&  // TODO extrema
        dists[l_dir].at(l.l_).d_[kMin].to_saw(ch_traffic_days).min().count() <
            l.d_[kMin]) {  // TODO nonce?
      continue;
    }*/
    std::cout << "steop " << l.l_ << " "
              << tt.get_default_translation(tt.locations_.names_.at(l.l_))
              << " nonce: " << l.d_[kMin] << " " << " max: " << l.d_[kMax]
              << " other:"
              << dists[other_dir][l.l_].d_[kMax].to_saw(ch_traffic_days).max()
              << " " << l_dir << " l:" << tt.ch_levels_[prf_idx].at(l.l_)
              << std::endl;
    /*std::cout << "max " << dists[l_dir][l.l_].d_[kMax].to_saw(ch_traffic_days)
              << std::endl;
    std::cout << "max " << dists[l_dir][l.l_].d_[kMin].to_saw(ch_traffic_days)
              << std::endl;*/
    if (!dists[other_dir][l.l_].d_[kMax].saw_.empty()) {
      auto max_concat =
          dists[l_dir]
              .at(l.l_)
              .d_[kMax]
              .to_saw(ch_traffic_days)
              .concat(l_dir,
                      dists[other_dir][l.l_].d_[kMax].to_saw(ch_traffic_days),
                      ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid(), true,
                      tmp_saw);
      max_concat.simplify(saw<kChSawType>{min_max_dist, ch_traffic_days}, true,
                          new_max_dist);
      if (saw<kChSawType>{new_max_dist, ch_traffic_days} !=
          saw<kChSawType>{min_max_dist, ch_traffic_days}) {
        std::swap(min_max_dist, new_max_dist);
        meetpoints.emplace(l.l_);
      } else if (tmp_saw.clear();
                 dists[l_dir]
                     .at(l.l_)
                     .d_[kMin]
                     .to_saw(ch_traffic_days)
                     .concat(l_dir,
                             dists[other_dir][l.l_].d_[kMin].to_saw(
                                 ch_traffic_days),
                             ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid(),
                             false, tmp_saw) <=
                 saw<kChSawType>{min_max_dist, ch_traffic_days}) {
        meetpoints.emplace(l.l_);
      }
      std::cout << "mp found "
                << saw<kChSawType>{min_max_dist, ch_traffic_days}.max()
                << std::endl;

      new_max_dist.clear();
      tmp_saw.clear();
    }
    // std::cout << "mmd" << min_max_dist << std::endl;
    if (dists[l_dir].at(l.l_).d_[kMin].to_saw(ch_traffic_days) >
        saw<kChSawType>{min_max_dist, ch_traffic_days}) {
      /*if (mode == kMax) {
        auto buffer = std::vector<ch_label>{};
        while (!pq.empty()) {
          auto b = pq.top();
          //b.dir_ += kModeOffset;
          b.d_[kMax] =
      dists[l_dir].at(l.l_).d_[kMin].to_saw(ch_traffic_days).min().count();
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
                  << saw<kChSawType>{min_max_dist, ch_traffic_days}.max()
                  << " infty: " << std::numeric_limits<ch_label::dist_t>::max()
                  << std::endl;
        mode = kMin;
        continue;
      } else {
        std::cout << "reached ḿax with min " << counter << std::endl;
        break;
      }*/
      std::cout << "reached ḿax with min " << counter << std::endl;
      break;
    }

    //    std::cout << "follow " << mode << l.l_ << std::endl;
    follow_edges(l.l_, l_dir, u16_minutes::max(), l.d_);
  }
  pq.clear();

  auto const invert = [&](ch_label::dist_t d) {
    return static_cast<ch_label::dist_t>(kChMaxTravelTime.count() - d);
  };

  auto const const_min_max_dist = static_cast<int>(
      saw<kChSawType>{min_max_dist, ch_traffic_days}.max().count());
  std::cout << "downsearch " << counter << " " << meetpoints.size()
            << std::endl;
  for (auto const m : meetpoints) {
    dists[kForward][m]
        .d_[kMin]
        .to_saw(ch_traffic_days)
        .concat(dists[kReverse][m].d_[kMin].to_saw(ch_traffic_days),
                ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid(), false,
                tmp_saw);
    std::cout << "mp filter " << m << " "
              << tt.get_default_translation(tt.locations_.names_.at(m)) << " "
              << saw<kChSawType>{tmp_saw, ch_traffic_days}.min() << " "
              << saw<kChSawType>{min_max_dist, ch_traffic_days}.max()
              << std::endl;
    if (saw<kChSawType>{tmp_saw, ch_traffic_days} >
            saw<kChSawType>{min_max_dist, ch_traffic_days} ||
        saw<kChSawType>{tmp_saw, ch_traffic_days}.min() >
            saw<kChSawType>{min_max_dist, ch_traffic_days}
                .max()) {  // TODO werid
      tmp_saw.clear();
      continue;
    }
    std::cout << "taken" << std::endl;
    std::cout << "concat " << saw<kChSawType>{tmp_saw, ch_traffic_days}
              << std::endl;
    std::cout << "min_max" << saw<kChSawType>{min_max_dist, ch_traffic_days}
              << std::endl;

    tmp_saw.clear();
    for (auto const dir : {kForward, kReverse}) {
      auto const other_dir = dir ^ 1U;

      saw<kChSawType>{min_max_dist, ch_traffic_days}.simplify(
          dists[dir][m].d_[kMax].to_saw(ch_traffic_days), true, tmp_saw);
      std::swap(dists[dir][m].d_[kMax].saw_, tmp_saw);
      tmp_saw.clear();
      auto const d = std::min(
          static_cast<ch_label::dist_t>(std::max(
              const_min_max_dist - static_cast<int>(dists[other_dir][m]
                                                        .d_[kMin]
                                                        .to_saw(ch_traffic_days)
                                                        .min()
                                                        .count()),
              0)),
          dists[dir][m].d_[kMax].to_saw(ch_traffic_days).max().count());
      pq.push(ch_label{
          m,
          {invert(d), static_cast<ch_label::dist_t>(nonce_map.at(m) + 1)},
          static_cast<std::uint8_t>(dir)});
      std::cout << "added mp " << d << std::endl;
    }
  }
  std::cout << "starting pq" << std::endl;
  auto queue =
      std::queue<std::tuple<ch_edge_idx_t, ch_label::dist_t, bool, bool>>{};
  auto visited = vector_map<ch_edge_idx_t, ch_label::dist_t>{};
  visited.resize(tt.ch_graph_edges_[prf_idx].size());

  while (!pq.empty()) {
    auto l = pq.top();
    auto const l_d_max = static_cast<int>(invert(l.d_[kMax]));
    pq.pop();

    if (l.d_[kMin] <= nonce_map.at(l.l_)) {
      continue;
    }
    std::cout << "down " << l.l_ << " "
              << tt.get_default_translation(tt.locations_.names_.at(l.l_))
              // << " min: " << dists[l.dir_][l.l_].d_[kMin] << " "
              << " max: " << l_d_max << " nonce: " << l.d_[kMin]
              << " dir:" << (l.dir_ == kForward ? "fwd" : "bwd")
              << "| l:" << tt.ch_levels_[prf_idx].at(l.l_) << std::endl;
    nonce_map.at(l.l_) = l.d_[kMin];
    saw<kChSawType>{min_max_dist, ch_traffic_days}.simplify(
        dists[l.dir_][l.l_].d_[kMax].to_saw(ch_traffic_days), true, tmp_saw);
    /*if (saw<kChSawType>{tmp_saw, ch_traffic_days} ==
        dists[l.dir_][l.l_].d_[kMax].to_saw(
            ch_traffic_days)) {  // TODO improve leq pre-pq-push?
      mark_relevant_stop(l.l_);
      tmp_saw.clear();
      continue;
    }*/
    std::swap(dists[l.dir_][l.l_].d_[kMax].saw_,
              tmp_saw);  // TODO min with l_d_max-e.min_dur
    tmp_saw.clear();
    mark_relevant_stop(l.l_);
    auto const& graph = l.dir_ == kReverse ? tt.fwd_search_ch_graph_[prf_idx]
                                           : tt.bwd_search_ch_graph_[prf_idx];

    for (auto const& e_idx : graph[l.l_]) {
      // std::cout << "edge" << e_idx << std::endl;
      auto const e = tt.ch_graph_edges_[prf_idx][e_idx];
      auto const edge_target = l.dir_ == kReverse ? e.to_ : e.from_;
      if (tt.ch_levels_[prf_idx][l.l_] < tt.ch_levels_[prf_idx][edge_target]) {
        continue;
      }
      auto const& prev_label = dists[l.dir_][edge_target];
      if (prev_label.d_[kMin].saw_.empty()) {
        continue;
      }
      std::cout << "down edge " << l_d_max << " "
                << tt.get_default_translation(tt.locations_.names_.at(
                       tt.ch_graph_edges_[prf_idx][e_idx].from_))
                << " -> "
                << tt.get_default_translation(tt.locations_.names_.at(
                       tt.ch_graph_edges_[prf_idx][e_idx].to_))
                << std::endl;
      prev_label.d_[kMin]
          .to_saw(ch_traffic_days)
          .concat(l.dir_,
                  saw<kChSawType>{tt.ch_graph_min_[prf_idx].at(e_idx),
                                  ch_traffic_days},
                  ch_edge_idx_t::invalid(), ch_edge_idx_t::invalid(), false,
                  tmp_saw);
      auto min_dist_via_prev = saw<kChSawType>{tmp_saw, ch_traffic_days};
      if (min_dist_via_prev.max().count() >=
          kChMaxTravelTime.count()) {  // TODO expensive
        std::cout << "weird" << min_dist_via_prev.max().count() << " "
                  << l_d_max << " " << std::endl;
        tmp_saw.clear();
        continue;
      }
      auto min_dist_via_prev_const = min_dist_via_prev.min().count();
      if (min_dist_via_prev_const <= l_d_max &&
          min_dist_via_prev.leq(
              dists[l.dir_][l.l_].d_[kMax].to_saw(ch_traffic_days),
              true)) {  // TODO l_d_max cheat, stopping criterion, cutoff?
        // TODO move to pq pop?

        if (kToothUnpackMode) {
          auto max = false;
          for (auto const& saw :
               {saw<kChSawType>{tt.ch_graph_min_[prf_idx].at(e_idx),
                                ch_traffic_days},
                saw<kChSawType>{tt.ch_graph_max_[prf_idx].at(e_idx),
                                ch_traffic_days}}) {
            for (auto it = saw.begin(); it != saw.end(); ++it) {
              if (it->start_ != ch_edge_idx_t::invalid()) {
                mark_relevant_stop(
                    tt.ch_graph_edges_[prf_idx].at(it->start_).to_);
                queue.push({it->start_, it->start_idx_, max, false});
                if (it->end_ != ch_edge_idx_t::invalid()) {
                  queue.push({it->end_, it->end_idx_, max, true});
                }
              }
            }
            max = true;
          }
        } else {
          queue.push(
              {e_idx,
               std::min(
                   static_cast<ch_label::dist_t>(std::max(
                       l_d_max - static_cast<int>(prev_label.d_[kMin]
                                                      .to_saw(ch_traffic_days)
                                                      .min()
                                                      .count()),
                       0)),
                   static_cast<ch_label::dist_t>(saw<kChSawType>{
                       tt.ch_graph_max_[prf_idx].at(e_idx), ch_traffic_days}
                                                     .max()
                                                     .count())),
               false, false});
        }

        while (!queue.empty()) {
          auto [child_edge_idx, child_max_dur, child_max, child_end] =
              queue.front();
          std::cout << "stack " << child_edge_idx << " cmd: " << child_max_dur
                    << std::endl;

          queue.pop();
          if (!kToothUnpackMode &&
              visited.at(child_edge_idx) >=
                  child_max_dur) {  // TODO use pq ordered by child_max_dur?
            continue;
          }
          visited[child_edge_idx] = child_max_dur;
          auto const child_max_dur_saw = saw<kChSawType>{
              tt.ch_graph_max_[prf_idx].at(child_edge_idx),
              ch_traffic_days};  // TODO at least min with min_max_dur?

          if (kToothUnpackMode) {
            auto const tooth_idx = child_max_dur;  // TODO cleanup

            auto const tooth =
                child_max
                    ? tt.ch_graph_max_[prf_idx].at(child_edge_idx).at(tooth_idx)
                    : tt.ch_graph_min_[prf_idx]
                          .at(child_edge_idx)
                          .at(tooth_idx);  // TODO avoid copy
            if (tooth.start_ != ch_edge_idx_t::invalid()) {
              mark_relevant_stop(
                  tt.ch_graph_edges_[prf_idx].at(tooth.start_).to_);
              std::cout
                  << "ft ldmax" << l_d_max << " "
                  << tt.ch_graph_edges_[prf_idx][child_edge_idx].from_ << " l:"
                  << tt.ch_levels_[prf_idx].at(
                         tt.ch_graph_edges_[prf_idx][child_edge_idx].from_)
                  << " "
                  << tt.get_default_translation(tt.locations_.names_.at(
                         tt.ch_graph_edges_[prf_idx][child_edge_idx].from_))
                  << " -> " << tt.ch_graph_edges_[prf_idx][child_edge_idx].to_
                  << " l:"
                  << tt.ch_levels_[prf_idx].at(
                         tt.ch_graph_edges_[prf_idx][child_edge_idx].to_)
                  << " "
                  << tt.get_default_translation(tt.locations_.names_.at(
                         tt.ch_graph_edges_[prf_idx][child_edge_idx].to_))
                  << " transfer a "
                  << tt.ch_graph_edges_[prf_idx].at(tooth.start_).to_ << " l:"
                  << tt.ch_levels_[prf_idx].at(
                         tt.ch_graph_edges_[prf_idx].at(tooth.start_).to_)
                  << " "
                  << tt.get_default_translation(tt.locations_.names_.at(
                         tt.ch_graph_edges_[prf_idx].at(tooth.start_).to_))
                  << " transfer b "
                  << tt.ch_graph_edges_[prf_idx].at(tooth.end_).from_ << " l:"
                  << tt.ch_levels_[prf_idx].at(
                         tt.ch_graph_edges_[prf_idx].at(tooth.end_).from_)
                  << " "
                  << tt.get_default_translation(tt.locations_.names_.at(
                         tt.ch_graph_edges_[prf_idx].at(tooth.end_).from_))
                  << std::endl;
              std::cout << "stack push " << tooth.start_ << " "
                        << tooth.start_idx_ << " " << tooth.end_ << " "
                        << tooth.end_idx_ << std::endl;
              queue.push({tooth.start_, tooth.start_idx_, child_max, false});
              if (tooth.end_ != ch_edge_idx_t::invalid()) {
                queue.push({tooth.end_, tooth.end_idx_, child_max, true});
              }
            }

          } else {
            for (auto const [unpack, transfer] :
                 utl::zip(tt.ch_graph_unpack_[prf_idx].at(child_edge_idx),
                          tt.ch_graph_transfers_[prf_idx].at(child_edge_idx))) {
              if (unpack.second == ch_edge_idx_t::invalid()) {
                if (transfer != location_idx_t::invalid()) {
                  mark_relevant_stop(transfer);
                }
                continue;
              }
              auto const arr_min_saw = saw<kChSawType>{
                  tt.ch_graph_min_[prf_idx].at(unpack.first), ch_traffic_days};
              auto const dep_min_saw = saw<kChSawType>{
                  tt.ch_graph_min_[prf_idx].at(unpack.second), ch_traffic_days};
              auto const arr_min = arr_min_saw.min().count();
              auto const dep_min = dep_min_saw.min().count();
              auto const arr_max =
                  std::min(static_cast<ch_label::dist_t>(
                               std::max(static_cast<int>(child_max_dur) -
                                            static_cast<int>(dep_min),
                                        0)),
                           static_cast<ch_label::dist_t>(saw<kChSawType>{
                               tt.ch_graph_max_[prf_idx].at(unpack.first),
                               ch_traffic_days}
                                                             .max()
                                                             .count()));
              auto const dep_max =
                  std::min(static_cast<ch_label::dist_t>(
                               std::max(static_cast<int>(child_max_dur) -
                                            static_cast<int>(arr_min),
                                        0)),  // TODO deconcat?
                           static_cast<ch_label::dist_t>(saw<kChSawType>{
                               tt.ch_graph_max_[prf_idx].at(unpack.second),
                               ch_traffic_days}
                                                             .max()
                                                             .count()));

              if (transfer != location_idx_t::invalid()) {
                std::cout
                    << "ft ldmax" << l_d_max << " "
                    << tt.get_default_translation(tt.locations_.names_.at(
                           tt.ch_graph_edges_[prf_idx][child_edge_idx].from_))
                    << " -> "
                    << tt.get_default_translation(tt.locations_.names_.at(
                           tt.ch_graph_edges_[prf_idx][child_edge_idx].to_))
                    << " transfer "
                    << tt.get_default_translation(
                           tt.locations_.names_.at(transfer))
                    << " arr: " << arr_min << " " << arr_max << " "
                    << " dep: " << dep_min << " " << dep_max << std::endl;
              }
              if (arr_min_saw > child_max_dur_saw ||
                  dep_min_saw > child_max_dur_saw || arr_min > arr_max ||
                  dep_min > dep_max) {
                std::cout << "skip" << std::endl;
                continue;  // TODO count occurs
              }
              if (transfer != location_idx_t::invalid()) {
                mark_relevant_stop(transfer);
              }
              queue.push({unpack.first, arr_max, false, false});
              queue.push({unpack.second, dep_max, false, false});
              std::cout << "stack push " << unpack.first << " " << unpack.second
                        << std::endl;
            }
          }
        }
        auto const x = saw<kChSawType>{tt.ch_graph_min_[prf_idx].at(e_idx),
                                       ch_traffic_days}  // TODO deconcat?
                           .min();
        utl::verify(x != u16_minutes::max(), "min is infty");
        std::cout << "diff " << x << " ld " << l_d_max << " em "
                  << dists[l.dir_][edge_target]
                         .d_[kMax]
                         .to_saw(ch_traffic_days)
                         .max()
                         .count()
                  << std::endl;
        auto const diff = std::max(l_d_max - static_cast<int>(x.count()), 0);
        pq.push(ch_label{
            edge_target,
            {static_cast<ch_label::dist_t>(
                 invert(std::min(static_cast<ch_label::dist_t>(diff),
                                 dists[l.dir_][edge_target]
                                     .d_[kMax]
                                     .to_saw(ch_traffic_days)
                                     .max()
                                     .count()))),
             static_cast<ch_label::dist_t>(nonce_map[edge_target] +
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